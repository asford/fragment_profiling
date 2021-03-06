import logging
import copy

from collections import namedtuple

import numpy

from .profile_calculation import extract_logscore_profile_scores, select_by_additive_profile_score

from interface_fragment_matching.fragment_fitting.rmsd_calc import atom_array_broadcast_rmsd
from interface_fragment_matching.fragment_fitting.store import FragmentSpecification

FragmentProfilerParameters = namedtuple("FragmentProfilerParameters", ["fragment_specification", "logscore_substitution_profile", "select_fragments_per_query_position"])

class ProfileFragmentQuality(object):
    logger = logging.getLogger("fragment_profling.profile_fragment_quality.ProfileFragmentQuality")

    aa_codes ='ARNDCEQGHILKMFPSTWYV'
    aa_encoding = dict((aa, i) for i, aa in enumerate(aa_codes))

    def __init__(self, source_residues, logscore_substitution_profile, select_fragments_per_query_position, profiler_benchmark_summaries = {}):
        self.source_residues = source_residues
        self.encoded_source_residue_sequences = self.sequence_array_to_encoding(source_residues["sc"]["aa"])

        if isinstance(logscore_substitution_profile, basestring):
            import Bio.SubsMat.MatrixInfo
            assert logscore_substitution_profile in Bio.SubsMat.MatrixInfo.available_matrices

            self.logscore_substitution_profile_name = logscore_substitution_profile
            self.logscore_substitution_profile_data = Bio.SubsMat.MatrixInfo.__dict__[logscore_substitution_profile]
        else:
            self.logscore_substitution_profile_name = None
            self.logscore_substitution_profile_data = logscore_substitution_profile
        
        self.logscore_substitution_profile_data = self.encode_score_table(self.logscore_substitution_profile_data)
        
        self.select_fragments_per_query_position = select_fragments_per_query_position

        # Cache fragment start indicies over multiple profiling runs.
        self._cached_fragment_start_length = None
        self._cached_fragment_start_indicies = None
    
        self.profiler_benchmark_summaries = dict(profiler_benchmark_summaries)

    def get_fragment_start_residues(self, fragment_length):
        """Get all starting indicies in self.source_residues for the given fragment length."""

        if fragment_length != self._cached_fragment_start_length:
            fspec = FragmentSpecification(fragment_length)
            self._cached_fragment_start_length = fragment_length
            self._cached_fragment_start_indicies = fspec.fragment_start_residues_from_residue_array(self.source_residues)

        return self._cached_fragment_start_indicies
        
    def perform_fragment_analysis(self, fragments):
        """Perform profile-based fragment quality analysis for given fragments.
        
        fragments - array((n,), dtype=[("sc", ("aa", "S1")), ("coordinates", ...)]

        returns - ProfileFragmentQualityResult
        """

        self.logger.info("perform_fragment_analysis(<%s fragments>)", len(fragments))

        fragments = numpy.array(fragments)
        (num_query_fragments,) = fragments.shape

        fspec = FragmentSpecification.from_fragment_dtype(fragments.dtype)

        source_fragment_start_indicies = self.get_fragment_start_residues(fspec.fragment_length)

        # Select dummy fragment with fspec to get dtype
        per_position_selected_fragments = numpy.empty(
                (len(fragments), self.select_fragments_per_query_position),
                dtype = fspec.fragments_from_start_residues(
                    self.source_residues, numpy.array([0]),
                    additional_per_residue_fields=["bb", "sc", "ss"]).dtype)

        per_position_selection_scores = numpy.empty_like(per_position_selected_fragments, dtype=float)
        per_position_selection_rmsds = numpy.empty_like(per_position_selection_scores)
        
        for n in xrange(len(fragments)):
            self.logger.debug("profiling fragment: %s", n)
            frag = fragments[n]

            self.logger.debug("profiling sequence: %s", frag["sc"]["aa"])
            frag_sequence = self.sequence_array_to_encoding(frag["sc"]["aa"])
            frag_profile = self.position_profile_from_sequence(frag_sequence, self.logscore_substitution_profile_data)
            
            self.logger.debug("profile table: %s", frag_profile)

            score_selections = select_by_additive_profile_score(
                    frag_profile,
                    self.encoded_source_residue_sequences,
                    source_fragment_start_indicies,
                    self.select_fragments_per_query_position)

            assert len(score_selections) == self.select_fragments_per_query_position
            

            per_position_selection_scores[n] = score_selections["score"]
            per_position_selected_fragments[n] = fspec.fragments_from_start_residues(
                self.source_residues,
                score_selections["index"],
                additional_per_residue_fields=["bb", "sc", "ss"])
            per_position_selection_rmsds[n] = atom_array_broadcast_rmsd(
                                            fragments[n]["coordinates"],
                                            per_position_selected_fragments[n]["coordinates"])

        search_parameters = FragmentProfilerParameters(fspec, self.logscore_substitution_profile_name, self.select_fragments_per_query_position)

        if search_parameters in self.profiler_benchmark_summaries:
            query_benchmark_data = self.profiler_benchmark_summaries[search_parameters]
            quantile_indicies = numpy.searchsorted(query_benchmark_data["global_quantile_value"], per_position_selection_rmsds)
            quantile_indicies[quantile_indicies >= len(query_benchmark_data["global_quantile_value"])] = len(query_benchmark_data["global_quantile_value"]) - 1
            selected_fragment_quantiles = query_benchmark_data["quantile"][quantile_indicies]
        else:
            if self.profiler_benchmark_summaries:
                # Log warning if profiler has benchmark data but user's query isn't benchmarked.
                self.logger.warning("Query parameters not benchmarked, result quantiles will not be calculated. %s", search_parameters)
            
            selected_fragment_quantiles = None

        return ProfileFragmentQualityResult(fragments, per_position_selected_fragments, per_position_selection_rmsds, selected_fragment_quantiles)

    def profile_fragment_scoring(self, fragments):
        fragments = numpy.array(fragments)
        (num_query_fragments,) = fragments.shape

        fspec = FragmentSpecification.from_fragment_dtype(fragments.dtype)

        source_fragment_start_indicies = self.get_fragment_start_residues(fspec.fragment_length)

        # Pre-allocate result score tables.
        source_fragment_total_scores = numpy.empty((len(fragments), len(source_fragment_start_indicies)), dtype=self.logscore_substitution_profile_data.dtype)

        for n in xrange(len(fragments)):
            frag = fragments[n]
            frag_sequence = self.sequence_array_to_encoding(frag["sc"]["aa"])
            frag_profile = self.position_profile_from_sequence(frag_sequence, self.logscore_substitution_profile_data)
            
            extract_logscore_profile_scores(
                    frag_profile,
                    self.encoded_source_residue_sequences,
                    source_fragment_start_indicies,
                    source_fragment_total_scores[n])

        return source_fragment_total_scores
        
    def position_profile_from_sequence(self, input_sequence, score_table):
        """Create position specific profile table from input sequence and score table.
        
        input_sequence - array-like((sequence_length), int) integer encoded input sequence.
        score_table - array((n, n)) score table.
        
        returns - array((sequence_length, n)) position specific score.
        """
        assert score_table.ndim == 2 and score_table.shape[0] == score_table.shape[1]
    
        profile = numpy.zeros((len(input_sequence), score_table.shape[1]), dtype=score_table.dtype)
    
        for i in xrange(len(input_sequence)):
            profile[i] = score_table[input_sequence[i]]
    
        return profile

    def encode_score_table(self, score_table):
        """Convert pairwise score table to integer sequence encoding.
        
        score_table - {(from, to) : score, [...]} dictionary of score table entries.
            
            score_table is assumed to be pre-encoded if array input provided.

        returns - array((encoding_size, encoding_size), float) score table.
        """
        
        if isinstance(score_table, numpy.ndarray):
            return score_table.copy()
        
        result = numpy.zeros((len(self.aa_encoding), len(self.aa_encoding)), dtype=float)
    
        for (f, t), v in score_table.items():
            if f in self.aa_encoding and t in self.aa_encoding:
                result[self.aa_encoding[f], self.aa_encoding[t]] = v
                result[self.aa_encoding[t], self.aa_encoding[f]] = v
    
        return result
    
    def sequence_array_to_encoding(self, sequence_array):
        """Convert sequence array to integer encoding."""
    
        result = numpy.empty_like(sequence_array, dtype="u2")
        result[:] = max(self.aa_encoding.values()) + 1
    
        for aa, i in self.aa_encoding.items():
            result[sequence_array == aa] = i
    
        assert numpy.alltrue(result != max(self.aa_encoding.values()) + 1)
        return result

    def encoding_to_sequence_array(self, encoding_array):
        """Convert encoded aa array to sequence array."""
        
        result = numpy.empty_like(encoding_array, dtype="S1")
        
        result[:] = "."
        for aa, i in self.aa_encoding.items():
            result[encoding_array == i] = aa
            
        assert numpy.alltrue(result != ".")
        
        return result

class ProfileFragmentQualityResult(namedtuple("ProfileFragmentQualityResult", ["query_fragments", "selected_fragments", "selected_fragment_rmsds", "selected_fragment_quantiles"])):
    """Result container for fragment profiling runs."""
    
    #def __new__(_cls, query_fragments, selected_fragments, selected_fragment_rmsds, selected_fragment_quantiles = None):
        #"""Create new result tuple, defaulting to no selected_fragment_quantiles."""
        #return super(FragmentSpecification, _cls).__new__(_cls, query_fragments, selected_fragments, selected_fragment_rmsds, selected_fragment_quantiles)
    @property
    def result_fragment_count(self):
        return self.selected_fragments.shape[1]

    @property
    def query_fragment_count(self):
        return self.selected_fragments.shape[0]

    def generate_result_summary(self, fragment_count = 0):
        """Generate descriptive summary of result, optionally including top fragment_count fragments.
        
        returns - Summary array fields:
            "id" - query id
            "resn" - query resn
            "count" - total fragment count
            "mean" - mean fragment rmsd
            "std" - fragment rmsd stddev
            "quartile" - 0, .25, .5, .75, 1.0 quantiles of result rmsds
            ["selected_fragments" - top fragments]
            ["fragment_quantile" - fragment profile quantile]

        """ 

        result_dtype = [
                    ("id", "u4"), ("resn", "u4"),
                    ("count", "u4"), ("mean", float), ("std", float), ("quartile", float, (5,))]

        if fragment_count and fragment_count > 0:
            result_dtype += [("selected_fragments", self.selected_fragments.dtype, (fragment_count,))]

        if self.selected_fragment_quantiles is not None:
            result_dtype += [("fragment_quantile", float)]
            
        result = numpy.empty_like(
                self.query_fragments,
                result_dtype)

        result["id"] = self.query_fragments["id"]
        result["resn"] = self.query_fragments["resn"]
        result["count"] = self.selected_fragments.shape[-1]

        numpy.mean(self.selected_fragment_rmsds, axis=-1, out = result["mean"])
        numpy.std(self.selected_fragment_rmsds, axis=-1, out = result["std"])
        qr = numpy.percentile(self.selected_fragment_rmsds, [0., 25., 50., 75., 100.], axis=-1)
        for q in xrange(len(qr)):
            result["quartile"][:,q] = qr[q]

        if fragment_count and fragment_count > 0:
            fragment_selections = numpy.argsort(self.selected_fragment_rmsds)[...,:fragment_count]
            idx = (numpy.expand_dims(numpy.arange(fragment_selections.shape[0]), -1), fragment_selections)
            result["selected_fragments"] = self.selected_fragments[idx]

        if self.selected_fragment_quantiles is not None:
            result["fragment_quantile"] = numpy.min(self.selected_fragment_quantiles, axis=-1)

        return result

    def restrict_to_top_fragments(self, fragment_count):
        """Return result subset corrosponding to top fragment_count fragments, as ranked by rmsd."""

        if fragment_count >= self.selected_fragments.shape[1]:
            return copy.deepcopy(self)

        fragment_selections = numpy.argsort(self.selected_fragment_rmsds)[...,:fragment_count]
        # Create broadcasted selection indicies for fragment selections
        idx = (numpy.expand_dims(numpy.arange(fragment_selections.shape[0]), -1), fragment_selections)
        
        return ProfileFragmentQualityResult(self.query_fragments.copy(), self.selected_fragments[idx], self.selected_fragment_rmsds[idx], self.selected_fragment_quantiles[idx] if not self.selected_fragment_quantiles is None else None)

    def prune_fragments_by_start_residue(self):
        """Remove result fragments with identical id/resn as query fragment.
        
        Reduces size of result set by amount needed to remove matching fragments.
        """
        from interface_fragment_matching.structure_database.store import ResidueCache

        matching_start_residue_mask = \
            numpy.expand_dims(ResidueCache.residue_unique_id(self.query_fragments), -1) == \
            ResidueCache.residue_unique_id(self.selected_fragments)

        max_num_duplicates = max(numpy.sum(matching_start_residue_mask, axis=-1))

        # Set fragments with matching start residue to infinite RMSD and select subset of fragments
        # by rmsd.
        work_copy = copy.deepcopy(self)
        work_copy.selected_fragment_rmsds[matching_start_residue_mask] = numpy.inf

        return work_copy.restrict_to_top_fragments(self.result_fragment_count - max_num_duplicates)

    def plot_per_position_fragment_analysis(self, position, target_axis = None):
        from matplotlib import pylab

        position = int(position)
        
        if position < 0 or position >= self.selected_fragment_rmsds.shape[0]:
            raise ValueError("Invalid fragment position specified.")
            
        if target_axis is None:
            target_axis = pylab.gca()

        fragment_rmsds = numpy.sort(self.selected_fragment_rmsds[position], axis=-1)

        target_axis.set_title("Fragment %s rmsd distribution." % position)
        target_axis.hist(fragment_rmsds, bins=50, normed=True)
        target_axis.grid(False, axis="y")
        target_axis.set_xlabel("RMSD")
        target_axis = target_axis.twinx()
        target_axis.set_yscale("symlog")
        target_axis.plot(fragment_rmsds, pylab.arange(len(fragment_rmsds)), label="Fragment count at RMSD.", color="red")

        target_axis.legend()
        
        return target_axis
    
    def plot_all_fragment_analysis(self, target_axis = None):
        from matplotlib import pylab

        if target_axis is None:
            fig = pylab.figure()
            target_axis = fig.gca()
            
        target_axis.set_title("Per-position fragment profile.")
        target_axis.set_xlabel("Position")
        target_axis.set_ylabel("Fragment RMSD distribution.")
        target_axis.boxplot(self.selected_fragment_rmsds.T)
        
        if self.selected_fragment_quantiles is not None:
            rs = self.generate_result_summary()
            ax2 = target_axis.twinx()
            ax2.plot(
                         numpy.arange(len(rs)) + 1,
                         rs["fragment_quantile"],
                         color="red", alpha=.7, linewidth=4, label="Fragment benchmark quantile.")
            ax2.set_ylim(0, 1)
            ax2.grid("off", axis="y")
            ax2.set_ylabel("Benchmark quantile.")

            ax2.legend(loc="best")
