import logging

from collections import namedtuple

import numpy
import pylab

from .profile_calculation import extract_logscore_profile_scores, select_by_additive_profile_score

from interface_fragment_matching.fragment_fitting.rmsd_calc import atom_array_broadcast_rmsd
from interface_fragment_matching.fragment_fitting.store import FragmentSpecification

class ProfileFragmentQuality(object):
    logger = logging.getLogger("fragment_profling.profile_fragment_quality.ProfileFragmentQuality")

    aa_codes ='ARNDCEQGHILKMFPSTWYV'
    aa_encoding = dict((aa, i) for i, aa in enumerate(aa_codes))

    def __init__(self, source_residues, logscore_substitution_profile, select_fragments_per_query_position=200):
        self.source_residues = source_residues
        self.encoded_source_residue_sequences = self.sequence_array_to_encoding(source_residues["sc"]["aa"])
        
        self.logscore_substitution_profile = self.encode_score_table(logscore_substitution_profile)
        
        self.select_fragments_per_query_position = select_fragments_per_query_position

        # Cache fragment start indicies over multiple profiling runs.
        self._cached_fragment_start_length = None
        self._cached_fragment_start_indicies = None

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
            self.logger.info("profiling fragment: %s", n)
            frag = fragments[n]

            self.logger.info("profiling sequence: %s", frag["sc"]["aa"])
            frag_sequence = self.sequence_array_to_encoding(frag["sc"]["aa"])
            frag_profile = self.position_profile_from_sequence(frag_sequence, self.logscore_substitution_profile)
            
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

        return ProfileFragmentQualityResult(fragments, per_position_selected_fragments, per_position_selection_rmsds)

    def profile_fragment_scoring(self, fragments):
        fragments = numpy.array(fragments)
        (num_query_fragments,) = fragments.shape

        fspec = FragmentSpecification.from_fragment_dtype(fragments.dtype)

        source_fragment_start_indicies = self.get_fragment_start_residues(fspec.fragment_length)

        # Pre-allocate result score tables.
        source_fragment_total_scores = numpy.empty((len(fragments), len(source_fragment_start_indicies)), dtype=self.logscore_substitution_profile.dtype)

        for n in xrange(len(fragments)):
            frag = fragments[n]
            frag_sequence = self.sequence_array_to_encoding(frag["sc"]["aa"])
            frag_profile = self.position_profile_from_sequence(frag_sequence, self.logscore_substitution_profile)
            
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

class ProfileFragmentQualityResult(namedtuple("ProfileFragmentQualityResultTuple", ["query_fragments", "selected_fragments", "selected_fragment_rmsds"])):
    """Result container for fragment profiling runs."""

    def plot_per_position_fragment_analysis(self, position, target_axis = None):
        position = int(position)
        
        if position < 0 or position >= self.selected_fragment_rmsds.shape[0]:
            raise ValueError("Invalid fragment position specified.")
            
        if target_axis is None:
            fig = pylab.figure()
            target_axis = fig.gca()

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
        if target_axis is None:
            fig = pylab.figure()
            target_axis = fig.gca()
            
        result_rmsds = numpy.sort(self.selected_fragment_rmsds, axis=-1)
        
        target_axis.set_title("Per-position fragment RMSD profile.")
        target_axis.set_xlabel("Position")
        target_axis.set_ylabel("RMSD")
        target_axis.boxplot(result_rmsds.T)
        target_axis.plot(
                         numpy.arange(len(result_rmsds)) + 1,
                         result_rmsds[:,1],
                         color="red", label="Minimum fragment rmsd.")
        target_axis.legend()
