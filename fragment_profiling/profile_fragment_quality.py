import numpy
import pylab

from .profile_calculation import extract_profile_scores

from interface_fragment_matching.fragment_fitting.rmsd_calc import atom_array_broadcast_rmsd

class ProfileFragmentQuality(object):
    aa_codes ='ARNDCEQGHILKMFPSTWYV'
    aa_encoding = dict((aa, i) for i, aa in enumerate(aa_codes))

    def __init__(self, source_fragments, logscore_substitution_profile, select_fragments_per_position=200):
        self.source_fragments = source_fragments
        self.encoded_source_fragment_sequences = self.sequence_array_to_encoding(source_fragments["sc"]["aa"])
        
        self.logscore_substitution_profile = self.encode_score_table(logscore_substitution_profile)
        
        self.select_fragments_per_position = select_fragments_per_position
        self.target_score_quantile = 1.0 - (float(select_fragments_per_position) / len(source_fragments))
        
        self.per_position_selected_fragments = None
        self.per_position_selection_rmsds = None
        
    def perform_fragment_analysis(self, fragments):
        """Perform profile-based fragment quality analysis for given fragments.
        
        fragments - array((n,), dtype=[("sc", ("aa", "S1")), ("coordinates", ...)]
        """
        
        num_fragments = len(fragments)
        
        # Pre-allocating result score tables.
        source_fragment_position_scores = numpy.empty_like(self.encoded_source_fragment_sequences, dtype=self.logscore_substitution_profile.dtype)
        source_fragment_total_scores = numpy.empty(self.encoded_source_fragment_sequences.shape[:-1], dtype=self.logscore_substitution_profile.dtype)
        
        self.per_position_selected_fragments = numpy.empty((len(fragments), self.select_fragments_per_position), dtype=self.source_fragments.dtype)
        self.per_position_selection_rmsds = numpy.empty_like(self.per_position_selected_fragments, dtype=float)
        
        for n in xrange(len(fragments)):
            frag = fragments[n]
            frag_sequence = self.sequence_array_to_encoding(frag["sc"]["aa"])
            frag_profile = self.position_profile_from_sequence(frag_sequence, self.logscore_substitution_profile)
            
            extract_profile_scores(frag_profile, self.encoded_source_fragment_sequences, source_fragment_position_scores)
            numpy.sum(source_fragment_position_scores, axis=-1, out=source_fragment_total_scores)
            
            cutoff_score = numpy.percentile(source_fragment_total_scores, self.target_score_quantile * 100)
            
            selected_fragments = self.source_fragments[source_fragment_total_scores >= cutoff_score][:self.select_fragments_per_position]
            selection_rmsd = atom_array_broadcast_rmsd(frag["coordinates"], selected_fragments["coordinates"])
            
            fragment_ordering = numpy.argsort(selection_rmsd)
            self.per_position_selected_fragments[n] = selected_fragments[fragment_ordering]
            self.per_position_selection_rmsds[n] = selection_rmsd[fragment_ordering]
    
    def plot_per_position_fragment_analysis(self, position, target_axis = None):
        position = int(position)
        
        if position < 0 or position >= self.per_position_selection_rmsds.shape[0]:
            raise ValueError("Invalid fragment position specified.")
            
        if target_axis is None:
            fig = pylab.figure()
            target_axis = fig.gca()
        
        target_axis.set_title("Fragment %s rmsd distribution." % position)
        target_axis.hist(self.per_position_selection_rmsds[position], bins=50, normed=True)
        target_axis.grid(False, axis="y")
        target_axis.set_xlabel("RMSD")
        target_axis = target_axis.twinx()
        target_axis.set_yscale("symlog")
        target_axis.plot(self.per_position_selection_rmsds[position], pylab.arange(len(self.per_position_selection_rmsds[position])), label="Fragment count at RMSD.", color="red")

        target_axis.legend()
        
        return target_axis
    
    def plot_all_fragment_analysis(self, target_axis = None):
        if target_axis is None:
            fig = pylab.figure()
            target_axis = fig.gca()
            
        result_rmsds = numpy.array(len(self.per_position_selection_rmsds))
        
        target_axis.set_title("Per-position fragment RMSD profile.")
        target_axis.set_xlabel("Position")
        target_axis.set_ylabel("RMSD")
        target_axis.boxplot(self.per_position_selection_rmsds.T)
        target_axis.plot(
                         numpy.arange(self.per_position_selection_rmsds.shape[0]) + 1,
                         self.per_position_selection_rmsds[:,1],
                         color="red", label="Minimum fragment rmsd.")
        target_axis.legend()
        
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
