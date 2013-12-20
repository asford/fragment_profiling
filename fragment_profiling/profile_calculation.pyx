# distutils: language = c
import cython
cimport cython

import numpy
cimport numpy

ctypedef fused index:
    cython.char
    cython.uchar
    cython.uint
    cython.int
    cython.ushort
    cython.short
    cython.long
    cython.ulong
    
@cython.boundscheck(False)
def extract_profile_scores(cython.floating[:,:] input_profile, index[:,:] sequences, cython.floating[:, :] out):
    """Extract profile matrix scores for given collection of sequences.
    
    input_profile - shape (sequence_length, alphabet_size) profile scores.
    sequences - shape (n, sequence_length) sequence entries on the range [0, alphabet_size).
    
    returns - profile_scores array((n, sequence_length), float) profile scores.
    """
    
    cdef int num_sequences = sequences.shape[0]
    cdef int sequence_length = sequences.shape[1]
    
    assert out.shape[0] == sequences.shape[0]
    assert out.shape[1] == sequences.shape[1]
        
    assert input_profile.shape[0] == sequence_length
    cdef int alphabet_size = input_profile.shape[1]
    
    assert numpy.max(sequences) < alphabet_size
    assert numpy.min(sequences) >= 0
    
    cdef int n, p
    for n in range(num_sequences):
        for p in range(sequence_length):
            out[n, p] = input_profile[p, sequences[n,p]]
    
    return out
