# distutils: language = c++

import cython
cimport cython

import numpy
cimport numpy

cdef extern from "ProfileCalculator.hpp" namespace "fragment_profiling":
    cdef cppclass ProfileCalculator[Real, AlphabetIntegralType, Index]:
      void extract_additive_profile_scores(
          Real* input_profile,
          int sequence_length,
          int alphabet_size,
          AlphabetIntegralType* source_sequences,
          Index* source_start_indicies,
          int num_sequences,
          Real* outscore) nogil

      int select_by_additive_profile_score(
          Real* input_profile,
          int sequence_length,
          int alphabet_size,
          AlphabetIntegralType* source_sequences,
          Index* source_start_indicies,
          int num_sequences,
          Index* result_indicies,
          Real* result_scores,
          size_t result_count) nogil

ctypedef fused score_type:
    cython.float
    cython.double
    cython.short
    cython.int
    cython.long

ctypedef fused index:
    #cython.uhar
    cython.uchar
    #cython.short
    cython.ushort
    #cython.int
    #cython.uint
    #cython.long
    #cython.ulong
    
@cython.boundscheck(False)
def extract_logscore_profile_scores(numpy.ndarray[score_type, ndim=2] input_profile, index[:] source_sequences, cython.integral[:] source_start_indicies, numpy.ndarray[score_type, ndim=1] out):
    """Extract profile matrix scores for given collection of sequences.
    
    input_profile - shape (sequence_length, alphabet_size) profile scores.
    sequences - shape (n, sequence_length) sequence entries on the range [0, alphabet_size).
    
    returns - profile_scores array((n, sequence_length), float) profile scores.
    """
    
    cdef int num_sequences = source_start_indicies.shape[0]
    cdef int sequence_length = input_profile.shape[0]
    cdef int alphabet_size = input_profile.shape[1]
    
    if out is None:
        out = numpy.empty(num_sequences, dtype=input_profile.dtype)

    assert out.shape[0] == num_sequences
    
    #assert numpy.max(sequences) < alphabet_size
    #assert numpy.min(sequences) >= 0

    cdef ProfileCalculator[score_type, index, cython.integral] calc
    
    cdef int n = 0, p = 0, sequence_start_index = 0

    calc.extract_additive_profile_scores(
            &input_profile[0,0],
            sequence_length,
            alphabet_size,
            &source_sequences[0],
            &source_start_indicies[0],
            num_sequences,
            &out[0])

    return out

def select_by_additive_profile_score(numpy.ndarray[score_type, ndim=2] input_profile, numpy.ndarray[index, ndim=1] source_sequences, numpy.ndarray[cython.integral, ndim=1] source_start_indicies, int result_count):
    """Extract profile matrix scores for given collection of sequences.
    
    input_profile - shape (sequence_length, alphabet_size) profile scores.
    sequences - shape (n, sequence_length) sequence entries on the range [0, alphabet_size).
    
    returns - profile_scores array((n, sequence_length), float) profile scores.
    """
    
    cdef int num_sequences = source_start_indicies.shape[0]
    cdef int sequence_length = input_profile.shape[0]
    cdef int alphabet_size = input_profile.shape[1]
    
    cdef numpy.ndarray[cython.integral, ndim=1] result_indicies = numpy.empty(result_count, dtype=source_start_indicies.dtype)
    cdef numpy.ndarray[score_type, ndim=1] result_scores = numpy.empty(result_count, dtype=input_profile.dtype)
    
    #assert numpy.max(sequences) < alphabet_size
    #assert numpy.min(sequences) >= 0

    cdef ProfileCalculator[score_type, index, cython.integral] calc
    
    total_results = calc.select_by_additive_profile_score(
            &input_profile[0,0],
            sequence_length,
            alphabet_size,
            &source_sequences[0],
            &source_start_indicies[0],
            num_sequences,
            &result_indicies[0],
            &result_scores[0],
            result_count)
    
    result = numpy.empty(total_results, [("index", int), ("score", "f8")])
    result["index"] = result_indicies[:total_results]
    result["score"] = result_scores[:total_results]

    return result
