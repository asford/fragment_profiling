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
def extract_logscore_profile_scores(cython.floating[:,:] input_profile, index[:] source_sequences, cython.integral[:] source_start_indicies, out):
    """Extract profile matrix scores for given collection of sequences.
    
    input_profile - shape (sequence_length, alphabet_size) profile scores.
    sequences - shape (n, sequence_length) sequence entries on the range [0, alphabet_size).
    
    returns - profile_scores array((n, sequence_length), float) profile scores.
    """
    
    cdef int num_sequences = source_start_indicies.shape[0]
    cdef int sequence_length = input_profile.shape[0]
    cdef int alphabet_size = input_profile.shape[1]
    
    cdef cython.floating[:] result

    if out is not None:
        result = out
    else:
        if cython.floating is float:
            result = numpy.empty(num_sequences, "f4")
        else:
            result = numpy.empty(num_sequences, "f8")

    assert result.shape[0] == num_sequences
    
    #assert numpy.max(sequences) < alphabet_size
    #assert numpy.min(sequences) >= 0

    cdef ProfileCalculator[cython.floating, index, cython.integral] calc
    
    cdef int n = 0, p = 0, sequence_start_index = 0

    calc.extract_additive_profile_scores(
            &input_profile[0,0],
            sequence_length,
            alphabet_size,
            &source_sequences[0],
            &source_start_indicies[0],
            num_sequences,
            &result[0])

    return result

def select_by_additive_profile_score(double[:,:] input_profile, index[:] source_sequences, long[:] source_start_indicies, int result_count):
    """Extract profile matrix scores for given collection of sequences.
    
    input_profile - shape (sequence_length, alphabet_size) profile scores.
    sequences - shape (n, sequence_length) sequence entries on the range [0, alphabet_size).
    
    returns - profile_scores array((n, sequence_length), float) profile scores.
    """
    
    cdef int num_sequences = source_start_indicies.shape[0]
    cdef int sequence_length = input_profile.shape[0]
    cdef int alphabet_size = input_profile.shape[1]
    
    cdef numpy.ndarray[long, ndim=1] result_indicies = numpy.empty(result_count, long)
    cdef numpy.ndarray[double, ndim=1] result_scores

    if double is float:
        result_scores = numpy.empty(result_count, "f4")
    else:
        result_scores = numpy.empty(result_count, "f8")
    
    #assert numpy.max(sequences) < alphabet_size
    #assert numpy.min(sequences) >= 0

    cdef ProfileCalculator[double, index, long] calc
    
    cdef long n = 0, p = 0, total_results = 0

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
