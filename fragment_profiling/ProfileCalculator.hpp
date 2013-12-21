#ifndef _fragment_fitting_rmsd_calc_QCP_Kernal_HPP_
#define _fragment_fitting_rmsd_calc_QCP_Kernal_HPP_

#include <cstdlib>

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace fragment_profiling
{

template <class Real, class AlphabetIntegralType, class Index>
class ProfileCalculator
{
  public:

	ProfileCalculator() {}
	virtual ~ProfileCalculator() {}

  int select_by_additive_profile_score(
      Real* input_profile,
      int sequence_length,
      int alphabet_size,
      AlphabetIntegralType* source_sequences,
      Index* source_start_indicies,
      int num_sequences,
      Index* result_indicies,
      Real* result_scores,
      int result_count)
  {
    typedef typename std::pair<Real, Index> ScorePair;
    typedef typename std::priority_queue< ScorePair, std::vector<ScorePair>, std::greater<ScorePair> > ResultQueue;

    ResultQueue global_results;

    #pragma omp parallel
    {
      ResultQueue local_results;

      #pragma omp for schedule(static)
      for (int n = 0; n < num_sequences; n++)
      {
        ScorePair index_score(0, source_start_indicies[n]);

        for (int p = 0; p < sequence_length; p++)
        {
          index_score.first += input_profile[(p * alphabet_size) + source_sequences[source_start_indicies[n] + p]];
        }

        if (local_results.empty() || index_score > local_results.top())
        {
          local_results.push(index_score);

          while (local_results.size() > result_count)
          {
            local_results.pop();
          }
        }
      }

      #pragma omp critical
      {
        while (!local_results.empty())
        {
          if(global_results.size() < result_count  || local_results.top() > global_results.top())
          {
            global_results.push(local_results.top());

            while (global_results.size() > result_count)
            {
              global_results.pop();
            }
          }

          local_results.pop();
        }
      }
    }
    
    int final_result_count;
    for (final_result_count = 0; !global_results.empty(); final_result_count++)
    {
      result_indicies[final_result_count] = global_results.top().second;
      result_scores[final_result_count] = global_results.top().first;
      global_results.pop();
    }

    return final_result_count;
  }

  void extract_additive_profile_scores(
      Real* input_profile,
      int sequence_length,
      int alphabet_size,
      AlphabetIntegralType* source_sequences,
      Index* source_start_indicies,
      int num_sequences,
      Real* outscore)
  {
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < num_sequences; n++)
    {
      outscore[n] = 0;

      for (int p = 0; p < sequence_length; p++)
      {
        outscore[n] += input_profile[(p * alphabet_size) + source_sequences[source_start_indicies[n] + p]];
      }
    }
  }
};

}

#endif
