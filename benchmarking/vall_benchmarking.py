import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")

import itertools

import jug
from jug import Task, TaskGenerator, Tasklet, bvalue
from jug.compound import CompoundTask

import numpy

import rosetta

from interface_fragment_matching.parallel.utility import map_partitions

from interface_fragment_matching.structure_database.store import StructureDatabase
from interface_fragment_matching.fragment_fitting.store import FragmentSpecification

from fragment_profiling.tasks import BenchmarkProfileFragmentQualityTask

def read_structure_ids(db):
    return StructureDatabase(db).structures.read()["id"]
    
def agglomerative_reduction_task(reduction_function, maximum_reduction_entries, sub_values):
    if len(sub_values) <= maximum_reduction_entries:
        return Task(reduction_function, sub_values)

    splits = range(0, len(sub_values), maximum_reduction_entries)
    subresults = [agglomerative_reduction_task(reduction_function, maximum_reduction_entries, sub_values[s:e]) for s, e in zip(splits, splits[1:] + [len(sub_values)])]

    return Task(reduction_function, subresults)

def agglomerative_reduce(reduction_function, maximum_reduction_entries, values):
    return CompoundTask(agglomerative_reduction_task, reduction_function, maximum_reduction_entries, values)

def extract_result_summary_table(result):
    summary_table = numpy.empty(
        result.query_fragments.shape,
        dtype =
            result.query_fragments[["id", "resn"]].dtype.descr +
            [("lookup_rmsd", result.selected_fragment_rmsds.dtype, result.selected_fragment_rmsds[0].shape)])
    
    summary_table["id"] = result.query_fragments["id"]
    summary_table["resn"] = result.query_fragments["resn"]
    summary_table["lookup_rmsd"] = numpy.sort(result.selected_fragment_rmsds, axis=-1)
    
    return summary_table

# Define function so jug status show more informative name than 'concatenate'
def reduce_result_summary(summary_data):
    return numpy.concatenate(summary_data)

def profile_structure_collection(target_structure_database, target_ids, fragment_specification, profile_source_database, logscore_substitution_profile, select_fragments_per_query_position, keep_top_fragments_per_query_position):
    # Add one to result set size to accommodate fragment pruning.
    profiler_task = BenchmarkProfileFragmentQualityTask(target_structure_database, fragment_specification, profile_source_database, logscore_substitution_profile, select_fragments_per_query_position, keep_top_fragments_per_query_position)

    profiler_results = map_partitions(profiler_task, target_ids, 4000)

    result_summary = agglomerative_reduction_task(reduce_result_summary, 100, [Tasklet(r, extract_result_summary_table) for r in profiler_results])

    return result_summary

profile_source_database = "/work/fordas/test_sets/vall_store.h5"
target_structure_database = "/work/fordas/test_sets/vall_store.h5"
target_ids = Task(read_structure_ids, target_structure_database)
keep_top_fragments_per_query_position = 5

candidate_parameter_values = dict(
    logscore_substitution_profile = ('blosum100',"blosum62"),
    select_fragments_per_query_position = (300, 200),
    fragment_specification = (FragmentSpecification(9, "CA"), FragmentSpecification(9, ("N", "CA", "C")))
)
logging.info("candidate_parameter_values: %s", candidate_parameter_values)

parameter_keys = candidate_parameter_values.keys()
logging.info("parameter_keys: %s", parameter_keys)

result_summaries = []
for parameter_values in itertools.product(*[candidate_parameter_values[k] for k in parameter_keys]):
    logging.info("parameter_values: %s", parameter_values)
    parameter_values = dict(zip(parameter_keys, parameter_values))
    logging.info("parameter_values: %s", parameter_values)

    final_result_summary =  CompoundTask(
                                profile_structure_collection,
                                target_structure_database,
                                target_ids,
                                parameter_values["fragment_specification"],
                                profile_source_database,
                                parameter_values["logscore_substitution_profile"],
                                parameter_values["select_fragments_per_query_position"],
                                keep_top_fragments_per_query_position)

    result_summaries.append((parameter_values, final_result_summary))
