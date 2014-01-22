import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")

import itertools

import jug
from jug import Task, TaskGenerator, Tasklet, bvalue, barrier
from jug.compound import CompoundTask

import numpy

import tables

import rosetta

from interface_fragment_matching.parallel.utility import map_partitions

from interface_fragment_matching.structure_database.store import StructureDatabase
from interface_fragment_matching.fragment_fitting.store import FragmentSpecification

from fragment_profiling.tasks import BenchmarkProfileFragmentQualityTask

from fragment_profiling.store import FragmentProfilingDatabase
from fragment_profiling.profile_fragment_quality import FragmentProfilerParameters

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
    result_summary = agglomerative_reduction_task(reduce_result_summary, 100, profiler_results)

    return result_summary

def generate_quantile_summary(result_summary_table, quantiles = numpy.linspace(0, 1, 101)):
    from scipy.stats.mstats import mquantiles
    import pandas
    
    summary_table = pandas.DataFrame.from_items([("id", result_summary_table["id"]), ("rmsd", result_summary_table["quartile"][...,0])])
    
    result = numpy.empty_like(quantiles, dtype=[("quantile", float), ("global_quantile_value", float), ("worst_per_structure_quantile_value", float)])
    result["quantile"] = quantiles
    result["global_quantile_value"] = mquantiles(summary_table["rmsd"].values, quantiles)
    result["worst_per_structure_quantile_value"] = mquantiles(summary_table.groupby("id")["rmsd"].max().values, quantiles)
    
    return result

def dict_element_product(options_dict):
    ks = options_dict.keys()
    return [tuple(zip(ks, vs)) for vs in itertools.product(*[options_dict[k] for k in ks])]

profile_source_database = "/work/fordas/test_sets/vall_store.h5"
target_structure_database = "/work/fordas/test_sets/vall_store.h5"
target_ids = Task(read_structure_ids, target_structure_database)
keep_top_fragments_per_query_position = 5

initial_candidate_parameter_values = dict(
    logscore_substitution_profile = ('blosum100',),
    select_fragments_per_query_position = (300, 200),
    fragment_specification = (FragmentSpecification(9, "CA"), FragmentSpecification(9, ("N", "CA", "C")))
)

query_size_sweep_parameter_values = dict(
    logscore_substitution_profile = ('blosum100',),
    select_fragments_per_query_position = (10, 20, 50, 100, 200, 400, 500, 1000),
    fragment_specification = (FragmentSpecification(9, "CA"), ))

parameter_keys = initial_candidate_parameter_values.keys()
logging.info("parameter_keys: %s", parameter_keys)

parameter_sets = set(dict_element_product(initial_candidate_parameter_values) + dict_element_product(query_size_sweep_parameter_values))

result_summaries = []
quantile_summaries = []

for parameter_values in sorted(parameter_sets):
    input_parameter_values = dict(parameter_values)
    logging.info("parameter_values: %s", input_parameter_values)

    final_result_summary =  CompoundTask(
                                profile_structure_collection,
                                target_structure_database,
                                target_ids,
                                input_parameter_values["fragment_specification"],
                                profile_source_database,
                                input_parameter_values["logscore_substitution_profile"],
                                input_parameter_values["select_fragments_per_query_position"],
                                keep_top_fragments_per_query_position)

    result_summaries.append((parameter_values, final_result_summary))

    barrier()

    quantile_summaries.append((parameter_values, Task(generate_quantile_summary, final_result_summary)))

def write_summary_store(store_name, collection_name, target_residue_name, quantile_summaries):
    quantile_summaries = dict(( FragmentProfilerParameters(**dict(q)), v) for q, v in quantile_summaries)

    with FragmentProfilingDatabase(tables.open_file(store_name, "w")) as profile_db:
        profile_db.setup()
        profile_db.add_profiling_benchmark(collection_name, "/work/fordas/test_sets/vall_store.h5:/residues", quantile_summaries)

Task(write_summary_store, "vall_store_fragment_profiling.h5", "vall_benchmarking", "%s:/residues" % profile_source_database, quantile_summaries)
