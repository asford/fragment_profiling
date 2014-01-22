import logging

import os
from os import path

import itertools

import tables

class FragmentProfilingDatabase(object):
    logger = logging.getLogger("FragmentProfilingDatabase")

    """Table-backed fragment profiling database. Stores residue reference and fragment profiling benchmarking data.
    
    Contains table entries:
        /fragment_profiling_benchmarks/
    """

    def __init__(self, store):
        """Load fragment database view over store."""

        self.store = None

        if isinstance(store, basestring):
            self.logger.info("Opening file: %s", store)
            try:
                is_store = tables.is_pytables_file(store)
            except tables.HDF5ExtError:
                is_store = False

            if is_store:
                if os.access(store, os.W_OK):
                    self.store = tables.open_file(store, "r+")
                else:
                    self.store = tables.open_file(store, "r")
            else:
                raise ValueError("Store is not a PyTables file: %s" % store)
        elif isinstance(store, tables.file.File):
            self.logger.info("Opening : %s", store)
            self.store = store
        else:
            raise ValueError("Unable to open store: %s" % store)

    def __hash__(self):
        return hash(self.store.filename)

    def __repr__(self):
        return "%s(store=<%s>)" % (
                self.__class__.__name__,
                self.store.filename)

    def __enter__(self):
        """Support for with statement."""
        return self

    def __exit__(self, *args, **kwargs):
        """Support for with statement."""
        self.close()

    def close(self):
        """Close database handle and underlying store."""
        if self.store is not None:
            self.logger.debug("Closing store: %s", self.store)
            self.store.close()
    
    def is_setup(self):
        """Check if database is setup with standard paths."""
        return ("/fragment_profiling_benchmarks" in self.store)

    def setup(self):
        """Create core tables and groups within store."""

        self.logger.info("Setting up store: %s", self.store)
        self.store.create_group("/", "fragment_profiling_benchmarks", "Profiling benchmark sets.")
        self.store.flush()
    
    @property
    def profiling_benchmarks(self):
        """Return list of fragment profiling benchmark sets."""
        return set(i._v_name for i in self.store.iter_nodes("/fragment_profiling_benchmarks"))
    
    def add_profiling_benchmark(self, name, target_residue_table, profiling_results):
        """Add profiler benchmark data to database."""
        
        self.store.create_group("/fragment_profiling_benchmarks", name, name)
        groupname = path.join("/fragment_profiling_benchmarks", name)
        
        if ":" in target_residue_table:
            self.store.create_external_link(groupname, "residues", target_residue_table)
        else:
            self.store.create_soft_link(groupname, "residues", target_residue_table)
        
        self.store.create_group(groupname, "benchmark_sets", "Calculated benchmark distributions for target residues.")
        benchmark_group = path.join(groupname, "benchmark_sets")
        
        for c, (params, quantile_results) in zip(itertools.count(), profiling_results.items()):
            qt = self.store.create_table(benchmark_group, "profile_%i" % c, quantile_results)
            qt.attrs["profile_parameters"] = params
            qt.flush()
        
        self.store.flush()
        
    def get_profiling_benchmark(self, name):
        """Get profiler data from the given benchmark set name."""
        
        profiler_residues = self.store.get_node(path.join("/fragment_profiling_benchmarks", name, "residues"))
        profiler_residues = profiler_residues().read()
        
        profiler_benchmarks = {}
        
        for b in self.store.get_node(path.join("/fragment_profiling_benchmarks", name, "benchmark_sets")):
            profiler_benchmarks[b.attrs["profile_parameters"]] = b.read()
        
        return profiler_residues, profiler_benchmarks
