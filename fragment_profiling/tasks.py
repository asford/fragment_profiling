import logging

import numpy

from interface_fragment_matching.tasks.utility import TaskBase

from interface_fragment_matching.structure_database.store import StructureDatabase

from .profile_fragment_quality import ProfileFragmentQuality, FragmentProfilerParameters
from .store import FragmentProfilingDatabase

# Set default omp parameters for interactive use
from interface_fragment_matching.parallel import openmp_utils
if not openmp_utils.omp_get_max_threads():
    logging.warning("omp_set_num_threads(%s)", 4)
    openmp_utils.omp_set_num_threads(4)

class ProfileFragmentQualityTask(TaskBase):
    """Perform fragment quality profiling on structure residues."""

    @property
    def state_keys(self):
        return ["fragment_specification", "profile_source_database", "logscore_substitution_profile", "select_fragments_per_query_position"]
    
    def __init__(self, fragment_specification, profile_source_database, logscore_substitution_profile, select_fragments_per_query_position):
        """Create profiling task.

        fragent_specifiction - Specifiction used to defined profiled fragments.
        profile_source_database - Source database to profiling queries.
        logscore_substitution_profile - Substitution profile used in profiler.
        select_fragments_per_query_position - Result fragments per position.
        """

        self.fragment_specification = fragment_specification
        self.profile_source_database = profile_source_database
        
        self.logscore_substitution_profile = logscore_substitution_profile
        self.select_fragments_per_query_position = select_fragments_per_query_position
        TaskBase.__init__(self)
    
    def setup(self):
        """Load source structure database and prepare profiler."""
        source_residues = StructureDatabase(self.profile_source_database).residues.read()
        
        self.profiler = ProfileFragmentQuality(
                                source_residues,
                                self.logscore_substitution_profile,
                                self.select_fragments_per_query_position)
        
        TaskBase.setup(self)
    
    def execute(self, target_residues):
        """Segment target residues into fragments and perform per-fragment profiling."""
        _, target_fragments = self.fragment_specification.fragments_from_source_residues(
                                        target_residues,
                                        additional_per_residue_fields=["bb", "sc", "ss"])

        return self.profiler.perform_fragment_analysis(target_fragments)

class BenchmarkedProfileFragmentQualityTask(TaskBase):
    """Perform fragment quality profiling on structure residues via a benchmarked profiling configuration."""

    @property
    def state_keys(self):
        return ["profiling_database", "profile_benchmark_name", "profiler_parameters"]
    
    def __init__(self, profiling_database, profile_benchmark_name, profiler_parameters):
        """Create profiling task.

        profiling_database - Source profiling database.
        profile_benchmark_name - Source profiling database.
        profiler_parameters - Profiling parameters for run.
        """

        self.profiling_database = profiling_database
        self.profile_benchmark_name = profile_benchmark_name
        self.profiler_parameters = profiler_parameters

        TaskBase.__init__(self)
    
    def setup(self):
        """Load source structure database and prepare profiler."""
        with FragmentProfilingDatabase(self.profiling_database) as prof_db:
            source_residues, source_benchmarks = prof_db.get_profiling_benchmark(self.profile_benchmark_name)

        if not self.profiler_parameters in source_benchmarks:
            raise ValueError("Provided profiler_parameters %s not present in database. Available parameters:\n%s", self.profiler_parameters, source_benchmarks.keys())

        self.profiler = ProfileFragmentQuality(
                                source_residues,
                                self.profiler_parameters.logscore_substitution_profile,
                                self.profiler_parameters.select_fragments_per_query_position,
                                source_benchmarks)
        
        TaskBase.setup(self)
    
    def execute(self, target_residues):
        """Segment target residues into fragments and perform per-fragment profiling."""
        _, target_fragments = self.profiler_parameters.fragment_specification.fragments_from_source_residues(
                                        target_residues,
                                        additional_per_residue_fields=["bb", "sc", "ss"])

        return self.profiler.perform_fragment_analysis(target_fragments)

class BenchmarkProfileFragmentQualityTask(ProfileFragmentQualityTask):
    @property
    def state_keys(self):
        return super(BenchmarkProfileFragmentQualityTask, self).state_keys + ["target_structure_database", "return_fragments_per_query_position"]

    def __init__(self, target_structure_database, fragment_specification, profile_source_database, logscore_substitution_profile, select_fragments_per_query_position, return_fragments_per_query_position):
        ProfileFragmentQualityTask.__init__(self, fragment_specification, profile_source_database, logscore_substitution_profile, select_fragments_per_query_position)

        self.target_structure_database = target_structure_database
        self.return_fragments_per_query_position = return_fragments_per_query_position

    def setup(self):
        super(BenchmarkProfileFragmentQualityTask, self).setup()
        self.target_residue_cache = StructureDatabase(self.target_structure_database).residue_cache

    def execute(self, target_structure_ids):
        self.logger.info("execute(<%s ids>)", len(target_structure_ids))

        ids = numpy.atleast_1d(numpy.array(target_structure_ids)).astype("u4")
        spans = self.target_residue_cache.id_span(ids)
        source_residues = numpy.concatenate([self.target_residue_cache.residues[spans[i]["start"]:spans[i]["end"]] for i in xrange(len(spans))])

        self.logger.info("source_residues: <%s residues> ", len(source_residues))

        result = ProfileFragmentQualityTask.execute(self, source_residues)

        return result.prune_fragments_by_start_residue().generate_result_summary(self.return_fragments_per_query_position)
