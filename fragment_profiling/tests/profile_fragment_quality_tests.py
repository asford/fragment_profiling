import unittest

import numpy
import numpy.testing

from interface_fragment_matching.fragment_fitting.rmsd_calc import atom_array_broadcast_rmsd
from interface_fragment_matching.fragment_fitting.store import FragmentSpecification
from interface_fragment_matching.structure_database import StructureDatabase

from fragment_profiling.profile_fragment_quality import ProfileFragmentQuality

class TestProfileFragmentQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from interface_fragment_matching.parallel import openmp_utils
        openmp_utils.omp_set_num_threads(2)

    def setUp(self):
        structure_db = StructureDatabase("/work/fordas/test_sets/vall_store.h5")
        test_structure_residues = structure_db.residues.readWhere("id == 1")

        self.test_fragment_spec = FragmentSpecification(9, "CA")
        self.source_test_segment = test_structure_residues[:self.test_fragment_spec.fragment_length + 2]

        self.test_structures = numpy.empty((750, len(self.source_test_segment)), self.source_test_segment.dtype)

        self.test_structures[:] = numpy.expand_dims(self.source_test_segment, 0)
        self.test_structures["id"] = numpy.arange(750).reshape((750, 1))
        self.test_structures["sc"]["aa"] = "A"
        self.test_structures["sc"]["aa"][...,-2:] = "G"

        _, self.test_query_fragments = self.test_fragment_spec.fragments_from_source_residues(
                self.source_test_segment, additional_per_residue_fields=["bb", "sc", "ss"])

        self.test_query_rmsds = atom_array_broadcast_rmsd(
                self.test_query_fragments["coordinates"],
                self.test_query_fragments["coordinates"])

    def test_profiler(self):
        query_fragment = self.test_query_fragments[1].copy()
        query_fragment["sc"]["aa"] = self.test_structures[0]["sc"]["aa"][:9]

        # Perform query w/ fragment 0 sequence and fragment 1 conformation.
        # Should select all fragment 0 instances
        profiler_one = ProfileFragmentQuality(self.test_structures.ravel(), "blosum100", 750)
        result_one = profiler_one.perform_fragment_analysis(numpy.expand_dims(query_fragment, 0))

        numpy.testing.assert_array_almost_equal(
                result_one.selected_fragment_rmsds,
                numpy.repeat(self.test_query_rmsds[0,1], 750).reshape(1, 750))
        numpy.testing.assert_array_almost_equal(
                result_one.selected_fragments["resn"],
                numpy.repeat(self.test_query_fragments[0]["resn"], 750).reshape(1, 750))
        numpy.testing.assert_array_almost_equal(
                numpy.sort(result_one.selected_fragments["id"].ravel()),
                numpy.arange(750))

        # Should select all fragment 0 and single fragment 1
        profiler_two = ProfileFragmentQuality(self.test_structures.ravel(), "blosum100", 750 * 2)
        result_two = profiler_two.perform_fragment_analysis(numpy.expand_dims(query_fragment, 0))

        numpy.testing.assert_array_almost_equal(
                numpy.sort(result_two.selected_fragment_rmsds),
                numpy.repeat(
                    [self.test_query_rmsds[0, 0], self.test_query_rmsds[0,1]],
                    750).reshape(1, 750 * 2))
        numpy.testing.assert_array_almost_equal(
                numpy.sort(result_two.selected_fragments["resn"]),
                numpy.repeat(
                    [self.test_query_fragments[0]["resn"], self.test_query_fragments[1]["resn"]],
                    750).reshape(1, 750 * 2))
        numpy.testing.assert_array_almost_equal(
                numpy.sort(result_two.selected_fragments["id"].ravel()),
                numpy.repeat(numpy.arange(750), 2))
