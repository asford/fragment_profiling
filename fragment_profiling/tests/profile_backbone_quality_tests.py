import unittest
import logging

import os
from os import path

import numpy

from interface_fragment_matching.fragment_fitting.lookup import FragmentMatchLookup
from interface_fragment_matching.fragment_fitting.store import FragmentDatabase, FragmentSpecification
from interface_fragment_matching.structure_database import StructureDatabase

from fragment_profiling.profile_backbone_quality import ProfileBackboneQuality

class TestProfileBackboneQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rosetta
        rosetta.init("-out:levels", "all:warning")

        from interface_fragment_matching.parallel import openmp_utils
        openmp_utils.omp_set_num_threads(1)

    def setUp(self):
        import rosetta
        fragment_db = FragmentDatabase("/work/fordas/workspace/fragment_fitting/threshold_test_fragments/test_sets.h5")
        self.test_fragments = fragment_db.fragments["source_fragments_4_mer"].read()

        test_fragment_length = fragment_db.fragments["source_fragments_4_mer"].attrs.fragment_length
        test_fragment_atoms = fragment_db.fragments["source_fragments_4_mer"].attrs.fragment_atoms.split(",")
        self.test_fragment_spec = FragmentSpecification(test_fragment_length, tuple(test_fragment_atoms))

        pass_test_structure = rosetta.pose_from_pdb(path.join(path.dirname(__file__), "foldit_17_0001.pdb" ))
        self.pass_test_residues = StructureDatabase.extract_residue_entries_from_pose(pass_test_structure)
        _, self.pass_test_fragments = self.test_fragment_spec.fragments_from_source_residues(self.pass_test_residues)

        fail_test_structure = rosetta.pose_from_pdb(path.join(path.dirname(__file__), "foldit_18_0001.pdb" ))
        self.fail_test_residues = StructureDatabase.extract_residue_entries_from_pose(fail_test_structure)
        _, self.fail_test_fragments = self.test_fragment_spec.fragments_from_source_residues(self.fail_test_residues)

    def test_profiler(self):
        profiler = ProfileBackboneQuality(self.test_fragments, self.test_fragment_spec)
        pass_results = profiler.perform_backbone_analysis(self.pass_test_fragments).result_summary
        self.assertTrue(not any(numpy.isnan(pass_results["match_distance"])))

        profiler = ProfileBackboneQuality(self.test_fragments, self.test_fragment_spec)
        fail_results = profiler.perform_backbone_analysis(self.fail_test_fragments).result_summary
        self.assertTrue(any(numpy.isinf(fail_results["match_distance"])))
