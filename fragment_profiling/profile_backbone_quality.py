import logging

from collections import namedtuple

import numpy

from interface_fragment_matching.fragment_fitting.lookup import FragmentMatchLookup
from interface_fragment_matching.fragment_fitting.store import FragmentDatabase, FragmentSpecification

class ProfileBackboneQuality(object):
    logger = logging.getLogger("fragment_profiling.profile_backbone_quality.ProfileBackboneQuality")

    @staticmethod
    def from_database(fragment_database_name, fragment_group_name):
        """Initialize profiler from the given database path and fragment group."""
        with FragmentDatabase(fragment_database_name) as fragment_database:
            test_fragments = fragment_database.fragments[fragment_group_name].read()
            test_fragment_length = fragment_database.fragments[fragment_group_name].attrs.fragment_length
            test_fragment_atoms = fragment_database.fragments[fragment_group_name].attrs.fragment_atoms.split(",")
            test_fragment_spec = FragmentSpecification(test_fragment_length, test_fragment_atoms)
        
        return ProfileBackboneQuality(test_fragments, test_fragment_spec)

    def __init__(self, source_fragments, fragment_spec):
        self.source_fragments = source_fragments
        self.fragment_spec = fragment_spec
        self.lookup = FragmentMatchLookup(self.source_fragments)

    def perform_backbone_analysis(self, query_fragments):
        """Perform backbone analysis on given fragments."""
        query_fragment_coordinates = self.fragment_spec.fragments_to_coordinate_array(query_fragments)
        if not query_fragment_coordinates.shape[-2] == self.fragment_spec.fragment_length * len(self.fragment_spec.fragment_atoms):
            raise ValueError("query_fragments of incorrect length")

        lookup_result = self.lookup.closest_matching_fragment(query_fragment_coordinates)

        lookup_result_quantiles = numpy.ones_like(lookup_result, dtype=float)
        lookup_result_quantiles[numpy.isinf(lookup_result["match_distance"])] = numpy.nan 
    
        return ProfileBackboneQualityResult(query_fragments, lookup_result, lookup_result_quantiles)

class ProfileBackboneQualityResult(namedtuple("ProfileBackboneQualityResultTuple", ["query_fragments", "lookup_results", "lookup_result_quantiles"])):
    """Result container for fragment profiling runs."""
    
    @property
    def result_summary(self):
        result_summary = numpy.zeros_like(self.lookup_results, dtype=[
            ("query_id", "u4"), ("query_resn", "u4"),
            ("match_id", "u4"), ("match_resn", "u4"),
            ("match_distance", float),
            ("threshold_distance", float),
            ("match_quantile", float)])
        
        result_summary["query_id"] = self.query_fragments["id"]
        result_summary["query_resn"] = self.query_fragments["resn"]

        result_summary["match_distance"] = self.lookup_results["match_distance"]
        result_summary["match_quantile"] = self.lookup_result_quantiles

        result_summary["match_id"] = self.lookup_results["match"]["id"]
        result_summary["match_resn"] = self.lookup_results["match"]["resn"]
        result_summary["threshold_distance"] = self.lookup_results["match"]["threshold_distance"]

    
        return result_summary

    def residue_maximum_rmsd(self, source_residues, source_fragment_spec):
        from interface_fragment_matching.structure_database.store import ResidueCache
        residue_max_distance = numpy.empty_like(source_residues, dtype=float)
        residue_max_distance[:] = 0

        source_cache = ResidueCache(source_residues)
        
        for i in range(len(self.query_fragments)):
            f = self.query_fragments[i]
            distance = self.lookup_results[i]["match_distance"]
            fstart = source_cache.residue_index(f)

            fseg = residue_max_distance[fstart: fstart + source_fragment_spec.fragment_length]
            fseg[fseg < distance] = distance

        return residue_max_distance

    def plot_profile(self, ax = None):
        from matplotlib import pylab

        if ax is None:
            ax = pylab.gca()
        
        result_summary = self.result_summary
        match_residual = result_summary["match_distance"] - result_summary["threshold_distance"]

        indicies = numpy.arange(len(match_residual))
        nonmatching_points = list(numpy.flatnonzero(numpy.isinf(match_residual)))

        l = True
        for s, e in zip([0] + nonmatching_points, nonmatching_points + [len(match_residual)] ):
            ax.plot(indicies[s:e], match_residual[s:e], color="blue", label=("Match residual." if l else None))
            l = False

        l = True
        for n in nonmatching_points:
            ax.axvspan(n - .5, n + .5, color="red", alpha=.5, label=("Match failure." if l else None))
            l = False

        ax.legend()
    
    def display_profile(self, source_residues, source_fragment_spec, good_res_threshold = .2, bad_res_threshold = .35):
        from interface_fragment_matching.interactive.embed import residues_display
        import matplotlib.colors
        
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("gr", ["green", "gold", "red"])
        sm = matplotlib.cm.ScalarMappable(matplotlib.colors.Normalize(vmin=good_res_threshold, vmax=bad_res_threshold), cmap=cmap)
    
        residue_maximum_distance = self.residue_maximum_rmsd(source_residues, source_fragment_spec)
        res_color = [matplotlib.colors.rgb2hex(c) for c in sm.to_rgba(residue_maximum_distance)]
    
        color_selector = ["%s:residue %i" % e for e in zip(res_color, xrange(1, len(res_color) + 1))]

        return residues_display(source_residues, color=color_selector)
