import os
import re
from typing import Callable
import unittest

import numpy as np

from opencsp.common.lib.cv.CacheableImage import CacheableImage
from opencsp.common.lib.cv.spot_analysis.SpotAnalysisOperable import SpotAnalysisOperable
from opencsp.common.lib.cv.spot_analysis.image_processor.AbstractAggregateImageProcessor import (
    AbstractAggregateImageProcessor,
)
import opencsp.common.lib.cv.SpotAnalysis as sa
import opencsp.common.lib.tool.file_tools as ft


class TestAbstractAggregateImageProcessor(unittest.TestCase):
    def setUp(self) -> None:
        path, _, _ = ft.path_components(__file__)
        self.data_dir = os.path.join(path, "data", "input", "AbstractAggregateImageProcessor")
        self.out_dir = os.path.join(path, "data", "output", "AbstractAggregateImageProcessor")
        ft.create_directories_if_necessary(self.data_dir)
        ft.create_directories_if_necessary(self.out_dir)

        # # generate the test data
        # a = np.array([0])
        # np.save(os.path.join(self.data_dir, "a1.np"), a, allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "b1.np"), a, allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "b2.np"), a, allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "a2.np"), a, allow_pickle=False)
        # np.save(os.path.join(self.data_dir, "a3.np"), a, allow_pickle=False)
        filenames = ["a1.np", "b1.np", "b2.np", "a2.np", "a3.np"]
        self.image_files = [os.path.join(self.data_dir, filename) for filename in filenames]

    def test_group_by_brightness(self):
        op0 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="op0")
        op10 = SpotAnalysisOperable(CacheableImage(np.array([10])), primary_image_source_path="op10")
        op20 = SpotAnalysisOperable(CacheableImage(np.array([20])), primary_image_source_path="op20")
        op30 = SpotAnalysisOperable(CacheableImage(np.array([30])), primary_image_source_path="op30")
        op40 = SpotAnalysisOperable(CacheableImage(np.array([40])), primary_image_source_path="op40")

        g = AbstractAggregateImageProcessor.group_by_brightness({10: 3, 20: 2, 30: 1})
        self.assertEqual(g(op0), 3)
        self.assertEqual(g(op10), 3)
        self.assertEqual(g(op20), 2)
        self.assertEqual(g(op30), 1)
        self.assertEqual(g(op40), 1)

    def test_group_by_name(self):
        opa1 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testastring")
        opb1 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testbstring")
        opc1 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testcstring")
        opa2 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testastring")
        opa3 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="tesastring")

        # normal operation
        g = AbstractAggregateImageProcessor.group_by_name(re.compile("test?(.)string"))
        self.assertEqual(g(opa1), 0)
        self.assertEqual(g(opb1), 1)
        self.assertEqual(g(opc1), 2)
        self.assertEqual(g(opa2), 0)
        self.assertEqual(g(opa3), 0)

        # doesn't match the pattern, return default (should also print a warning)
        opbad = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="b")
        self.assertEqual(g(opbad), 0)

    def test_group_trigger_on_change(self):
        opa1 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testastring")
        opb1 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testbstring")
        opc1 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testcstring")
        opa2 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="testastring")
        opa3 = SpotAnalysisOperable(CacheableImage(np.array([0])), primary_image_source_path="tesastring")

        t = AbstractAggregateImageProcessor.group_trigger_on_change()
        image_groups = []
        self.assertEqual(t(image_groups), None)
        image_groups.append((opa1, 0))
        self.assertEqual(t(image_groups), None)
        image_groups.append((opb1, 1))
        self.assertEqual(t(image_groups), 0)
        image_groups.append((opc1, 2))
        self.assertEqual(t(image_groups), 1)
        image_groups.append((opa2, 0))
        self.assertEqual(t(image_groups), 2)
        image_groups.append((opa3, 0))
        self.assertEqual(t(image_groups), None)

    def test_execute_aggregate(self):
        assigner = AbstractAggregateImageProcessor.group_by_name(re.compile(r"(.)[0-9].np"))
        always_trigger = lambda image_groups: image_groups[0][1]
        aggregator = ConcreteAggregateImageProcessor(assigner, always_trigger)
        spot_analysis = sa.SpotAnalysis("test_execute_aggregate", [aggregator])
        spot_analysis.set_primary_images(self.image_files)

        expected_group_order = [0, 1, 1, 0, 0]
        expected_group_sizes = [1, 1, 1, 1, 1]
        expected_is_last = [False, False, False, False, True]
        for i, operable in enumerate(spot_analysis):
            self.assertEqual(expected_group_order[i], aggregator.prev_executed_group)
            self.assertEqual(expected_group_sizes[i], aggregator.prev_group_size)
            self.assertEqual(expected_is_last[i], aggregator.prev_is_last)

    def test_execute_aggregate_(self):
        assigner = AbstractAggregateImageProcessor.group_by_name(re.compile(r"(.)[0-9].np"))
        triggerer = AbstractAggregateImageProcessor.group_trigger_on_change()
        aggregator = ConcreteAggregateImageProcessor(assigner, triggerer)
        spot_analysis = sa.SpotAnalysis("test_execute_aggregate", [aggregator])
        spot_analysis.set_primary_images(self.image_files)

        expected_group_order = [0, 1, 1, 0, 0]
        expected_group_sizes = [1, 2, 2, 2, 2]
        expected_is_last = [False, False, False, True, True]
        for i, operable in enumerate(spot_analysis):
            self.assertEqual(expected_group_order[i], aggregator.prev_executed_group)
            self.assertEqual(expected_group_sizes[i], aggregator.prev_group_size)
            self.assertEqual(expected_is_last[i], aggregator.prev_is_last)


class ConcreteAggregateImageProcessor(AbstractAggregateImageProcessor):
    def __init__(
        self,
        images_group_assigner: Callable[[SpotAnalysisOperable], int],
        group_execution_trigger: Callable[[list[tuple[SpotAnalysisOperable, int]]], int | None] = None,
        *vargs,
        **kwargs
    ):
        super().__init__(images_group_assigner, group_execution_trigger, *vargs, **kwargs)
        self.prev_executed_group = None
        self.prev_group_size = 0
        self.prev_is_last = False

    def _execute_aggregate(
        self, group: int, operables: list[SpotAnalysisOperable], is_last: bool
    ) -> list[SpotAnalysisOperable]:
        self.prev_executed_group = group
        self.prev_group_size = len(operables)
        self.prev_is_last = is_last
        return operables


if __name__ == "__main__":
    unittest.main()
