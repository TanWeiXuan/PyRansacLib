import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyRansacLib.ransac import (
    LORansacOptions,
    LocallyOptimizedMSAC,
)


class LineSolver:
    """Example solver for normalized 2D lines ax + by + c = 0."""

    def __init__(self, points):
        self.points = list(points)

    def min_sample_size(self):
        return 2

    def non_minimal_sample_size(self):
        return 2

    def num_data(self):
        return len(self.points)

    def MinimalSolver(self, sample):
        model = self._line_from_two_points(sample[0], sample[1])
        return [] if model is None else [model]

    def NonMinimalSolver(self, sample, initial_model):
        if len(sample) < self.non_minimal_sample_size():
            return None
        return self._least_squares_line(sample)

    def EvaluateModelOnPoint(self, model, i):
        a, b, c = model
        x, y = self.points[i]
        distance = a * x + b * y + c
        return distance * distance

    def LeastSquares(self, sample, model):
        if len(sample) < self.min_sample_size():
            return model
        return self._least_squares_line(sample)

    def _line_from_two_points(self, first_index, second_index):
        x1, y1 = self.points[first_index]
        x2, y2 = self.points[second_index]
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        return self._normalize((a, b, c))

    def _least_squares_line(self, sample):
        selected = [self.points[i] for i in sample]
        mean_x = sum(x for x, _ in selected) / float(len(selected))
        mean_y = sum(y for _, y in selected) / float(len(selected))

        sxx = sum((x - mean_x) ** 2 for x, _ in selected)
        syy = sum((y - mean_y) ** 2 for _, y in selected)
        sxy = sum((x - mean_x) * (y - mean_y) for x, y in selected)

        direction_angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
        normal_angle = direction_angle + math.pi / 2.0
        a = math.cos(normal_angle)
        b = math.sin(normal_angle)
        c = -(a * mean_x + b * mean_y)
        return self._normalize((a, b, c))

    def _normalize(self, model):
        a, b, c = model
        norm = math.hypot(a, b)
        if norm == 0.0:
            return None
        a /= norm
        b /= norm
        c /= norm
        if b < 0.0 or (b == 0.0 and a < 0.0):
            a, b, c = -a, -b, -c
        return a, b, c


def make_line_data():
    noise = [-0.025, 0.015, 0.0, 0.02, -0.01, 0.01, -0.015]
    inliers = [
        (float(x), 2.0 * float(x) + 1.0 + noise[x % len(noise)])
        for x in range(-20, 21)
    ]
    outliers = [
        (-18.0, 19.0),
        (-13.5, -22.0),
        (-7.0, 30.0),
        (-2.0, -17.0),
        (4.0, 26.0),
        (8.0, -25.0),
        (15.0, 5.0),
        (21.0, -8.0),
    ]
    return inliers + outliers


def line_slope_intercept(model):
    a, b, c = model
    return -a / b, -c / b


class LineFittingExampleTest(unittest.TestCase):
    def make_options(self, seed=7, lo_steps=10):
        return LORansacOptions(
            min_num_iterations_=20,
            max_num_iterations_=300,
            success_probability_=0.999,
            squared_inlier_threshold_=0.01,
            random_seed_=seed,
            num_lo_steps_=lo_steps,
            lo_starting_iterations_=10,
            final_least_squares_=True,
        )

    def test_lo_msac_recovers_dominant_line_with_outliers(self):
        solver = LineSolver(make_line_data())
        model, stats = LocallyOptimizedMSAC().estimate_model(
            self.make_options(), solver
        )

        slope, intercept = line_slope_intercept(model)
        self.assertGreaterEqual(stats.best_num_inliers, 40)
        self.assertAlmostEqual(slope, 2.0, delta=0.01)
        self.assertAlmostEqual(intercept, 1.0, delta=0.03)
        self.assertGreaterEqual(stats.number_lo_iterations, 1)

    def test_msac_mode_is_deterministic_with_fixed_seed(self):
        solver = LineSolver(make_line_data())
        ransac = LocallyOptimizedMSAC()
        options = self.make_options(seed=3, lo_steps=0)

        model_a, stats_a = ransac.estimate_model(options, solver)
        model_b, stats_b = ransac.estimate_model(options, solver)

        self.assertEqual(model_a, model_b)
        self.assertEqual(stats_a.inlier_indices, stats_b.inlier_indices)
        self.assertEqual(stats_a.best_num_inliers, stats_b.best_num_inliers)
        self.assertGreaterEqual(stats_a.best_num_inliers, 40)

    def test_estimate_model_alias_returns_model_and_statistics(self):
        solver = LineSolver(make_line_data())
        model, stats = LocallyOptimizedMSAC().EstimateModel(
            self.make_options(), solver
        )

        self.assertIsNotNone(model)
        self.assertGreaterEqual(stats.best_num_inliers, 40)


if __name__ == "__main__":
    unittest.main()
