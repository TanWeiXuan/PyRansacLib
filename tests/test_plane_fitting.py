import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyRansacLib.ransac import (
    LORansacOptions,
    LocallyOptimizedMSAC,
)


class PlaneSolver:
    """Example solver for normalized 3D planes ax + by + cz + d = 0."""

    def __init__(self, points):
        # The solver owns the input data; PyRansacLib only passes point indices.
        self.points = list(points)

    def min_sample_size(self):
        # Three non-collinear points are enough to define a 3D plane.
        return 3

    def non_minimal_sample_size(self):
        # This example can refine from any three or more points.
        return 3

    def num_data(self):
        return len(self.points)

    def MinimalSolver(self, sample):
        # MinimalSolver returns a list because some problems can produce multiple candidates.
        model = self._plane_from_three_points(sample[0], sample[1], sample[2])
        return [] if model is None else [model]

    def NonMinimalSolver(self, sample, initial_model):
        # LO-MSAC calls this with inlier samples to produce a refined model.
        if len(sample) < self.non_minimal_sample_size():
            return None
        return self._least_squares_plane(sample)

    def EvaluateModelOnPoint(self, model, point_idx):
        # RansacLib expects squared residuals; normalized planes make this squared distance.
        a, b, c, d = model
        x, y, z = self.points[point_idx]
        distance = a * x + b * y + c * z + d
        return distance * distance

    def LeastSquares(self, sample, model):
        # Optional final refinement hook.
        if len(sample) < self.min_sample_size():
            return model
        return self._least_squares_plane(sample)

    def _plane_from_three_points(self, first_index, second_index, third_index):
        # The cross product of two edges gives the plane normal.
        p1 = self.points[first_index]
        p2 = self.points[second_index]
        p3 = self.points[third_index]
        u = self._subtract(p2, p1)
        v = self._subtract(p3, p1)
        normal = self._cross(u, v)
        a, b, c = normal
        d = -(a * p1[0] + b * p1[1] + c * p1[2])
        return self._normalize((a, b, c, d))

    def _least_squares_plane(self, sample):
        # Fit z = mx + ny + k with normal equations, then convert to implicit form.
        selected = [self.points[i] for i in sample]
        sum_x = sum(x for x, _, _ in selected)
        sum_y = sum(y for _, y, _ in selected)
        sum_z = sum(z for _, _, z in selected)
        sum_xx = sum(x * x for x, _, _ in selected)
        sum_yy = sum(y * y for _, y, _ in selected)
        sum_xy = sum(x * y for x, y, _ in selected)
        sum_xz = sum(x * z for x, _, z in selected)
        sum_yz = sum(y * z for _, y, z in selected)
        count = float(len(selected))

        matrix = [
            [sum_xx, sum_xy, sum_x],
            [sum_xy, sum_yy, sum_y],
            [sum_x, sum_y, count],
        ]
        rhs = [sum_xz, sum_yz, sum_z]
        solution = self._solve_3x3(matrix, rhs)
        if solution is None:
            return None

        m, n, k = solution
        return self._normalize((-m, -n, 1.0, -k))

    def _solve_3x3(self, matrix, rhs):
        # Tiny Gaussian elimination helper keeps the example dependency-free.
        augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]
        for col in range(3):
            pivot_row = max(range(col, 3), key=lambda row: abs(augmented[row][col]))
            if abs(augmented[pivot_row][col]) < 1e-12:
                return None
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]

            pivot = augmented[col][col]
            for value_col in range(col, 4):
                augmented[col][value_col] /= pivot

            for row in range(3):
                if row == col:
                    continue
                factor = augmented[row][col]
                for value_col in range(col, 4):
                    augmented[row][value_col] -= factor * augmented[col][value_col]

        return [augmented[row][3] for row in range(3)]

    def _subtract(self, lhs, rhs):
        return tuple(lhs[i] - rhs[i] for i in range(3))

    def _cross(self, lhs, rhs):
        return (
            lhs[1] * rhs[2] - lhs[2] * rhs[1],
            lhs[2] * rhs[0] - lhs[0] * rhs[2],
            lhs[0] * rhs[1] - lhs[1] * rhs[0],
        )

    def _normalize(self, model):
        # Keep a canonical unit-normal representation so equality checks are stable.
        a, b, c, d = model
        norm = math.sqrt(a * a + b * b + c * c)
        if norm == 0.0:
            return None
        a /= norm
        b /= norm
        c /= norm
        d /= norm
        if c < 0.0 or (c == 0.0 and b < 0.0) or (c == 0.0 and b == 0.0 and a < 0.0):
            a, b, c, d = -a, -b, -c, -d
        return a, b, c, d


def make_plane_data():
    # The first block follows z = 0.5x - 0.25y + 2 with small noise.
    points = []
    noise = [-0.02, 0.01, 0.0, 0.015, -0.01, 0.02, -0.015]
    for xi, x in enumerate(range(-5, 6)):
        for yi, y in enumerate(range(-4, 5)):
            z = 0.5 * float(x) - 0.25 * float(y) + 2.0
            z += noise[(xi + 2 * yi) % len(noise)]
            points.append((float(x), float(y), z))

    outliers = [
        (-5.0, -4.0, 9.0),
        (-3.5, 3.0, -5.0),
        (-1.0, -2.0, 7.5),
        (2.0, 4.0, -4.5),
        (4.0, -3.0, 8.0),
        (5.5, 2.0, -3.0),
        (0.0, 5.5, 8.5),
        (6.0, -5.0, -2.0),
    ]
    return points + outliers


def plane_slope_intercept(model):
    # Convert ax + by + cz + d = 0 into z = mx + ny + k for easier reading.
    a, b, c, d = model
    return -a / c, -b / c, -d / c


def print_result(label, model, stats):
    m, n, k = plane_slope_intercept(model)
    print(f"\n{label}")
    print(f"  plane: z = {m:.6f}x + {n:.6f}y + {k:.6f}")
    print(
        "  model (a, b, c, d): "
        f"({model[0]:.6f}, {model[1]:.6f}, {model[2]:.6f}, {model[3]:.6f})"
    )
    print(f"  inliers: {stats.best_num_inliers}")
    print(f"  score: {stats.best_model_score:.6f}")
    print(f"  iterations: {stats.num_iterations}")
    print(f"  local optimization runs: {stats.number_lo_iterations}")


class PlaneFittingExampleTest(unittest.TestCase):
    def make_options(self, seed=7, lo_steps=10):
        # The threshold is squared because EvaluateModelOnPoint returns squared errors.
        return LORansacOptions(
            min_num_iterations_=30,
            max_num_iterations_=500,
            success_probability_=0.999,
            squared_inlier_threshold_=0.0025,
            random_seed_=seed,
            num_lo_steps_=lo_steps,
            lo_starting_iterations_=10,
            final_least_squares_=True,
        )

    def test_lo_msac_recovers_dominant_plane_with_outliers(self):
        print("\n--- LO-MSAC plane-fitting example ---")
        solver = PlaneSolver(make_plane_data())
        print(f"Input points: {solver.num_data()} total")
        print("Expected dominant plane: z = 0.5x - 0.25y + 2")

        model, stats = LocallyOptimizedMSAC().estimate_model(
            self.make_options(), solver
        )

        print_result("Recovered LO-MSAC result", model, stats)
        m, n, k = plane_slope_intercept(model)
        self.assertGreaterEqual(stats.best_num_inliers, 95)
        self.assertAlmostEqual(m, 0.5, delta=0.01)
        self.assertAlmostEqual(n, -0.25, delta=0.01)
        self.assertAlmostEqual(k, 2.0, delta=0.03)
        self.assertGreaterEqual(stats.number_lo_iterations, 1)

    def test_msac_mode_is_deterministic_with_fixed_seed(self):
        print("\n--- MSAC deterministic plane-fitting example ---")
        solver = PlaneSolver(make_plane_data())
        ransac = LocallyOptimizedMSAC()
        options = self.make_options(seed=3, lo_steps=0)
        print(f"Using random_seed_={options.random_seed_} and num_lo_steps_=0")

        model_a, stats_a = ransac.estimate_model(options, solver)
        model_b, stats_b = ransac.estimate_model(options, solver)

        print_result("First MSAC run", model_a, stats_a)
        print_result("Second MSAC run", model_b, stats_b)
        print("  deterministic models match:", model_a == model_b)
        print("  deterministic inlier lists match:", stats_a.inlier_indices == stats_b.inlier_indices)
        self.assertEqual(model_a, model_b)
        self.assertEqual(stats_a.inlier_indices, stats_b.inlier_indices)
        self.assertEqual(stats_a.best_num_inliers, stats_b.best_num_inliers)
        self.assertGreaterEqual(stats_a.best_num_inliers, 95)


if __name__ == "__main__":
    unittest.main()
