import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyRansacLib import utils
from PyRansacLib.ransac import LORansacOptions, LocallyOptimizedMSAC
from PyRansacLib.sampling import UniformSampling


class SequenceSampler:
    def __init__(self, random_seed, solver):
        self.samples = [[0], [1]]
        self.index = 0

    def Sample(self):
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        return list(sample)


class MinimalScoreRegressionSolver:
    def __init__(self):
        self.lo_initial_models = []

    def min_sample_size(self):
        return 1

    def non_minimal_sample_size(self):
        return 1

    def num_data(self):
        return 2

    def MinimalSolver(self, sample):
        return ["A_raw"] if sample[0] == 0 else ["B_raw"]

    def NonMinimalSolver(self, sample, initial_model):
        self.lo_initial_models.append(initial_model)
        if initial_model == "A_raw":
            return "A_lo"
        if initial_model == "B_raw":
            return "B_lo"
        return None

    def EvaluateModelOnPoint(self, model, point_idx):
        scores = {
            "A_raw": 100.0,
            "A_lo": 10.0,
            "B_raw": 50.0,
            "B_lo": 0.0,
        }
        return scores[model] / float(self.num_data())


class MinimalScoreRegressionTest(unittest.TestCase):
    def test_lo_score_does_not_replace_best_raw_minimal_score(self):
        solver = MinimalScoreRegressionSolver()
        options = LORansacOptions(
            min_num_iterations_=2,
            max_num_iterations_=2,
            success_probability_=0.999,
            squared_inlier_threshold_=1000.0,
            random_seed_=0,
            num_lo_steps_=1,
            lo_starting_iterations_=0,
            final_least_squares_=False,
        )

        model, stats = LocallyOptimizedMSAC(
            sampler_cls=SequenceSampler
        ).estimate_model(options, solver)

        self.assertEqual(model, "B_lo")
        self.assertEqual(stats.best_model_score, 0.0)
        self.assertEqual(solver.lo_initial_models, ["A_raw", "B_raw"])


class SamplingSolver:
    def __init__(self, num_data, sample_size):
        self._num_data = num_data
        self._sample_size = sample_size

    def min_sample_size(self):
        return self._sample_size

    def num_data(self):
        return self._num_data


class UtilityAndSamplerTest(unittest.TestCase):
    def test_uniform_sampler_draws_unique_samples(self):
        solver = SamplingSolver(num_data=49, sample_size=2)
        sampler = UniformSampling(11, solver)

        for _ in range(25):
            sample = sampler.Sample()
            self.assertEqual(len(sample), solver.min_sample_size())
            self.assertEqual(len(sample), len(set(sample)))
            self.assertTrue(all(0 <= i < solver.num_data() for i in sample))

    def test_uniform_sampler_can_draw_the_full_dataset(self):
        solver = SamplingSolver(num_data=2, sample_size=2)
        sample = UniformSampling(5, solver).Sample()

        self.assertEqual(sorted(sample), [0, 1])

    def test_num_required_iterations_clamps_edge_cases(self):
        self.assertEqual(utils.NumRequiredIterations(0.0, 0.01, 2, 5, 100), 100)
        self.assertEqual(utils.NumRequiredIterations(1.0, 0.01, 2, 5, 100), 5)

        iterations = utils.NumRequiredIterations(0.5, 0.01, 2, 5, 100)
        self.assertGreaterEqual(iterations, 5)
        self.assertLessEqual(iterations, 100)


class CountingScoreSolver:
    def __init__(self, squared_errors):
        self.squared_errors = list(squared_errors)
        self.num_evaluations = 0

    def num_data(self):
        return len(self.squared_errors)

    def EvaluateModelOnPoint(self, model, point_idx):
        self.num_evaluations += 1
        return self.squared_errors[point_idx]

    def reset_count(self):
        self.num_evaluations = 0


class BoundedScoringTest(unittest.TestCase):
    def test_bounded_scoring_can_stop_before_exact_scoring(self):
        solver = CountingScoreSolver([1.0] * 20)
        ransac = LocallyOptimizedMSAC()

        exact_score = ransac.ScoreModel(
            solver, model="candidate", squared_inlier_threshold=10.0
        )
        exact_evaluations = solver.num_evaluations

        solver.reset_count()
        bounded_score = ransac._score_model_bounded(
            solver,
            model="candidate",
            squared_inlier_threshold=10.0,
            max_score=5.0,
        )

        self.assertEqual(exact_score, 20.0)
        self.assertEqual(exact_evaluations, solver.num_data())
        self.assertGreaterEqual(bounded_score, 5.0)
        self.assertLess(solver.num_evaluations, exact_evaluations)


if __name__ == "__main__":
    unittest.main()
