"""
Unofficial Python port of RansacLib/ransac.h from RansacLib.

Original project:
    RansacLib
    Copyright (c) 2019, Torsten Sattler
    License: BSD 3-Clause

Unofficial Python port:
    Copyright (c) 2026 Tan Wei Xuan

See LICENSE and NOTICE for details.
"""

import copy
import math
import random
from dataclasses import dataclass, field

try:
    from . import utils
    from .sampling import UniformSampling
except ImportError:  # pragma: no cover - supports direct module execution.
    import utils
    from sampling import UniformSampling


@dataclass
class RansacOptions:
    min_num_iterations_: int = 100
    max_num_iterations_: int = 10000
    success_probability_: float = 0.9999
    squared_inlier_threshold_: float = 1.0
    random_seed_: int = 0


@dataclass
class LORansacOptions(RansacOptions):
    num_lo_steps_: int = 10
    threshold_multiplier_: float = math.sqrt(2.0)
    num_lsq_iterations_: int = 4
    min_sample_multiplicator_: int = 7
    non_min_sample_multiplier_: int = 3
    lo_starting_iterations_: int = 50
    final_least_squares_: bool = False


@dataclass
class RansacStatistics:
    num_iterations: int = 0
    best_num_inliers: int = 0
    best_model_score: float = math.inf
    inlier_ratio: float = 0.0
    inlier_indices: list = field(default_factory=list)
    number_lo_iterations: int = 0


class RansacBase:
    def ResetStatistics(self, statistics):
        statistics.best_num_inliers = 0
        statistics.best_model_score = math.inf
        statistics.num_iterations = 0
        statistics.inlier_ratio = 0.0
        statistics.inlier_indices.clear()
        statistics.number_lo_iterations = 0

    reset_statistics = ResetStatistics


class LocallyOptimizedMSAC(RansacBase):
    """LO-MSAC estimator ported from RansacLib's ``ransac.h``."""

    def __init__(self, sampler_cls=UniformSampling):
        self.sampler_cls = sampler_cls

    def estimate_model(self, options, solver):
        return self.EstimateModel(options, solver)

    def EstimateModel(self, options, solver):
        statistics = RansacStatistics()
        self.ResetStatistics(statistics)

        min_sample_size = int(solver.min_sample_size())
        num_data = int(solver.num_data())
        if min_sample_size > num_data or min_sample_size <= 0:
            return None, statistics

        sampler = self.sampler_cls(options.random_seed_, solver)
        rng = random.Random(options.random_seed_)
        max_num_iterations = max(
            int(options.max_num_iterations_), int(options.min_num_iterations_)
        )
        squared_inlier_threshold = float(options.squared_inlier_threshold_)

        best_model = None
        best_minimal_model = None
        best_min_model_score = math.inf

        while statistics.num_iterations < max_num_iterations:
            if (
                statistics.num_iterations == options.lo_starting_iterations_
                and best_min_model_score < math.inf
            ):
                statistics.number_lo_iterations += 1
                best_model, statistics.best_model_score = self.LocalOptimization(
                    options,
                    solver,
                    rng,
                    best_model,
                    statistics.best_model_score,
                )
                self._update_statistics(
                    statistics,
                    solver,
                    best_model,
                    squared_inlier_threshold,
                    num_data,
                )
                max_num_iterations = self._update_max_iterations(
                    statistics,
                    options,
                    min_sample_size,
                )

            minimal_sample = sampler.Sample()
            estimated_models = self._call_minimal_solver(solver, minimal_sample)
            if not estimated_models:
                statistics.num_iterations += 1
                continue

            best_local_score, best_local_model_id = self.GetBestEstimatedModelId(
                solver, estimated_models, squared_inlier_threshold
            )

            if (
                best_local_score < best_min_model_score
                or statistics.num_iterations == options.lo_starting_iterations_
            ):
                is_best_min_model = best_local_score < best_min_model_score

                if is_best_min_model:
                    best_min_model_score = best_local_score
                    best_minimal_model = self._copy_model(
                        estimated_models[best_local_model_id]
                    )
                    best_model, statistics.best_model_score = self.UpdateBestModel(
                        best_local_score,
                        best_minimal_model,
                        statistics.best_model_score,
                        best_model,
                    )

                run_lo = (
                    statistics.num_iterations >= options.lo_starting_iterations_
                    and best_min_model_score < math.inf
                )

                if is_best_min_model or run_lo:
                    if run_lo:
                        statistics.number_lo_iterations += 1
                        lo_model, lo_score = self.LocalOptimization(
                            options,
                            solver,
                            rng,
                            best_minimal_model,
                            best_min_model_score,
                        )
                        best_model, statistics.best_model_score = (
                            self.UpdateBestModel(
                                lo_score,
                                lo_model,
                                statistics.best_model_score,
                                best_model,
                            )
                        )

                    self._update_statistics(
                        statistics,
                        solver,
                        best_model,
                        squared_inlier_threshold,
                        num_data,
                    )
                    max_num_iterations = self._update_max_iterations(
                        statistics,
                        options,
                        min_sample_size,
                    )

            statistics.num_iterations += 1

        if (
            statistics.num_iterations <= options.lo_starting_iterations_
            and statistics.best_model_score < math.inf
            and best_model is not None
        ):
            statistics.number_lo_iterations += 1
            best_model, statistics.best_model_score = self.LocalOptimization(
                options,
                solver,
                rng,
                best_model,
                statistics.best_model_score,
            )
            self._update_statistics(
                statistics,
                solver,
                best_model,
                squared_inlier_threshold,
                num_data,
            )

        if (
            options.final_least_squares_
            and best_model is not None
            and statistics.inlier_indices
        ):
            refined_model = self._call_least_squares(
                solver, statistics.inlier_indices, best_model
            )
            score = self.ScoreModel(solver, refined_model, squared_inlier_threshold)
            if score < statistics.best_model_score:
                statistics.best_model_score = score
                best_model = refined_model
                self._update_statistics(
                    statistics,
                    solver,
                    best_model,
                    squared_inlier_threshold,
                    num_data,
                )

        return best_model, statistics

    def GetBestEstimatedModelId(self, solver, models, squared_inlier_threshold):
        best_score = math.inf
        best_model_id = 0
        for model_index, model in enumerate(models):
            score = self.ScoreModel(solver, model, squared_inlier_threshold)
            if score < best_score:
                best_score = score
                best_model_id = model_index
        return best_score, best_model_id

    get_best_estimated_model_id = GetBestEstimatedModelId

    def ScoreModel(self, solver, model, squared_inlier_threshold):
        score = 0.0
        for point_index in range(int(solver.num_data())):
            squared_error = solver.EvaluateModelOnPoint(model, point_index)
            score += self.ComputeScore(squared_error, squared_inlier_threshold)
        return score

    score_model = ScoreModel

    def ComputeScore(self, squared_error, squared_error_threshold):
        return min(float(squared_error), float(squared_error_threshold))

    compute_score = ComputeScore

    def GetInliers(self, solver, model, squared_inlier_threshold):
        inliers = []
        for point_index in range(int(solver.num_data())):
            squared_error = solver.EvaluateModelOnPoint(model, point_index)
            if squared_error < squared_inlier_threshold:
                inliers.append(point_index)
        return inliers

    get_inliers = GetInliers

    def LocalOptimization(self, options, solver, rng, initial_model, initial_score):
        num_data = int(solver.num_data())
        min_non_min_sample_size = int(solver.non_minimal_sample_size())
        if min_non_min_sample_size > num_data:
            return self._copy_model(initial_model), initial_score

        min_sample_size = int(solver.min_sample_size())
        squared_inlier_threshold = float(options.squared_inlier_threshold_)
        threshold_multiplier = float(options.threshold_multiplier_)

        best_model = self._copy_model(initial_model)
        best_score = initial_score

        least_squares_model = self.LeastSquaresFit(
            options,
            squared_inlier_threshold * threshold_multiplier,
            solver,
            rng,
            best_model,
        )
        least_squares_score = self.ScoreModel(
            solver, least_squares_model, squared_inlier_threshold
        )
        best_model, best_score = self.UpdateBestModel(
            least_squares_score, least_squares_model, best_score, best_model
        )

        inliers_base = self.GetInliers(
            solver,
            least_squares_model,
            squared_inlier_threshold * threshold_multiplier,
        )
        if len(inliers_base) < min_non_min_sample_size:
            return best_model, best_score

        non_min_sample_size = max(
            min_non_min_sample_size,
            min(
                min_sample_size * int(options.non_min_sample_multiplier_),
                len(inliers_base) // 2,
            ),
        )
        non_min_sample_size = min(non_min_sample_size, len(inliers_base))

        for _ in range(int(options.num_lo_steps_)):
            sample = list(inliers_base)
            utils.RandomShuffleAndResize(non_min_sample_size, rng, sample)

            non_min_model = solver.NonMinimalSolver(sample, best_model)
            if non_min_model is None:
                continue

            non_min_score = self.ScoreModel(
                solver, non_min_model, squared_inlier_threshold
            )
            best_model, best_score = self.UpdateBestModel(
                non_min_score, non_min_model, best_score, best_model
            )

            non_min_model = self.LeastSquaresFit(
                options, squared_inlier_threshold, solver, rng, non_min_model
            )

            threshold = threshold_multiplier * squared_inlier_threshold
            if options.num_lsq_iterations_ > 1:
                threshold_update = (
                    (threshold_multiplier - 1.0)
                    * squared_inlier_threshold
                    / float(options.num_lsq_iterations_ - 1)
                )
            else:
                threshold_update = 0.0

            for _ in range(int(options.num_lsq_iterations_)):
                non_min_model = self.LeastSquaresFit(
                    options, threshold, solver, rng, non_min_model
                )
                non_min_score = self.ScoreModel(
                    solver, non_min_model, squared_inlier_threshold
                )
                best_model, best_score = self.UpdateBestModel(
                    non_min_score, non_min_model, best_score, best_model
                )
                threshold -= threshold_update

        return best_model, best_score

    local_optimization = LocalOptimization

    def LeastSquaresFit(self, options, thresh, solver, rng, model):
        lsq_sample_size = (
            int(options.min_sample_multiplicator_) * int(solver.min_sample_size())
        )
        inliers = self.GetInliers(solver, model, thresh)
        if len(inliers) < int(solver.min_sample_size()):
            return model

        lsq_data_size = min(lsq_sample_size, len(inliers))
        utils.RandomShuffleAndResize(lsq_data_size, rng, inliers)
        return self._call_least_squares(solver, inliers, model)

    least_squares_fit = LeastSquaresFit

    def UpdateBestModel(self, score_curr, model_curr, score_best, model_best):
        if score_curr < score_best:
            return self._copy_model(model_curr), score_curr
        return model_best, score_best

    update_best_model = UpdateBestModel

    def _update_statistics(
        self, statistics, solver, model, squared_inlier_threshold, num_data
    ):
        statistics.inlier_indices = self.GetInliers(
            solver, model, squared_inlier_threshold
        )
        statistics.best_num_inliers = len(statistics.inlier_indices)
        statistics.inlier_ratio = float(statistics.best_num_inliers) / float(num_data)

    def _update_max_iterations(self, statistics, options, min_sample_size):
        return utils.NumRequiredIterations(
            statistics.inlier_ratio,
            1.0 - options.success_probability_,
            min_sample_size,
            options.min_num_iterations_,
            options.max_num_iterations_,
        )

    def _call_minimal_solver(self, solver, sample):
        models = solver.MinimalSolver(sample)
        if not isinstance(models, list):
            raise TypeError("MinimalSolver(sample) must return a list of models")
        return models

    def _call_least_squares(self, solver, sample, model):
        if not hasattr(solver, "LeastSquares"):
            return model
        refined_model = solver.LeastSquares(sample, self._copy_model(model))
        return model if refined_model is None else self._copy_model(refined_model)

    def _copy_model(self, model):
        try:
            return copy.deepcopy(model)
        except Exception:
            return copy.copy(model)


__all__ = [
    "RansacOptions",
    "LORansacOptions",
    "RansacStatistics",
    "RansacBase",
    "LocallyOptimizedMSAC",
]
