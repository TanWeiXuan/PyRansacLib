"""
Unofficial Python port of RansacLib/utils.h from RansacLib.

Original project:
    RansacLib
    Copyright (c) 2019, Torsten Sattler
    License: BSD 3-Clause

Unofficial Python port:
    Copyright (c) 2026 Tan Wei Xuan

See LICENSE and NOTICE for details.
"""

import math
import random
from collections.abc import Sequence


def _as_rng(rng):
    return rng if rng is not None else random


def _is_nested_sequence(value):
    return bool(value) and all(isinstance(item, list) for item in value)


def RandomShuffle(*args):
    """In-place Fisher-Yates shuffle.

    Accepts either ``RandomShuffle(sample)`` or the C++-like
    ``RandomShuffle(rng, sample)`` form. ``rng`` only needs to provide
    ``randrange(start, stop)``.
    """
    if len(args) == 1:
        rng = random
        sample = args[0]
    elif len(args) == 2:
        rng, sample = args
    else:
        raise TypeError("RandomShuffle expects sample or rng, sample")

    rng = _as_rng(rng)
    for i in range(max(0, len(sample) - 1)):
        idx = rng.randrange(i, len(sample))
        sample[i], sample[idx] = sample[idx], sample[i]
    return sample


def random_shuffle(*args):
    return RandomShuffle(*args)


def RandomShuffleAndResize(*args):
    """Shuffle a sample in place and resize it.

    Supported forms:
      * ``RandomShuffleAndResize(target_size, sample)``
      * ``RandomShuffleAndResize(target_size, rng, sample)``
      * ``RandomShuffleAndResize(sample_sizes, rng, nested_sample)``

    The nested variants are kept for API compatibility with ``utils.h`` even
    though this port intentionally implements only LO-MSAC/MSAC.
    """
    if len(args) == 2:
        target_size, random_sample = args
        rng = random
    elif len(args) == 3:
        target_size, rng, random_sample = args
    else:
        raise TypeError(
            "RandomShuffleAndResize expects target_size, sample or "
            "target_size, rng, sample"
        )

    rng = _as_rng(rng)

    if isinstance(target_size, Sequence) and not isinstance(target_size, (str, bytes)):
        for i, sample_size in enumerate(target_size):
            if i >= len(random_sample):
                break
            RandomShuffleAndResize(min(len(random_sample[i]), sample_size), rng, random_sample[i])
        return random_sample

    target_size = int(target_size)
    if target_size < 0:
        raise ValueError("target_size must be non-negative")

    if _is_nested_sequence(random_sample):
        flat_data = []
        for data_type, values in enumerate(random_sample):
            for value in values:
                flat_data.append((data_type, value))
            values.clear()
        RandomShuffle(rng, flat_data)
        del flat_data[target_size:]
        for data_type, value in flat_data:
            random_sample[data_type].append(value)
        return random_sample

    RandomShuffle(rng, random_sample)
    del random_sample[target_size:]
    return random_sample


def random_shuffle_and_resize(*args):
    return RandomShuffleAndResize(*args)


def NumRequiredIterations(
    inlier_ratio,
    prob_missing_best_model,
    sample_size,
    min_iterations,
    max_iterations,
):
    """Compute the adaptive RANSAC iteration count from ``utils.h``."""
    min_iterations = int(min_iterations)
    max_iterations = int(max_iterations)

    if isinstance(inlier_ratio, Sequence) and not isinstance(inlier_ratio, (str, bytes)):
        prob_all_inlier_sample = 1.0
        for ratio, size in zip(inlier_ratio, sample_size):
            prob_all_inlier_sample *= math.pow(float(ratio), float(size))
    else:
        ratio = float(inlier_ratio)
        if ratio <= 0.0:
            return max_iterations
        if ratio >= 1.0:
            return min_iterations
        prob_all_inlier_sample = math.pow(ratio, float(sample_size))

    if prob_all_inlier_sample <= 0.0:
        return max_iterations
    if prob_all_inlier_sample >= 1.0:
        return min_iterations

    prob_non_inlier_sample = 1.0 - prob_all_inlier_sample
    if prob_non_inlier_sample >= 0.99999999999999:
        return max_iterations

    if prob_missing_best_model <= 0.0:
        return max_iterations
    if prob_missing_best_model >= 1.0:
        return min_iterations

    log_numerator = math.log(prob_missing_best_model)
    log_denominator = math.log(prob_non_inlier_sample)
    num_iters = math.ceil(log_numerator / log_denominator + 0.5)
    num_req_iterations = min(int(num_iters), max_iterations)
    return max(min_iterations, num_req_iterations)


def num_required_iterations(*args):
    return NumRequiredIterations(*args)


__all__ = [
    "RandomShuffle",
    "RandomShuffleAndResize",
    "NumRequiredIterations",
    "random_shuffle",
    "random_shuffle_and_resize",
    "num_required_iterations",
]
