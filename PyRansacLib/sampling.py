"""
Unofficial Python port of RansacLib/sampling.h from RansacLib.

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

try:
    from . import utils
except ImportError:  # pragma: no cover - supports direct module execution.
    import utils


class UniformSampling:
    """Uniform minimal sampler for RANSAC.

    This mirrors ``UniformSampling`` from ``sampling.h`` while using Python's
    standard-library Mersenne Twister implementation.
    """

    def __init__(self, random_seed, solver):
        self.rng_ = random.Random(random_seed)
        self.num_data_ = int(solver.num_data())
        self.sample_size_ = int(solver.min_sample_size())
        if self.num_data_ <= 0:
            raise ValueError("solver.num_data() must be positive")
        if self.sample_size_ <= 0:
            raise ValueError("solver.min_sample_size() must be positive")
        if self.sample_size_ > self.num_data_:
            raise ValueError("sample size cannot exceed the number of data points")
        self.draw_sample_ = self.DrawBetterThanShuffle(
            self.sample_size_, self.num_data_
        )

    def Sample(self, random_sample=None):
        """Draw a minimal sample.

        If ``random_sample`` is provided, it is mutated in place. The sampled
        list is returned in all cases for Python convenience.
        """
        if self.draw_sample_:
            sample = self.DrawSample()
        else:
            sample = self.ShuffleSample()

        if random_sample is not None:
            random_sample[:] = sample
            return random_sample
        return sample

    sample = Sample

    def DrawBetterThanShuffle(self, sample_size, num_elements):
        if sample_size >= num_elements:
            return False
        coeff = float(num_elements) / float(num_elements - sample_size)
        return coeff < math.e

    draw_better_than_shuffle = DrawBetterThanShuffle

    def DrawSample(self, random_sample=None):
        sample = []
        for _ in range(self.sample_size_):
            value = self.rng_.randrange(0, self.num_data_)
            while value in sample:
                value = self.rng_.randrange(0, self.num_data_)
            sample.append(value)

        if random_sample is not None:
            random_sample[:] = sample
            return random_sample
        return sample

    draw_sample = DrawSample

    def ShuffleSample(self, random_sample=None):
        sample = list(range(self.num_data_))
        if self.sample_size_ != self.num_data_:
            self.RandomShuffle(sample)
            del sample[self.sample_size_:]

        if random_sample is not None:
            random_sample[:] = sample
            return random_sample
        return sample

    shuffle_sample = ShuffleSample

    def RandomShuffle(self, random_sample):
        return utils.RandomShuffle(self.rng_, random_sample)

    random_shuffle = RandomShuffle


__all__ = ["UniformSampling"]
