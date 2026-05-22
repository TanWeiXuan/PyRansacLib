# PyRansacLib

PyRansacLib is an unofficial, Python-only algorithmic port of Torsten Sattler's
[RansacLib](https://github.com/tsattler/RansacLib). It reimplements the
LO-MSAC / MSAC solver from the original C++ `ransac.h` using only the Python
standard library.

The goal of this project is to keep the public structure, logic, and algorithmic
flow close to the original implementation while using idiomatic Python
interfaces. It is not intended to be an exact clone of the C++ template API.
There are no runtime dependencies outside the standard library.

## Scope

Implemented:

- LO-MSAC as described in Lebeda, Matas, and Chum, *Fixing the Locally Optimized
  RANSAC*, BMVC 2012.
- MSAC-style scoring with a truncated quadratic cost as used by Torr and
  Zisserman, *Robust computation and parametrization of multiple view relations*,
  ICCV 1998.
- Uniform minimal sampling and RANSAC iteration count utilities from the
  original headers.

Intentionally out of scope:

- HybridRANSAC.
- Concrete geometric solvers such as pose estimation. As in the original
  RansacLib, users provide a solver object for their own model.

## Basic Usage

```python
from PyRansacLib.ransac import LORansacOptions, LocallyOptimizedMSAC

options = LORansacOptions(
    squared_inlier_threshold_=1.0,
    min_num_iterations_=100,
    max_num_iterations_=10000,
    random_seed_=0,
)

best_model, statistics = LocallyOptimizedMSAC().estimate_model(options, solver)
```

See `tests/test_line_fitting.py` for a complete dependency-free line-fitting
solver and example usage.

## Solver Protocol

A solver object should provide this Python-style interface:

```python
class MySolver:
    def min_sample_size(self) -> int:
        ...

    def non_minimal_sample_size(self) -> int:
        ...

    def num_data(self) -> int:
        ...

    def MinimalSolver(self, sample) -> list[Model]:
        ...

    def NonMinimalSolver(self, sample, initial_model) -> Model | None:
        ...

    def EvaluateModelOnPoint(self, model, point_idx) -> float:
        ...
```

`MinimalSolver(sample)` must return a list of candidate models. Return `[]` if no
valid model could be produced. Returning a single bare model and C++-style output
list mutation are intentionally unsupported.

`NonMinimalSolver(sample, initial_model)` must return a refined model on success
and `None` on failure. Mutable model holders and status/model tuples are
intentionally unsupported.

`EvaluateModelOnPoint(model, point_idx)` returns the squared residual for one
data point. Models only need to be copyable by Python's standard `copy` module.

Solvers may also provide an optional `LeastSquares(sample, model) -> Model | None`
method. If omitted, the least-squares refinement steps leave the input model
unchanged. Returning `None` is accepted and means no refinement was applied.

This port intentionally avoids C++-style output parameters in user solver
interfaces.

To run plain MSAC-style estimation, set:

```python
options.num_lo_steps_ = 0
```

## Testing

Run the test suite with:

```bash
python -m compileall -q PyRansacLib tests
python -m unittest discover -v tests
```

The tests use only `unittest` and the Python standard library.

## Attribution

Original project:

- RansacLib by Torsten Sattler and contributors
- Copyright (c) 2019, Torsten Sattler
- BSD 3-Clause License

Unofficial Python port:

- Copyright (c) 2026 Tan Wei Xuan

This project follows the structure and algorithms of the original project where
practical, but it is a separate Python implementation. See `LICENSE` for the
license text.

## Citing

If you use this library in scientific work, please cite the original RansacLib
repository and the relevant method papers:

```bibtex
@misc{Sattler2019Github,
  title = {{RansacLib - A Template-based *SAC Implementation}},
  author = {Torsten Sattler and others},
  URL = {https://github.com/tsattler/RansacLib},
  year = {2019}
}
```

```bibtex
@inproceedings{Lebeda2012BMVC,
  title = {{Fixing the Locally Optimized RANSAC}},
  author = {Karel Lebeda and Jiri Matas and Ondrej Chum},
  booktitle = {British Machine Vision Conference (BMVC)},
  year = {2012}
}
```

```bibtex
@inproceedings{Torr1998ICCV,
  title = {{Robust computation and parametrization of multiple view relations}},
  author = {Phil H. S. Torr and Andrew Zisserman},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year = {1998}
}
```
