# chebyshev_inductive

Chebyshev semi-iteration for an SPD system `A x = b`, written two ways --
**with** inductively defined functions and **without** -- in the style of
`apps/iir_cascade`. It is a worked example of using an inductive Func for an
*iterative solver* recurrence.

## The recurrence

Chebyshev semi-iteration has a single-sequence three-term form

```
x_{k+1} = (1 + w_k) x_k - w_k x_{k-1} + a_k (b - A x_k)
```

The coefficients `a_k`, `w_k` come only from the spectral bounds `[lmin, lmax]`
(no inner products), so a step is exactly **one mat-vec** plus a bounded-lag,
same- component combination:

- one reduction per step (the mat-vec) -> fits an inductive update's single
  RDom;
- bounded lag (2) -> folds to O(1) columns of storage;
- the mat-vec `A x_k` is a self-reference at a *shifted* iteration index, read
  at every component.

## Inductive vs. non-inductive

The generator (`chebyshev_inductive_generator.cpp`) has a
`GeneratorParam<bool> inductive`, exactly like `iir_cascade`:

Both versions express the *same* three-term recurrence, and both keep only three
live columns of iterate history. The difference is who manages that folding.

- **inductive = true** -- the iterate `X` is one Func with a (non-inductive)
  pure definition plus a single update. The Chebyshev step is a **mat-vec
  recurrence**: column `k` reads the previous iterate at *every* component, so
  the self-reference `X(j, k-1)` puts a component index where the output
  component sits. That index cannot be the inductive (monotonically-decreasing)
  one -- so the **output component `i` and the mat-vec component `j` are BOTH
  reduction variables** of a 2-D `RDom`, nested inside the one inductive
  variable `k`. Only `k` recurses (with a bounded lag of 2), and
  `fold_storage(k, 3)` lets Halide keep three live columns automatically. The
  endpoint `x_M` is extracted with the iteration index as the outermost loop.

- **inductive = false** -- the identical recurrence written as an ordinary 3-D
  reduction scan (`RDom` over `i`, `j`, `k`), with the iteration dimension
  folded **by hand** into a 3-column ring: storage is indexed `k % 3`, the
  coefficient tables are indexed by the true `k-1`, and the mat-vec accumulation
  is seeded at `j == 0` with an assignment (since the reused ring slot still
  holds the stale `x_{k-3}`, there is no pure-def zero to accumulate onto). This
  makes the storage comparison fair -- both versions use O(n) working memory.
  Note the hand-rolled ring hard-codes the lag-2 window (size 3) and `M % 3`
  endpoint addressing, whereas `fold_storage` is a general knob.

`test.cpp` runs both, checks they agree with each other and with a plain-C++
reference, and reports timing. With storage equalised, the two are effectively a
**tie** on this small CPU case (~1.3 ms each): a mat-vec recurrence has no large
*foldable* intermediate with a clean forward consumer -- the whole previous
column must be live either way -- so folding does not win here, it just avoids
the penalty of materialising the full trajectory. The inductive form's advantage
over a hand-rolled ring would show only where the automatic fold buys something
the manual index cannot (larger problems, GPU), and its real value is expressing
the fold without hard-coding the window.

## Build & run

With CMake (from the apps build):

```
cmake --build . --target chebyshev_inductive_test
ctest -R chebyshev_inductive_test
```

or with the Makefile:

```
make test
```
