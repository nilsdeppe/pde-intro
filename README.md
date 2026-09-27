# Introduction to solving PDEs: finite differences → nodal discontinuous Galerkin

Two interactive [marimo](https://marimo.io) notebooks for two 75-minute lectures in a senior-level
computational physics course.

## Open in your browser (nothing to install)

| Lecture | Exercises | Solutions |
|---|---|---|
| 1 — Finite differences and the method of lines | [**open exercises**](https://nilsdeppe.github.io/pde-intro/lecture1-exercises.html) | [open solutions](https://nilsdeppe.github.io/pde-intro/lecture1-solutions.html) |
| 2 — Nodal discontinuous Galerkin methods | [**open exercises**](https://nilsdeppe.github.io/pde-intro/lecture2-exercises.html) | [open solutions](https://nilsdeppe.github.io/pde-intro/lecture2-solutions.html) |

The notebooks run entirely in your browser (Python via WebAssembly). The first load downloads Python and
the scientific packages, which can take up to a minute. Nothing you type is stored on a server: to keep your
work, download the notebook from the menu at the top right (**Export…**).

Complete the cells marked `# YOUR CODE HERE`; the coloured box under each exercise tells you whether your code
passes (amber: waiting for your code, red: something is wrong, green: all good).

## Contents

| Lecture | Topics |
|---|---|
| 1 | FD stencils (sympy derivation of the 4th-order stencil), first-order reduction of the scalar wave equation, constraint damping, initial data, method of lines with Dormand–Prince 5(4) (`scipy.integrate.solve_ivp(method="RK45")`), CFL, error-based stepping, dense output, convergence tests |
| 2 | Nodal DG derivation, LGL interpolation and differentiation, nonlinearities in nodal methods, boundary corrections (upwind, Rusanov, central), multi-element DG solver, h- and p-convergence |

## Running locally

```sh
uv sync
uv run marimo edit release/lecture1_fd/lecture1_fd.py    # exercises
uv run marimo edit source/lecture1_fd/lecture1_fd.py     # solutions
```

## Layout

- `source/<lecture>/<lecture>.py`: notebooks with solutions (between `### BEGIN SOLUTION` / `### END SOLUTION` markers).
- `release/<lecture>/<lecture>.py`: exercise notebooks, generated from `source/` by
  [mograder](https://github.com/jameskermode/mograder). Do not edit these by hand.
- `scripts/build_site.py`: exports all four notebooks as in-browser (WebAssembly) pages.
- `.github/workflows/pages.yml`: on every push to `main`, checks the notebooks, builds the site and
  publishes it to the `gh-pages` branch.

## Editing the notebooks

Edit the files in `source/`, then regenerate the exercise versions and commit both:

```sh
uv run mograder generate lecture1_fd lecture2_dg
```

This runs the solution notebooks and requires every `check()` to pass, strips the solutions, and runs the
exercise notebooks, which must have no cell errors. CI runs the same command and fails if the committed
`release/` files are out of date.

Conventions used in the notebooks:

- A solution block inside a function should be followed by `return <name>`; mograder then inserts
  `<name> = ...` so the stub returns `...`, which the `incomplete()` helper detects.
- Derivation answers use a `response_text` cell, which mograder turns into an editable Markdown cell
  (only one such cell per notebook, since marimo names must be unique).
- Inside `$$ … $$`, don't start a line with `-`, `+`, `*` or `1.` followed by a space; Markdown would turn it
  into a list and break the equation.
