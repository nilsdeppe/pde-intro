# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "mograder",
#     "numpy",
#     "scipy",
#     "sympy",
# ]
# mograder-assignment = "lecture1_fd"
# mograder-cell-hashes = "a2e046dd,1fa92fa9,b1534735,cdcece45,199a8b35,7c08c6bb,02374ea0,87dc323e,c0a44626,2ed6ccb9,51c70226,1b9560e2,e240144f,be943441,66a05548,27309b7a,d0f29b09,c774109a,76a50567,5753641f,ee2388cb,4bcc56c5,a59286ec,1db3e7c7,a8033d1c,3efafa04,5277b8fe,c2234757,e3b092b1,caa892c2,5d7eaa21,58d0e18d,77520892,15e88334"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from mograder.runtime import check, hint
    from scipy.integrate import RK45, solve_ivp

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
            "legend.fontsize": 9,
        }
    )

    def incomplete(*objs):
        """True if any object is an unfinished-exercise placeholder (None or ...)."""
        for obj in objs:
            if obj is None or obj is Ellipsis:
                return True
            if isinstance(obj, (tuple, list)) and incomplete(*obj):
                return True
        return False

    # Period of the traveling wave sin(x - t) on the periodic domain [0, 2π)
    T = 2 * np.pi
    return RK45, T, check, hint, incomplete, mo, np, plt, solve_ivp, sp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lecture 1 — Solving PDEs: finite differences and the method of lines

    You already know how to integrate ODEs $\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y})$.
    Today we turn a **partial** differential equation into a (large) system of ODEs and hand it to an
    ODE integrator. This is called the **method of lines**:

    $$
    \partial_t u(t,x) = \mathcal{L}[u](t,x)
    \quad\xrightarrow{\;\text{discretize } x\;}\quad
    \frac{d u_j}{dt} = \mathcal{L}_h[u]_j , \qquad u_j(t)\approx u(t, x_j).
    $$

    To make this work we need four ingredients, which define the plan for today:

    | Part | Topic | ≈ time |
    |---|---|---|
    | 1 | Finite-difference approximations of $\partial_x$ (Exercise 1: derive a 4th-order stencil with `sympy`) | 20 min |
    | 2 | The scalar wave equation as a first-order system: constraints and initial data | 15 min |
    | 3 | Evolving the wave with 4th-order finite differences + Dormand–Prince 5(4) (Exercise 2) | 15 min |
    | 4 | Time steps: the CFL condition, error-based step control, dense output | 15 min |
    | 5 | Convergence testing (Exercise 3) | 10 min |

    **How the exercises work.** Cells containing `# YOUR CODE HERE` are for you to complete.
    A coloured box below each exercise tells you whether your code passes the checks
    (amber = waiting for your code, red = something is wrong, green = all good).
    Cells further down that depend on an exercise will tell you to finish it first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1 — Finite-difference derivatives

    Place grid points $x_j = j\,h$ with spacing $h=\Delta x$ and write $u_j = u(x_j)$.
    Taylor expanding about $x_j$,

    $$
    u_{j\pm1} = u_j \pm h\,u'_j + \frac{h^2}{2}u''_j \pm \frac{h^3}{6}u'''_j + \frac{h^4}{24}u''''_j \pm \frac{h^5}{120}u^{(5)}_j + \mathcal{O}(h^6).
    $$

    **First derivative.** Subtracting the two expansions, all *even* derivatives cancel:

    $$
    u_{j+1} - u_{j-1} = 2h\,u'_j + \frac{h^3}{3}u'''_j + \mathcal{O}(h^5)
    \quad\Longrightarrow\quad
    \boxed{u'_j = \frac{u_{j+1}-u_{j-1}}{2h} \;-\; \frac{h^2}{6}u'''_j + \mathcal{O}(h^4).}
    $$

    **Second derivative.** Adding the two expansions, all *odd* derivatives cancel:

    $$
    u_{j+1} + u_{j-1} = 2u_j + h^2 u''_j + \frac{h^4}{12}u''''_j + \mathcal{O}(h^6)
    \quad\Longrightarrow\quad
    \boxed{u''_j = \frac{u_{j+1}-2u_j+u_{j-1}}{h^2} \;-\; \frac{h^2}{12}u''''_j + \mathcal{O}(h^4).}
    $$

    Both are **second order**: the error scales as $h^2$, so halving $h$ reduces the error by $4$.
    The symmetry of the centered stencil is what buys us the extra order — the one-sided (forward) difference
    $(u_{j+1}-u_j)/h = u'_j + \tfrac{h}{2}u''_j + \dots$ is only first order.

    Let's verify the orders numerically for $u = \sin x$ at $x=1$.
    """)
    return


@app.cell
def _(np, plt):
    _x0 = 1.0
    _hs = 2.0 ** -np.arange(1, 21)
    _err_d1 = np.abs((np.sin(_x0 + _hs) - np.sin(_x0 - _hs)) / (2 * _hs) - np.cos(_x0))
    _err_d2 = np.abs(
        (np.sin(_x0 + _hs) - 2 * np.sin(_x0) + np.sin(_x0 - _hs)) / _hs**2 + np.sin(_x0)
    )

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.loglog(_hs, _err_d1, "o-", label=r"$u'$: $(u_{j+1}-u_{j-1})/2h$")
    _ax.loglog(_hs, _err_d2, "s-", label=r"$u''$: $(u_{j+1}-2u_j+u_{j-1})/h^2$")
    _ax.loglog(_hs, 0.1 * _hs**2, "k--", label=r"$\propto h^2$")
    _ax.set_xlabel("$h$")
    _ax.set_ylabel("absolute error")
    _ax.set_title(r"Second-order centered stencils, $u=\sin x$ at $x=1$")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that for very small $h$ the errors stop decreasing and start growing again:
    the stencils subtract nearly equal numbers and divide by $h$ or $h^2$, so **round-off error**
    ($\sim \epsilon_{\rm mach}/h$ for $u'$, $\sim \epsilon_{\rm mach}/h^2$ for $u''$) eventually wins. Higher-order stencils reach a small
    truncation error at much larger $h$, before round-off matters.

    ### Exercise 1 — Derive the 4th-order centered first derivative with `sympy`

    Wider stencils let us cancel more terms of the Taylor series. We seek weights $a_k$ such that

    $$
    u'_j \approx \frac{1}{h}\sum_{k=-2}^{2} a_k\, u_{j+k}
    $$

    is as accurate as possible. This is the **method of undetermined coefficients**:

    1. Replace each $u_{j+k} = u(x_j + kh)$ by its Taylor series $\sum_n \frac{(kh)^n}{n!}u^{(n)}_j$.
    2. Collect the coefficient multiplying each derivative $u^{(n)}_j$.
    3. Demand that the combination equals $h\,u'_j$: the coefficients of
       $u_j, u'_j, u''_j, u'''_j, u''''_j$ must be $0, h, 0, 0, 0$. Five equations, five unknowns.
    4. Whatever is left over is the truncation error.

    Below, the symbol `du[n]` stands for $u^{(n)}(x_j)$. Fill in `taylor(k)` and then compute
    `coeffs` $=[a_{-2},a_{-1},a_0,a_1,a_2]$ and `leading_error`, the leading term of
    $\frac{1}{h}\sum_k a_k u_{j+k} - u'_j$.
    """)
    return


@app.cell
def _(hint):
    hint(
        r"Write $u(x+kh)=\sum_{n=0}^{\rm order}\frac{(kh)^n}{n!}u^{(n)}(x)$. With `du[n]` standing for "
        r"$u^{(n)}(x)$ this is a one-line `sum(...)` over `range(order + 1)`; `sp.factorial(n)` gives $n!$.",
        r"Build `combo = sp.expand(sum(a_k * taylor(k) for a_k, k in zip(a, offsets)))`, which should "
        r"approximate $h\,u'(x)$. Then `combo.coeff(du[n])` extracts the coefficient multiplying $u^{(n)}$.",
        r"Make a list of equations `sp.Eq(combo.coeff(du[n]), h if n == 1 else 0)` for `n` in `range(5)` and "
        r"solve them with `sp.solve(eqs, a)`, which returns a dictionary `{a_k: value}`.",
        r"The error is what is left of `combo.subs(solution) / h - du[1]`. After `sp.simplify` it should be "
        r"proportional to $h^4 u^{(5)}$ (why does the $h^5 u^{(6)}$ term vanish?).",
    )
    return


@app.cell
def _(sp):
    h = sp.symbols("h", positive=True)
    offsets = [-2, -1, 0, 1, 2]
    a = sp.symbols("a_m2 a_m1 a_0 a_p1 a_p2")  # the unknown weights a_k
    du = sp.symbols("u0:7")  # du[n] represents the n-th derivative of u at x_j


    def taylor(k, order=6):
        """Taylor expansion of u(x + k h) about x, through the h**order term."""
        expansion = ...
        # YOUR CODE HERE
        pass
        return expansion

    return a, du, h, offsets, taylor


@app.cell
def _(a, du, h, offsets, sp, taylor):
    # Compute `coeffs` = [a_{-2}, a_{-1}, a_0, a_1, a_2] and `leading_error`.
    # Use underscore-prefixed names (e.g. _combo) for scratch variables: marimo keeps them local to this cell.
    coeffs = ...
    leading_error = ...
    # YOUR CODE HERE
    pass
    return coeffs, leading_error


@app.cell(hide_code=True)
def _(check, coeffs, du, h, incomplete, leading_error, sp):
    _expected = [sp.Rational(1, 12), sp.Rational(-2, 3), 0, sp.Rational(2, 3), sp.Rational(-1, 12)]
    _checks = []
    if not incomplete(coeffs, leading_error):
        _checks = [
            (
                len(coeffs) == 5
                and all(sp.simplify(c - e) == 0 for c, e in zip(coeffs, _expected)),
                "The weights should be (1/12, -2/3, 0, 2/3, -1/12).",
            ),
            (
                sp.simplify(sp.expand(leading_error) + h**4 * du[5] / 30) == 0,
                "The leading error should be -h^4 u^(5)/30 (did you expand to high enough order?).",
            ),
        ]
    check("Exercise 1: fourth-order centered stencil", _checks)
    return


@app.cell(hide_code=True)
def _(coeffs, incomplete, leading_error, mo, sp):
    mo.stop(incomplete(coeffs, leading_error))
    _terms = " ".join(
        f"{'+' if c >= 0 else '-'} {sp.latex(abs(c))}\\, u_{{j{k:+d}}}".replace("{j+0}", "{j}")
        for c, k in zip(coeffs, [-2, -1, 0, 1, 2])
        if c != 0
    )
    mo.md(
        r"""
    **Result.** $\displaystyle u'_j = \frac{1}{h}\left(TERMS\right) + \mathcal{O}(h^4),\qquad$
    truncation error $= ERR$.

    Every centered stencil on a periodic grid follows the same recipe; the weights for any order and any
    derivative can be generated this way (Fornberg's algorithm does it efficiently).
    """.replace("TERMS", _terms).replace("ERR", sp.latex(leading_error))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — The scalar wave equation as a first-order system

    The (flat-space, 1d, unit speed) scalar wave equation is

    $$
    \partial_t^2 \Psi = \partial_x^2 \Psi .
    $$

    It is **second order in time**, but ODE integrators (like Dormand–Prince) solve first-order
    systems $\dot{\mathbf y} = \mathbf f(t,\mathbf y)$. You have seen the fix before: Newton's law
    $\ddot x = F/m$ becomes a first-order system for $(x, v)$. Here we introduce

    $$
    \Pi \equiv -\partial_t \Psi, \qquad \Phi \equiv \partial_x \Psi ,
    $$

    (the minus sign in $\Pi$ is a convention common in numerical relativity codes such as SpECTRE) and obtain
    the **first-order system**

    $$
    \begin{aligned}
    \partial_t \Psi &= -\Pi, \\
    \partial_t \Pi  &= -\partial_x \Phi, \\
    \partial_t \Phi &= -\partial_x \Pi + \gamma_2\left(\partial_x\Psi - \Phi\right).
    \end{aligned}
    $$

    Check: $\partial_t^2\Psi = -\partial_t\Pi = \partial_x\Phi = \partial_x^2\Psi$. ✓

    Why also introduce $\Phi$ (instead of just $\partial_t\Pi = -\partial_x^2\Psi$)? A **fully first-order**
    system only ever needs a first-derivative operator, and it is the natural setting for the
    characteristic analysis and discontinuous Galerkin methods of Lecture 2. Signals in this system
    travel with speeds $\pm1$ (and $0$), which will set our time step.

    ### The reduction constraint

    Introducing $\Phi$ added information that is not independent: a solution of the first-order system is only a
    solution of the wave equation if the **constraint**

    $$
    \mathcal{C} \equiv \partial_x\Psi - \Phi = 0
    $$

    holds. How does $\mathcal C$ evolve? Using the evolution equations,

    $$
    \partial_t \mathcal{C} = \partial_x\partial_t\Psi - \partial_t\Phi
    = -\partial_x\Pi - \left(-\partial_x\Pi + \gamma_2\,\mathcal{C}\right) = -\gamma_2\,\mathcal{C}
    \quad\Longrightarrow\quad
    \mathcal{C}(t,x) = \mathcal{C}(0,x)\,e^{-\gamma_2 t}.
    $$

    * Constraint violations **do not propagate** (their speed is zero).
    * With $\gamma_2=0$ any violation — from imperfect initial data or accumulated truncation error — **stays forever**
      (and in more complicated systems, e.g. Einstein's equations, it can grow exponentially).
    * The term $\gamma_2(\partial_x\Psi-\Phi)$ vanishes for true solutions, so it does not change the physics, but it
      **damps** violations on a time scale $1/\gamma_2$. This is **constraint damping**.
      (In 2d/3d there is a second constraint $\partial_{[i}\Phi_{j]}=0$, which is damped the same way.)
    * Don't make $\gamma_2$ huge: the damping term has eigenvalue $-\gamma_2$, which an explicit integrator must also
      resolve ($\gamma_2\Delta t \lesssim 3$).

    ### Choosing initial data

    A second-order-in-time equation needs **two** pieces of initial data: $\Psi(0,x)$ and $\partial_t\Psi(0,x)$.
    In the first-order system:

    * $\Psi(0,x)$ is free;
    * $\Pi(0,x) = -\partial_t\Psi(0,x)$ is the second free function;
    * $\Phi(0,x)$ is **not** free: it must satisfy the constraint, $\Phi(0,x) = \partial_x\Psi(0,x)$.

    The choice of $\Pi$ determines where the wave goes. By d'Alembert, $\Psi = f(x-t) + g(x+t)$, so
    $\Phi = f' + g'$ and $\Pi = f' - g'$:
    $\Pi=\Phi$ gives a purely **right-moving** wave, $\Pi=-\Phi$ a **left-moving** one, and $\Pi=0$ a standing wave.

    We will evolve the right-moving traveling wave on the periodic domain $x\in[0,2\pi)$,

    $$
    \Psi = \sin(x-t),\qquad \Pi = \cos(x-t),\qquad \Phi = \cos(x-t),
    $$

    whose period is $T=2\pi$. Knowing the exact solution at all times makes it a perfect test problem.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3 — Evolving the wave with 4th-order finite differences

    ### Exercise 2 — The semi-discrete system

    Use $N$ grid points $x_j = j\,\Delta x$, $j=0,\dots,N-1$, with $\Delta x = 2\pi/N$.
    Periodicity means $u_{N} = u_0$, $u_{-1}=u_{N-1}$, etc., which `np.roll` handles for us:
    `np.roll(u, -1)[j] == u[j+1]` and `np.roll(u, 1)[j] == u[j-1]`.

    The ODE state vector stacks the three fields, $\mathbf{y} = [\Psi_0\dots\Psi_{N-1},\ \Pi_0\dots\Pi_{N-1},\ \Phi_0\dots\Phi_{N-1}]$,
    and `np.split(y, 3)` recovers them.

    Fill in:

    1. `d_dx_fd4(u, dx)`: the 4th-order centered derivative from Exercise 1 (periodic);
    2. `initial_data(x)`: $(\Psi, \Pi, \Phi)$ at $t=0$ for the right-moving wave;
    3. `rhs(t, y, dx, gamma2)`: $d\mathbf{y}/dt$ for the first-order system, including the constraint-damping term.
    """)
    return


@app.cell
def _(np):
    def d_dx_fd4(u, dx):
        """Fourth-order centered first derivative of periodic data u with grid spacing dx.

        Hint: np.roll(u, -1)[j] == u[j + 1] and np.roll(u, 1)[j] == u[j - 1].
        """
        du_dx = ...
        # YOUR CODE HERE
        pass
        return du_dx

    return (d_dx_fd4,)


@app.cell
def _(np):
    def initial_data(x):
        """(psi, pi, phi) at t = 0 for the right-moving wave Psi = sin(x - t).

        Remember: Pi = -dPsi/dt and Phi = dPsi/dx.
        """
        psi = ...
        pi = ...
        phi = ...
        # YOUR CODE HERE
        pass
        return psi, pi, phi

    return (initial_data,)


@app.cell
def _(d_dx_fd4, np):
    def rhs(t, y, dx, gamma2):
        """dy/dt for the first-order scalar wave system on a periodic grid."""
        psi, pi, phi = np.split(y, 3)
        # Compute dt_psi, dt_pi and dt_phi using d_dx_fd4, then stack them:
        #     dt_y = np.concatenate([dt_psi, dt_pi, dt_phi])
        dt_y = ...
        # YOUR CODE HERE
        pass
        return dt_y

    return (rhs,)


@app.cell(hide_code=True)
def _(T, check, d_dx_fd4, incomplete, initial_data, np, rhs):
    def _grid(n):
        return np.arange(n) * T / n, T / n

    _checks = []
    _x32, _dx32 = _grid(32)
    _x64, _dx64 = _grid(64)
    _d_ok = not incomplete(d_dx_fd4(np.sin(_x32), _dx32))
    _id_ok = not incomplete(initial_data(_x64))
    _rhs_ok = (
        _d_ok
        and _id_ok
        and not incomplete(rhs(0.0, np.concatenate(initial_data(_x64)), _dx64, 1.0))
    )

    if _d_ok:
        _e32 = np.max(np.abs(d_dx_fd4(np.sin(_x32), _dx32) - np.cos(_x32)))
        _e64 = np.max(np.abs(d_dx_fd4(np.sin(_x64), _dx64) - np.cos(_x64)))
        _checks += [
            (_e64 < 1e-5, f"d_dx_fd4 is not accurate enough: max error {_e64:.2e} for N=64 on sin(x)."),
            (
                13 < _e32 / _e64 < 19,
                f"Halving dx should reduce the error by 2^4 = 16, but got {_e32 / _e64:.2f}.",
            ),
        ]
    if _id_ok:
        _psi, _pi, _phi = initial_data(_x64)
        _checks += [
            (np.allclose(_psi, np.sin(_x64)), "Psi(0, x) should be sin(x)."),
            (np.allclose(_pi, np.cos(_x64)), "Pi(0, x) = -dPsi/dt should be cos(x) for sin(x - t)."),
            (np.allclose(_phi, np.cos(_x64)), "Phi(0, x) = dPsi/dx should be cos(x)."),
        ]
    if _d_ok and _rhs_ok:
        _y0 = np.concatenate(initial_data(_x64))
        _exact_dt = np.concatenate([-np.cos(_x64), np.sin(_x64), np.sin(_x64)])
        _shift = np.concatenate([np.zeros(128), 1e-3 * np.ones(64)])  # constant offset in Phi
        _damp = (rhs(0.0, _y0 + _shift, _dx64, 2.0) - rhs(0.0, _y0, _dx64, 2.0))[128:]
        _checks += [
            (
                np.max(np.abs(rhs(0.0, _y0, _dx64, 1.0) - _exact_dt)) < 1e-5,
                "rhs(0, y0) does not match the exact time derivative (-cos x, sin x, sin x).",
            ),
            (
                np.allclose(_damp, -2.0 * 1e-3),
                "The constraint-damping term in dt_phi is wrong: a constant offset delta in Phi "
                "should change dt_phi by -gamma2 * delta.",
            ),
        ]
    ex2_done = _d_ok and _id_ok and _rhs_ok
    check("Exercise 2: finite-difference scalar wave", _checks)
    return (ex2_done,)


@app.cell
def _(np):
    def exact_solution(t, x):
        """Exact (psi, pi, phi) for the right-moving wave Psi = sin(x - t)."""
        return np.sin(x - t), np.cos(x - t), np.cos(x - t)

    return (exact_solution,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Time integration with Dormand–Prince 5(4)

    SciPy's `solve_ivp(method="RK45")` *is* the Dormand–Prince 5(4) embedded Runge–Kutta pair with adaptive
    step-size control. `dense_output=True` asks it to also build a continuous interpolant, `sol.sol(t)`,
    valid for any $t$ in the integration interval (more on this in Part 4).

    Use the controls to change the resolution and the constraint-damping parameter, and scrub through time.
    """)
    return


@app.cell
def _(T, mo):
    N_slider = mo.ui.slider(steps=[8, 16, 32, 64, 128, 256], value=16, label="grid points $N$")
    gamma2_slider = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label=r"$\gamma_2$")
    t_slider = mo.ui.slider(
        0.0, 2 * T, step=2 * T / 400, value=0.25 * T, label="output time $t$", show_value=True
    )
    mo.hstack([N_slider, gamma2_slider, t_slider], justify="start", gap=2)
    return N_slider, gamma2_slider, t_slider


@app.cell
def _(N_slider, T, ex2_done, gamma2_slider, initial_data, mo, np, rhs, solve_ivp):
    mo.stop(
        not ex2_done,
        mo.callout(mo.md("⬆ Complete **Exercise 2** to run the evolution."), kind="warn"),
    )
    evo_N = N_slider.value
    evo_dx = T / evo_N
    evo_x = np.arange(evo_N) * evo_dx
    evo_sol = solve_ivp(
        rhs,
        (0.0, 2 * T),
        np.concatenate(initial_data(evo_x)),
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
        dense_output=True,
        args=(evo_dx, gamma2_slider.value),
    )
    mo.md(
        f"Evolved to $t=2T$ with **{evo_sol.t.size - 1} accepted steps** and "
        f"**{evo_sol.nfev} right-hand-side evaluations** (`rtol = atol = 1e-8`)."
    )
    return evo_sol, evo_x


@app.cell
def _(T, evo_sol, evo_x, exact_solution, np, plt, t_slider):
    _t = t_slider.value
    _num = np.split(evo_sol.sol(_t), 3)
    _ex = exact_solution(_t, evo_x)
    _xf = np.linspace(0, T, 400)
    _exf = exact_solution(_t, _xf)
    _names = [r"$\Psi$", r"$\Pi$", r"$\Phi$"]

    _fig, (_ax0, _ax1) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    for _i, (_u, _name) in enumerate(zip(_num, _names)):
        _ax0.plot(_xf, _exf[_i], color=f"C{_i}", lw=1, alpha=0.6)
        _ax0.plot(evo_x, _u, "o", color=f"C{_i}", ms=4, label=_name)
        _ax1.semilogy(evo_x, np.abs(_u - _ex[_i]) + 1e-17, "o-", color=f"C{_i}", ms=3, label=_name)
    _ax0.set_title(f"t = {_t / T:.3f} T   (markers: numerical, lines: exact)")
    _ax0.legend(loc="upper right")
    _ax1.set_ylabel("|numerical − exact|")
    _ax1.set_xlabel("$x$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demo — constraint damping in action

    Now start with initial data that deliberately **violates** the constraint,
    $\Phi(0,x) = \cos x + 0.1\sin 3x \neq \partial_x\Psi(0,x)$, and monitor
    $\|\mathcal{C}\| = \|D\Psi - \Phi\|$ where $D$ is our discrete derivative.
    Because the same $D$ appears in both the constraint and the evolution equations, the semi-discrete
    constraint obeys $\dot{\mathcal C}_h = -\gamma_2\,\mathcal{C}_h$ *exactly*, so we should see a
    perfect exponential. Change $\gamma_2$ with the slider above.
    """)
    return


@app.cell
def _(T, d_dx_fd4, ex2_done, gamma2_slider, mo, np, plt, rhs, solve_ivp):
    mo.stop(not ex2_done, mo.callout(mo.md("⬆ Complete **Exercise 2** first."), kind="warn"))
    _N = 64
    _dx = T / _N
    _x = np.arange(_N) * _dx
    _y0 = np.concatenate([np.sin(_x), np.cos(_x), np.cos(_x) + 0.1 * np.sin(3 * _x)])
    _ts = np.linspace(0, T, 200)

    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _g2 in sorted({0.0, gamma2_slider.value}):
        _sol = solve_ivp(
            rhs, (0, T), _y0, method="RK45", rtol=1e-10, atol=1e-10, dense_output=True, args=(_dx, _g2)
        )
        _C = []
        for _t in _ts:
            _psi, _pi, _phi = np.split(_sol.sol(_t), 3)
            _C.append(np.sqrt(_dx * np.sum((d_dx_fd4(_psi, _dx) - _phi) ** 2)))
        _ax.semilogy(_ts / T, _C, lw=2, label=rf"numerical, $\gamma_2={_g2:.1f}$")
        _ax.semilogy(_ts / T, _C[0] * np.exp(-_g2 * _ts), "k--", lw=1)
    _ax.plot([], [], "k--", lw=1, label=r"$\|\mathcal{C}(0)\|\,e^{-\gamma_2 t}$")
    _ax.set_xlabel("$t/T$")
    _ax.set_ylabel(r"$\|\mathcal{C}\|_2$")
    _ax.set_title("Constraint violation")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 4 — Choosing the time step

    ### The CFL condition

    Insert a Fourier mode $u_j = e^{ikx_j}$ into the 4th-order stencil:

    $$
    D u_j = \frac{i}{\Delta x}\,\kappa(k\Delta x)\,u_j,\qquad
    \kappa(\theta) = \frac{8\sin\theta - \sin 2\theta}{6}, \qquad \max_\theta \kappa \approx 1.372 .
    $$

    For the wave system, each Fourier mode then behaves like the ODE $\dot y = \lambda y$ with
    $\lambda = \pm i\kappa/\Delta x$ (plus $-\gamma_2$ from the damping term). A Runge–Kutta method applied to
    $\dot y = \lambda y$ gives $y_{n+1} = R(\lambda\Delta t)\,y_n$, so the scheme is stable only if every
    $\lambda\Delta t$ lies in the **stability region** $|R(z)|\le 1$. Because $|\lambda| \propto 1/\Delta x$,
    this requires

    $$
    \Delta t \;\le\; C\,\frac{\Delta x}{|\lambda_{\rm char}|}
    \qquad\text{(Courant–Friedrichs–Lewy condition)},
    $$

    where $\lambda_{\rm char}=\pm1$ are the characteristic speeds and $C$ depends on both the spatial stencil
    and the time integrator. Physically: in one step, information must not travel farther than the stencil can "see".
    Consequence: doubling the resolution halves the time step, so the cost grows like $N^{2}$ in 1d
    ($N^{d+1}$ in $d$ dimensions).

    Below, the stability region of Dormand–Prince is computed directly from SciPy's Butcher tableau
    (`RK45.A`, `RK45.B`) via $R(z) = 1 + z\,\mathbf b^T(\mathbb 1 - z A)^{-1}\mathbf 1$,
    and overlaid with the scaled eigenvalues $\lambda\Delta t$ of our discretization.
    """)
    return


@app.cell
def _(mo):
    courant_slider = mo.ui.slider(
        0.1, 3.0, step=0.05, value=0.7, label=r"Courant number $\Delta t/\Delta x$", show_value=True
    )
    courant_slider
    return (courant_slider,)


@app.cell
def _(RK45, courant_slider, np, plt):
    _b = RK45.B
    _A = np.zeros((_b.size, _b.size))
    _A[:, : RK45.A.shape[1]] = RK45.A
    _ones = np.ones(_b.size)


    def stability_function(z):
        return 1 + z * _b @ np.linalg.solve(np.eye(_b.size) - z * _A, _ones)


    _X, _Y = np.meshgrid(np.linspace(-4.5, 1.5, 301), np.linspace(-4, 4, 301))
    _absR = np.vectorize(lambda z: abs(stability_function(z)))(_X + 1j * _Y)

    # Eigenvalues of the FD4 wave system for all Fourier modes (gamma2 = 0), scaled by dt
    _theta = np.linspace(-np.pi, np.pi, 401)
    _kappa = (8 * np.sin(_theta) - np.sin(2 * _theta)) / 6
    _zs = 1j * _kappa * courant_slider.value
    _amp = max(abs(stability_function(z)) for z in _zs)

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.contourf(_X, _Y, _absR, levels=[0, 1], colors=["C0"], alpha=0.25)
    _ax.contour(_X, _Y, _absR, levels=[1], colors=["C0"])
    _ax.plot(_zs.real, _zs.imag, "r.", ms=3, label=r"$\lambda\Delta t$ (all modes)")
    _ax.axhline(0, color="k", lw=0.5)
    _ax.axvline(0, color="k", lw=0.5)
    _ax.set_xlabel(r"Re$(z)$")
    _ax.set_ylabel(r"Im$(z)$")
    _ax.set_aspect("equal")
    _ax.set_title(f"Dormand–Prince stability region;  max |R| = {_amp:.6f}")
    _ax.legend(loc="lower left")
    _fig.tight_layout()
    _fig
    return (stability_function,)


@app.cell(hide_code=True)
def _(mo, np, stability_function):
    _ys = np.linspace(0.01, 4, 4000)
    _y_stable = _ys[np.argmax([abs(stability_function(1j * y)) > 1 + 1e-12 for y in _ys])]
    mo.md(rf"""
    The wave eigenvalues are purely imaginary, so what matters is how far the stability region extends
    **along the imaginary axis**. For Dormand–Prince, $|R(iy)|\le 1$ only up to $y\approx {_y_stable:.2f}$;
    beyond that $|R(iy)|$ exceeds $1$ only very slightly at first (by $\sim 10^{{-3}}$ at $y=1.5$), which means the
    highest-frequency modes grow slowly rather than blowing up immediately. The strict limit is therefore
    $\Delta t/\Delta x \le {_y_stable:.2f}/1.372 \approx {_y_stable / 1.372:.2f}$.
    Try Courant numbers around $0.7$, $1.5$ and $2.5$ above.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why error-based step control is the right tool for smooth solutions

    Every Dormand–Prince step computes both a 5th- and a 4th-order solution from the same stages. Their difference
    estimates the local error $\varepsilon$ of the step, which is compared with the requested tolerance
    $\text{atol} + \text{rtol}\,|y|$. The step is accepted if $\varepsilon$ is small enough, and the next step size is
    chosen as $\Delta t_{\rm new} \approx 0.9\,\Delta t\,(\text{tol}/\varepsilon)^{1/5}$.

    * You specify the **accuracy you want**, not a step size. There is no need to know $\lambda_{\max}$ or $C$.
    * The step adapts automatically as the solution changes.
    * **The CFL limit is enforced automatically.** If $\Delta t$ exceeds the stability limit, the highest-frequency
      modes start to grow, the error estimate sees this, and the step is rejected or shrunk.
      The controller ends up hovering near the stability boundary.
    * Time errors are controlled, so we can make them negligible compared with the spatial error, which is
      essential for convergence tests (Part 5).

    **Caveat — shocks.** The error estimate assumes the solution is smooth (Taylor-expandable) in time. Near a shock or
    discontinuity the estimate is large no matter how small the step is, so the controller shrinks the step
    drastically and rejects many steps. More importantly, what matters there is *nonlinear* stability
    (no new oscillations, positivity), not local truncation error. Shock-capturing codes therefore usually use
    strong-stability-preserving (SSP) Runge–Kutta methods at a fixed CFL number. We will come back to this.

    **Demo.** For a fixed tolerance, measure the typical step size the controller chooses as we refine the grid.
    On coarse grids the step is limited by **accuracy** (independent of $\Delta x$); on fine grids it becomes limited by
    **stability** ($\Delta t\propto\Delta x$). Notice that the controller settles at $\Delta t\approx\Delta x$, slightly
    *above* the strict limit: there the highest-frequency modes are amplified by only $|R|-1\sim10^{-3}$ per step,
    starting from round-off-sized amplitudes, and the error estimator keeps that growth in check.
    """)
    return


@app.cell
def _(T, ex2_done, initial_data, mo, np, plt, rhs, solve_ivp):
    mo.stop(not ex2_done, mo.callout(mo.md("⬆ Complete **Exercise 2** first."), kind="warn"))
    _Ns = np.array([16, 32, 64, 128, 256, 512, 1024, 2048])
    _median_dt = []
    _sols = {}
    for _N in _Ns:
        _dx = T / _N
        _x = np.arange(_N) * _dx
        _sols[_N] = solve_ivp(
            rhs, (0, T), np.concatenate(initial_data(_x)), method="RK45",
            rtol=1e-8, atol=1e-8, args=(_dx, 1.0),
        )
        _median_dt.append(np.median(np.diff(_sols[_N].t)))
    _dxs = T / _Ns

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10, 4))
    _ax0.loglog(_dxs, _median_dt, "o-", label="median accepted $\\Delta t$ (rtol = 1e-8)")
    _ax0.loglog(_dxs, 0.73 * _dxs, "k--", label=r"strict stability limit $\Delta t = 0.73\,\Delta x$")
    _ax0.set_xlabel(r"$\Delta x$")
    _ax0.set_ylabel(r"$\Delta t$")
    _ax0.legend()
    _ax0.set_title("small $\\Delta x$: stability-limited;  large $\\Delta x$: accuracy-limited", fontsize=10)
    for _N in [64, 512, 2048]:
        _ax1.semilogy(_sols[_N].t[1:] / T, np.diff(_sols[_N].t) / (T / _N), ".-", ms=3, label=f"N = {_N}")
    _ax1.set_xlabel("$t/T$")
    _ax1.set_ylabel(r"$\Delta t/\Delta x$")
    _ax1.set_title("accepted steps")
    _ax1.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dense output: getting the solution at the times *you* want

    The accepted step times `sol.t` are non-uniform and chosen by the controller, not by you. If you need output on a
    uniform time grid (for plots, movies, Fourier analysis) or at specific times (comparisons, convergence tests),
    **do not** force the integrator to land on them (e.g. with a tiny `max_step`, or by restarting `solve_ivp` between
    output times). That wastes work and restarts the step-size controller.

    Instead use **dense output**: Dormand–Prince comes with a free 4th-order continuous interpolant built from the stages
    it already computed. `sol.sol(t)` evaluates it at any $t$, as the time slider above does, and `t_eval=` uses the same
    interpolant internally. Its error is comparable to the error of the steps themselves.
    """)
    return


@app.cell
def _(T, ex2_done, exact_solution, initial_data, mo, np, plt, rhs, solve_ivp):
    mo.stop(not ex2_done, mo.callout(mo.md("⬆ Complete **Exercise 2** first."), kind="warn"))
    _N = 64
    _dx = T / _N
    _x = np.arange(_N) * _dx
    _sol = solve_ivp(
        rhs, (0, T), np.concatenate(initial_data(_x)), method="RK45",
        rtol=1e-4, atol=1e-4, dense_output=True, args=(_dx, 1.0),
    )
    _tu = np.linspace(0, T, 500)
    _psi0_dense = _sol.sol(_tu)[0]  # Psi at x_0 = 0 on a uniform time grid
    _psi0_steps = _sol.y[0]  # Psi at x_0 = 0 at the accepted steps

    _fig, (_ax0, _ax1) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    _ax0.plot(_tu / T, _psi0_dense, "-", label="dense output `sol.sol(t)`")
    _ax0.plot(_sol.t / T, _psi0_steps, "o", label=f"accepted steps `sol.t` ({_sol.t.size})")
    _ax0.set_ylabel(r"$\Psi(t, x=0)$")
    _ax0.legend(loc="upper right")
    _ax0.set_title("rtol = atol = 1e-4, N = 64")
    _ax1.semilogy(_tu / T, np.abs(_psi0_dense - exact_solution(_tu, 0.0)[0]), "-", label="dense output")
    _ax1.semilogy(_sol.t / T, np.abs(_psi0_steps - exact_solution(_sol.t, 0.0)[0]) + 1e-17, "o", label="steps")
    _ax1.set_xlabel("$t/T$")
    _ax1.set_ylabel("|error|")
    _ax1.legend(loc="lower right")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 5 — Convergence testing

    A code that produces pretty pictures is not necessarily correct. The single most important test of a numerical
    PDE solver is a **convergence test**: if the error behaves like $E(\Delta x)\approx C\,\Delta x^p$, then

    $$
    \frac{E(\Delta x)}{E(\Delta x/2)} = 2^p
    \quad\Longrightarrow\quad
    p = \log_2\frac{E_N}{E_{2N}},
    $$

    and $p$ must approach the design order ($4$ here) as $N$ grows. We measure errors in the discrete $L_2$ norm
    $\|e\|_2 = \big(\Delta x\sum_j e_j^2\big)^{1/2}$.

    Things to get right:

    1. **Time error must be negligible.** The spatial error is what we are testing, so we use `rtol = atol = 1e-12`.
    2. **Check every evolved variable.** A bug in the $\Phi$ equation may barely show up in $\Psi$ for a while.
    3. **Check at a generic time, not just a special one.** We check at exactly one period $t=T$ *and* at
       $t = 1.0927\,T$:
       * at $t=T$ the exact solution equals the initial data, so a code that does not evolve at all, or sends the wave
         in the **wrong direction** (a sign error in $\Pi$!), would pass;
       * $t = 1.0927\,T$ is not a step endpoint, so we are also testing the **dense-output** interpolant;
       * special times can hide errors through cancellation. Always also test at generic parameters.

    ### Exercise 3 — convergence test

    `evolve(N, t_final, ...)` (provided below) runs the solver and returns `(sol, x)`. Complete

    1. `l2_errors(sol, x, t)`: the $L_2$ errors of $(\Psi,\Pi,\Phi)$ at time $t$ (use dense output and `exact_solution`);
    2. the convergence loop: fill `errors[label]` (shape `(len(Ns), 3)`) and `orders[label]`
       (shape `(len(Ns) - 1, 3)`) for both check times.
    """)
    return


@app.cell
def _(T, initial_data, np, rhs, solve_ivp):
    def evolve(N, t_final, rtol=1e-12, atol=1e-12, gamma2=1.0, data=None):
        """Evolve the FD4 scalar wave on N points to t_final; returns (sol, x).

        `data` optionally replaces `initial_data` (a function x -> (psi, pi, phi)).
        """
        dx = T / N
        x = np.arange(N) * dx
        y0 = np.concatenate((initial_data if data is None else data)(x))
        sol = solve_ivp(
            rhs, (0.0, t_final), y0, method="RK45", rtol=rtol, atol=atol,
            dense_output=True, args=(dx, gamma2),
        )
        return sol, x

    return (evolve,)


@app.cell
def _(exact_solution, np):
    def l2_errors(sol, x, t):
        """L2 norms of the errors in (psi, pi, phi) at time t. Returns an array of shape (3,)."""
        dx = x[1] - x[0]
        errors_t = ...
        # YOUR CODE HERE
        pass
        return errors_t

    return (l2_errors,)


@app.cell
def _(T, evolve, ex2_done, l2_errors, mo, np):
    mo.stop(not ex2_done, mo.callout(mo.md("⬆ Complete **Exercise 2** first."), kind="warn"))
    Ns = np.array([16, 32, 64, 128, 256])
    t_checks = {"t = T": T, "t = 1.0927 T": 1.0927 * T}
    errors = {}  # errors[label]: array of shape (len(Ns), 3)
    orders = {}  # orders[label]: array of shape (len(Ns) - 1, 3)
    # Tip: evolve each resolution once to a time past both check times and reuse it.
    # YOUR CODE HERE
    pass
    return Ns, errors, orders, t_checks


@app.cell(hide_code=True)
def _(T, check, evolve, ex2_done, exact_solution, incomplete, l2_errors, np, orders, t_checks):
    _checks = []
    _l2_ok = False
    if ex2_done:
        _sol, _x = evolve(32, 1.2 * T, rtol=1e-10, atol=1e-10)
        _e = l2_errors(_sol, _x, 1.2 * T)
        _l2_ok = not incomplete(_e)
        if _l2_ok:
            _dx = _x[1] - _x[0]
            _ref = [
                np.sqrt(_dx * np.sum((u - v) ** 2))
                for u, v in zip(np.split(_sol.sol(1.2 * T), 3), exact_solution(1.2 * T, _x))
            ]
            _checks.append((np.allclose(_e, _ref, rtol=1e-6), "l2_errors does not return the L2 errors of (psi, pi, phi)."))
    if _l2_ok and len(orders) == len(t_checks):
        for _label in t_checks:
            _p = np.asarray(orders[_label])
            _checks.append(
                (
                    _p.shape[1:] == (3,) and np.all(np.abs(_p[-1] - 4) < 0.2),
                    f"At {_label} the finest-resolution orders should be close to 4, got {np.round(_p[-1], 2)}.",
                )
            )
    ex3_done = _l2_ok and len(orders) == len(t_checks)
    check("Exercise 3: convergence test", _checks if ex3_done or _checks else [])
    return (ex3_done,)


@app.cell
def _(Ns, errors, ex3_done, mo, np, orders, plt, t_checks):
    mo.stop(not ex3_done, mo.callout(mo.md("⬆ Complete **Exercise 3** to see the convergence plot."), kind="warn"))
    _names = [r"$\Psi$", r"$\Pi$", r"$\Phi$"]
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for _ax, _label in zip(_axes, t_checks):
        for _i in range(3):
            _ax.loglog(Ns, errors[_label][:, _i], "o-", label=_names[_i])
        _ax.loglog(Ns, errors[_label][0, 0] * (Ns / Ns[0]) ** -4.0, "k--", label=r"$\propto N^{-4}$")
        _ax.set_title(_label)
        _ax.set_xlabel("$N$")
        _ax.legend()
    _axes[0].set_ylabel(r"$L_2$ error")
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {Ns[_k]} → {Ns[_k + 1]} | "
        + " | ".join(" / ".join(f"{orders[_l][_k, _i]:.2f}" for _l in t_checks) for _i in range(3))
        + " |"
        for _k in range(len(Ns) - 1)
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                "**Observed orders** $\\log_2(E_N/E_{2N})$, shown as ($t=T$ / $t=1.0927\\,T$):\n\n"
                "| resolutions | $\\Psi$ | $\\Pi$ | $\\Phi$ |\n|---|---|---|---|\n" + _rows
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demo — what a bug looks like at a special time

    Suppose we got the sign of $\Pi$ wrong in the initial data, $\Pi(0,x) = -\cos x$. This produces the **left-moving**
    wave $\sin(x+t)$. Compare it against the (right-moving) exact solution:
    """)
    return


@app.cell
def _(T, evolve, ex3_done, l2_errors, mo, np):
    mo.stop(not ex3_done, mo.callout(mo.md("⬆ Complete **Exercise 3** first."), kind="warn"))
    _sol, _x = evolve(64, 1.1 * T, data=lambda x: (np.sin(x), -np.cos(x), np.cos(x)))
    _eT = l2_errors(_sol, _x, T)
    _eG = l2_errors(_sol, _x, 1.0927 * T)
    mo.md(
        "| check time | $\\Psi$ error | $\\Pi$ error | $\\Phi$ error |\n|---|---|---|---|\n"
        f"| $t=T$ | {_eT[0]:.1e} | {_eT[1]:.1e} | {_eT[2]:.1e} |\n"
        f"| $t=1.0927\\,T$ | {_eG[0]:.1e} | {_eG[1]:.1e} | {_eG[2]:.1e} |\n\n"
        "At $t=T$ the $\\Psi$ and $\\Phi$ errors look as good as a correct code's; only a generic time reveals the "
        "bug unambiguously. (Here $\\Pi$ happens to catch it at $t=T$ too, but for many bugs no variable would.)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Demo — the time-integration error floor

    Repeat the convergence test with a loose tolerance `rtol = atol = 1e-6`. Once the spatial error drops below the
    time-integration error, **convergence stalls**, and it can even look like the code gets *worse* with resolution
    (finer grids force smaller, more numerous steps, each contributing error). Whenever a convergence test
    misbehaves, first make sure you are not hitting a floor from another error source (time stepping, round-off,
    boundary treatment, …).
    """)
    return


@app.cell
def _(Ns, T, errors, evolve, ex3_done, l2_errors, mo, np, plt):
    mo.stop(not ex3_done, mo.callout(mo.md("⬆ Complete **Exercise 3** first."), kind="warn"))
    _loose = np.array([l2_errors(*evolve(_N, 1.1 * T, rtol=1e-6, atol=1e-6), 1.0927 * T) for _N in Ns])
    _names = [r"$\Psi$", r"$\Pi$", r"$\Phi$"]
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    for _i in range(3):
        _ax.loglog(Ns, errors["t = 1.0927 T"][:, _i], "o-", color=f"C{_i}", label=f"{_names[_i]}, tol = 1e-12")
        _ax.loglog(Ns, _loose[:, _i], "x--", color=f"C{_i}", label=f"{_names[_i]}, tol = 1e-6")
    _ax.set_xlabel("$N$")
    _ax.set_ylabel(r"$L_2$ error at $t=1.0927\,T$")
    _ax.legend(ncol=2)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    * **Method of lines:** discretize space → large ODE system → use a good ODE integrator.
    * **Finite differences** approximate the *derivative* from nearby point values; a stencil of width $2m+1$
      gives order $2m$, derived by matching Taylor series (Exercise 1).
    * Second-order-in-time equations are reduced to **first order**, which introduces **constraints**; we damp their
      violations with $\gamma_2$. Initial data must satisfy the constraints; the free data choose the physics
      (e.g. the direction of propagation).
    * Explicit time stepping obeys a **CFL condition** $\Delta t\lesssim C\,\Delta x$. Error-based adaptive stepping
      (Dormand–Prince 5(4)) picks the step for you and respects stability automatically for smooth solutions.
      Use **dense output** for output at chosen times.
    * **Always** do convergence tests: all variables, generic times, time error ≪ space error.

    **Next time:** our errors fall like $N^{-4}$. Can we do better? By approximating the **solution** itself with
    high-degree polynomials instead of approximating derivatives, we will get errors that fall *exponentially* with $N$:
    the discontinuous Galerkin method.
    """)
    return


if __name__ == "__main__":
    app.run()
