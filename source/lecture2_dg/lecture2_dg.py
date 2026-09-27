# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "mograder",
#     "numpy",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from mograder.runtime import check, hint
    from numpy.polynomial import legendre
    from scipy.integrate import solve_ivp

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
    return T, check, hint, incomplete, legendre, mo, np, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lecture 2 — Solving PDEs: (pseudo)spectral and nodal discontinuous Galerkin methods

    Last time, 4th-order finite differences gave errors $\propto N^{-4}$. To gain another factor of 10 in accuracy we
    needed $10^{1/4}\approx 1.8$ times more points (and, because of the CFL condition, $\approx 3.2$ times more work in 1d).
    Today we build a method whose error falls **exponentially** with the number of points for smooth solutions:
    the **nodal discontinuous Galerkin (DG) method**. We evolve the same scalar-wave system with the same time integrator,
    so the two methods can be compared directly.

    | Part | Topic | ≈ time |
    |---|---|---|
    | 1 | The idea, and a derivation of nodal DG in 1d (derivation steps D1–D3) | 20 min |
    | 2 | Polynomial interpolation: Legendre polynomials and Legendre–Gauss–Lobatto points (Exercise 4) | 10 min |
    | 3 | Differentiating the interpolant; why nodal methods make nonlinearities easy (Exercise 5) | 10 min |
    | 4 | Boundary corrections (numerical fluxes): required properties, upwinding and dissipation | 15 min |
    | 5 | A multi-element DG solver for the scalar wave (Exercise 6) | 15 min |
    | 6 | Convergence: $h$- and $p$-refinement (Exercise 7) | 10 min |

    As before: complete the cells marked `# YOUR CODE HERE`, and watch the coloured check boxes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 1 — From finite differences to nodal DG

    **Finite differences approximate the *derivative*.** A stencil combines a few neighbouring point values
    to estimate $\partial_x u$. The error is set by the truncation of a local Taylor series, so it falls
    algebraically, $\propto\Delta x^p$.

    **Spectral methods approximate the *solution*.** Instead, represent the solution itself by a
    high-degree polynomial,

    $$
    u(x) \approx u_h(x) = \sum_{j=0}^{N} u_j\,\ell_j(x),
    $$

    and then differentiate the approximation **exactly**: $\partial_x u_h = \sum_j u_j\,\ell_j'(x)$.
    There is no separate "derivative approximation". Once we have chosen how to approximate the solution,
    its derivative follows exactly. For smooth (analytic) solutions the approximation error, and with it the derivative error,
    decreases **exponentially** with $N$.

    **Discontinuous Galerkin** combines this with the geometric flexibility of finite volumes: split the domain into
    $K$ elements, use a polynomial of degree $N$ in each one, and allow the solution to be **discontinuous** across element
    boundaries. Neighbouring elements communicate only through a *numerical flux* at their shared faces (Part 4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(r"""
    **Three knobs control a DG discretization:**

    1. the **size of each element** $\Delta x$ ($h$-refinement),
    2. the **number of points per element** $N+1$, i.e. the polynomial degree $N$ ($p$-refinement),
    3. the **location of the points** inside each element.

    Knobs 2 and 3 **together** are what give exponential convergence: increasing $N$ only helps if the points are placed
    well (Part 2 shows that equally spaced points can make things *worse*). Knob 1 on its own gives algebraic
    convergence $\propto \Delta x^{N+1}$, just like a high-order finite-difference method.
    """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Derivation of nodal DG in 1d

    We consider a system in **flux-balance form**

    $$
    \partial_t u + \partial_x F(u) = S(u).
    $$

    Our scalar-wave system is of this form with

    $$
    u = \begin{pmatrix}\Psi\\ \Pi\\ \Phi\end{pmatrix},\qquad
    F(u) = \begin{pmatrix}0\\ \Phi\\ \Pi-\gamma_2\Psi\end{pmatrix},\qquad
    S(u) = \begin{pmatrix}-\Pi\\ 0\\ -\gamma_2\Phi\end{pmatrix}.
    $$

    (Check: the $\Phi$ row reads $\partial_t\Phi + \partial_x(\Pi - \gamma_2\Psi) = -\gamma_2\Phi$.)

    **Step 1 — Elements and the reference element.** Split $[0,2\pi)$ into $K$ elements
    $\Omega_k=[x_k, x_k+\Delta x]$ and map each one to the reference element $\xi\in[-1,1]$:

    $$
    x(\xi) = x_k + \frac{1+\xi}{2}\,\Delta x, \qquad J \equiv \frac{dx}{d\xi} = \frac{\Delta x}{2}.
    $$

    **Step 2 — Nodal expansion.** Choose $N+1$ nodes $\xi_0<\dots<\xi_N$ and the Lagrange polynomials
    $\ell_j(\xi) = \prod_{m\neq j}\frac{\xi-\xi_m}{\xi_j-\xi_m}$, which satisfy $\ell_j(\xi_i) = \delta_{ij}$. In each element

    $$
    u_h(t,\xi) = \sum_{j=0}^N u_j(t)\,\ell_j(\xi),\qquad
    F_h = \sum_{j=0}^N F\big(u_j\big)\,\ell_j(\xi),\qquad
    S_h = \sum_{j=0}^N S\big(u_j\big)\,\ell_j(\xi).
    $$

    The unknowns $u_j$ are simply the **values at the nodes**. Note that $F$ and $S$ are evaluated *pointwise* at the
    nodes, however nonlinear they are. This is what makes nodal methods so convenient for nonlinear problems.

    **Step 3 — Galerkin condition.** We cannot satisfy the PDE everywhere with a polynomial, so we require the residual
    to be orthogonal to every basis function:

    $$
    \int_{\Omega_k}\ell_i(x)\left[\partial_t u_h + \partial_x F_h - S_h\right]dx = 0,\qquad i = 0,\dots,N.
    $$

    **D1 — your turn: integrate the flux term by parts** to move the derivative onto $\ell_i$ (the *weak form*).
    You will get a boundary term $\big[\ell_i\,n F\big]$ evaluated at the two faces, with outward normal $n=\mp1$ at the
    left/right face.

    **Step 4 — Numerical flux.** At a face the solution is double-valued: $u^{\rm int}$ from our element and $u^{\rm ext}$
    from the neighbour. We replace $nF$ in the boundary term by a **numerical flux** $G(u^{\rm int}, u^{\rm ext})$.
    This is the *only* place where elements talk to each other.

    **D2 — your turn: integrate by parts once more** (undoing D1 for the volume integral) to obtain the **strong form**.
    The boundary term should now contain $G - nF^{\rm int}$.

    **Step 5 — Quadrature.** Evaluate all integrals with the $N+1$-point **Legendre–Gauss–Lobatto (LGL)** quadrature rule on
    the same nodes, $\int_{-1}^{1} f\,d\xi\approx\sum_m w_m f(\xi_m)$, which is exact for polynomials of degree $\le 2N-1$.

    **D3 — your turn: show that the mass matrix** $M_{ij} = \int_{\Omega_k}\ell_i\,\ell_j\,dx$ **becomes diagonal**,
    and find its entries. (Is the quadrature exact for $\ell_i\ell_j$?)
    """)
    return


@app.cell
def _(mo):
    response_text = "**D1.** *(weak form)* ... **D2.** *(strong form)* ... **D3.** *(mass matrix)* ... Write your derivation here by editing this cell; LaTeX such as $M_{ij}$ works."
    ### BEGIN SOLUTION
    response_text = r"""
    **D1 (weak form).** $\int \ell_i\,\partial_x F_h\,dx = \big[\ell_i F_h\big]_{x_k}^{x_k+\Delta x} - \int F_h\,\partial_x\ell_i\,dx$, so

    $$
    \int_{\Omega_k}\ell_i\,(\partial_t u_h - S_h)\,dx - \int_{\Omega_k}F_h\,\partial_x\ell_i\,dx +
    \sum_{\rm faces}\ell_i\, n\,F = 0,
    $$

    where the face sum is $\ell_i(x_k+\Delta x)F(x_k+\Delta x) - \ell_i(x_k)F(x_k)$, i.e. $n=+1$ on the right face and $n=-1$ on the left.

    **D2 (strong form).** Replace $nF\to G$ in the face term, then integrate $-\int F_h\partial_x\ell_i$ by parts again:
    $-\int F_h\partial_x\ell_i = \int\ell_i\partial_x F_h - \sum_{\rm faces}\ell_i\,nF^{\rm int}$. Hence

    $$
    \int_{\Omega_k}\ell_i\,(\partial_t u_h + \partial_x F_h - S_h)\,dx + \sum_{\rm faces}\ell_i\,\big(G - nF^{\rm int}\big) = 0.
    $$

    **D3 (mass matrix).** $M_{ij} = J\int_{-1}^1\ell_i\ell_j\,d\xi \approx J\sum_m w_m\,\ell_i(\xi_m)\ell_j(\xi_m)
    = J\sum_m w_m\,\delta_{im}\delta_{jm} = J\,w_i\,\delta_{ij}$.
    The product $\ell_i\ell_j$ has degree $2N$, one more than LGL quadrature integrates exactly, so this is a (very accurate)
    approximation called *mass lumping*. Its payoff is a diagonal, trivially invertible mass matrix.
    """
    ### END SOLUTION
    mo.md(response_text)
    return (response_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Result: the semi-discrete nodal DG scheme

    With LGL nodes the endpoints $\xi_0=-1$ and $\xi_N=+1$ are nodes, so $\ell_i(-1) = \delta_{i0}$ and $\ell_i(+1)=\delta_{iN}$.
    Applying the quadrature to the strong form and dividing by the diagonal mass matrix $M_{ii}=J w_i$:

    $$
    \boxed{
    \frac{du_i}{dt} = -\frac{1}{J}\sum_{j=0}^N D_{ij}\,F(u_j) + S(u_i) -
    \frac{\delta_{i0}}{J\,w_0}\big(G - nF^{\rm int}\big)\Big|_{\rm left} -
    \frac{\delta_{iN}}{J\,w_N}\big(G - nF^{\rm int}\big)\Big|_{\rm right}
    }
    $$

    where $D_{ij} = \ell_j'(\xi_i)$ is the **differentiation matrix**. The boundary correction $G-nF^{\rm int}$ is
    "**lifted**" into the volume with weight $1/(Jw)$; with LGL nodes it only touches the two endpoint nodes. For a
    continuous exact solution the correction vanishes and we are left with $\partial_t u + \partial_x F = S$ at every node.

    The rest of this lecture builds each ingredient: good nodes (Part 2), $D_{ij}$ (Part 3), $G$ (Part 4), and then assembles
    the solver (Part 5).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 2 — Polynomial interpolation: where to put the points

    **Legendre polynomials** $P_n(\xi)$ are orthogonal on $[-1,1]$: $\int_{-1}^1 P_m P_n\,d\xi = \frac{2}{2n+1}\delta_{mn}$.
    The **Legendre–Gauss–Lobatto (LGL) points** of degree $N$ are the endpoints $\pm1$ together with the $N-1$ roots of $P_N'(\xi)$,
    and the associated quadrature weights are $w_j = \frac{2}{N(N+1)\,P_N(\xi_j)^2}$.

    **Nodal vs. modal.** The same polynomial can be written in two bases:
    $u_h = \sum_j u_j\,\ell_j(\xi)$ (**nodal**, coefficients = values at the nodes) or
    $u_h = \sum_n a_n P_n(\xi)$ (**modal**). They are related by the Vandermonde matrix $V_{in} = P_n(\xi_i)$: $u = V a$.

    The functions below are provided. The interpolation matrix uses the numerically stable
    **barycentric formula** $u_h(\xi) = \dfrac{\sum_j \frac{\lambda_j}{\xi-\xi_j}u_j}{\sum_j \frac{\lambda_j}{\xi-\xi_j}}$
    with $\lambda_j = 1/\prod_{m\ne j}(\xi_j-\xi_m)$.
    """)
    return


@app.cell
def _(legendre, np):
    def lgl_nodes_weights(N):
        """Legendre-Gauss-Lobatto nodes and weights for polynomial degree N (N + 1 points)."""
        P_N = legendre.Legendre.basis(N)
        interior = np.sort(P_N.deriv().roots().real) if N > 1 else np.array([])
        nodes = np.concatenate([[-1.0], interior, [1.0]])
        weights = 2.0 / (N * (N + 1) * P_N(nodes) ** 2)
        return nodes, weights


    def equispaced_nodes(N):
        return np.linspace(-1.0, 1.0, N + 1)


    def barycentric_weights(nodes):
        return np.array([1.0 / np.prod(xj - np.delete(nodes, j)) for j, xj in enumerate(nodes)])


    def interp_matrix(nodes, x):
        """Matrix I with (I @ u)[k] = u_h(x[k]) for nodal values u at `nodes`."""
        lam = barycentric_weights(nodes)
        diff = x[:, None] - nodes[None, :]
        on_node = np.abs(diff) < 1e-14
        diff[on_node] = 1.0
        I = lam / diff
        I /= I.sum(axis=1, keepdims=True)
        rows = on_node.any(axis=1)
        I[rows] = on_node[rows].astype(float)
        return I


    def legendre_modal_coeffs(nodes, u):
        """Legendre coefficients a_n of the interpolating polynomial: u = V a."""
        V = legendre.legvander(nodes, len(nodes) - 1)
        return np.linalg.solve(V, u)

    return (
        barycentric_weights,
        equispaced_nodes,
        interp_matrix,
        legendre_modal_coeffs,
        lgl_nodes_weights,
    )


@app.cell
def _(equispaced_nodes, interp_matrix, lgl_nodes_weights, np, plt):
    def runge(x):
        return 1.0 / (1.0 + 25.0 * x**2)


    _N = 16
    _xf = np.linspace(-1, 1, 1000)
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for _ax, (_name, _nodes) in zip(
        _axes, [("equally spaced", equispaced_nodes(_N)), ("LGL", lgl_nodes_weights(_N)[0])]
    ):
        _ax.plot(_xf, runge(_xf), "k-", lw=1, label=r"$f(x)=1/(1+25x^2)$")
        _ax.plot(_xf, interp_matrix(_nodes, _xf) @ runge(_nodes), "C3-", label=f"interpolant, N = {_N}")
        _ax.plot(_nodes, runge(_nodes), "o", color="C0", ms=4, label="nodes")
        _ax.set_title(_name + " nodes")
        _ax.set_ylim(-1.5, 1.5)
        _ax.set_xlabel("$x$")
        _ax.legend(loc="lower center")
    _fig.tight_layout()
    _fig
    return (runge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the **Runge phenomenon**: with equally spaced points, raising the degree makes the interpolant oscillate
    wildly near the ends of the interval. LGL points cluster near the endpoints (spacing $\sim 1/N^2$ there), which is
    exactly what is needed to control the interpolant everywhere. This is knob 3, the *location* of the points, at work.

    ### Exercise 4 — interpolation error versus $N$

    Complete `interp_errors(f, nodes_fn, Ns, x_fine)`: for each $N$ in `Ns`, get the nodes from `nodes_fn(N)`, interpolate
    the values `f(nodes)` to the fine grid `x_fine` with `interp_matrix`, and record the maximum absolute error.
    """)
    return


@app.cell
def _(interp_matrix, np):
    def interp_errors(f, nodes_fn, Ns, x_fine):
        """Max-norm interpolation error of f on the fine grid, for each N in Ns."""
        errs = []
        ### BEGIN SOLUTION
        for N in Ns:
            nodes = nodes_fn(N)
            errs.append(np.max(np.abs(interp_matrix(nodes, x_fine) @ f(nodes) - f(x_fine))))
        ### END SOLUTION
        return np.array(errs)

    return (interp_errors,)


@app.cell(hide_code=True)
def _(check, equispaced_nodes, interp_errors, lgl_nodes_weights, np, runge):
    _xf = np.linspace(-1, 1, 1001)
    _lgl = lambda N: lgl_nodes_weights(N)[0]
    _e_smooth = interp_errors(lambda x: np.sin(np.pi * x), _lgl, [4, 8, 16], _xf)
    ex4_done = _e_smooth.size == 3
    _checks = []
    if ex4_done:
        _e_eq = interp_errors(runge, equispaced_nodes, [20], _xf)
        _checks = [
            (np.all(np.diff(_e_smooth) < 0), "The LGL error for sin(pi x) should decrease with N."),
            (_e_smooth[-1] < 1e-10, f"LGL error for sin(pi x) at N=16 should be < 1e-10, got {_e_smooth[-1]:.2e}."),
            (_e_eq[0] > 10, "Equispaced interpolation of the Runge function at N=20 should have a large error."),
        ]
    check("Exercise 4: interpolation errors", _checks)
    return (ex4_done,)


@app.cell
def _(equispaced_nodes, ex4_done, interp_errors, lgl_nodes_weights, mo, np, plt, runge):
    mo.stop(not ex4_done, mo.callout(mo.md("⬆ Complete **Exercise 4** to see the convergence plot."), kind="warn"))
    _Ns = np.arange(2, 41, 2)
    _xf = np.linspace(-1, 1, 2001)
    _lgl = lambda N: lgl_nodes_weights(N)[0]
    _fig, _ax = plt.subplots(figsize=(6.5, 4.2))
    for _fname, _f, _c in [
        (r"$\sin(\pi x)$", lambda x: np.sin(np.pi * x), "C0"),
        (r"$1/(1+25x^2)$", runge, "C1"),
    ]:
        _ax.semilogy(_Ns, interp_errors(_f, _lgl, _Ns, _xf), "o-", color=_c, label=f"{_fname}, LGL")
        _ax.semilogy(_Ns, interp_errors(_f, equispaced_nodes, _Ns, _xf), "x--", color=_c, label=f"{_fname}, equispaced")
    _ax.set_ylim(1e-16, 1e4)
    _ax.set_xlabel("polynomial degree $N$")
    _ax.set_ylabel("max interpolation error")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    On a semilog plot **exponential convergence is a straight line**. The rate depends on the function: $\sin\pi x$ is
    entire and converges super-exponentially down to round-off, while the Runge function converges more slowly because it
    has poles at $x=\pm i/5$, close to the interval. With equally spaced points, even $\sin\pi x$ eventually diverges
    because round-off errors are amplified exponentially.

    **Modal coefficients tell you how well resolved you are.** For a smooth function the Legendre coefficients $|a_n|$ decay
    at the same exponential rate as the error. Adaptive DG codes use exactly this decay to decide where to refine
    (in $h$ or in $p$).
    """)
    return


@app.cell
def _(legendre_modal_coeffs, lgl_nodes_weights, np, plt, runge):
    _N = 40
    _nodes = lgl_nodes_weights(_N)[0]
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _n = np.arange(_N + 1)
    # Odd functions have only odd Legendre modes and even functions only even ones, so plot just those.
    for _fname, _f, _modes in [
        (r"$\sin(\pi x)$ (odd modes)", lambda x: np.sin(np.pi * x), _n % 2 == 1),
        (r"$e^{\sin(\pi x)}$ (all modes)", lambda x: np.exp(np.sin(np.pi * x)), _n >= 0),
        (r"$1/(1+25x^2)$ (even modes)", runge, _n % 2 == 0),
    ]:
        _a = legendre_modal_coeffs(_nodes, _f(_nodes))
        _ax.semilogy(_n[_modes], np.abs(_a[_modes]), "o-", ms=3, label=_fname)
    _ax.set_xlabel("mode number $n$")
    _ax.set_ylabel(r"$|a_n|$")
    _ax.set_title(f"Legendre coefficients from N = {_N} LGL nodes")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 3 — Differentiating the interpolant

    The derivative of the interpolant at the nodes is

    $$
    \partial_\xi u_h(\xi_i) = \sum_{j} u_j\,\ell_j'(\xi_i) = \sum_j D_{ij}\,u_j,
    \qquad D_{ij} = \ell_j'(\xi_i) =
    \begin{cases}
    \dfrac{\lambda_j/\lambda_i}{\xi_i-\xi_j}, & i\neq j,\\[2mm]
    -\sum_{m\ne i} D_{im}, & i=j,
    \end{cases}
    $$

    and physical derivatives follow from the chain rule, $\partial_x = J^{-1}\partial_\xi$. This is **exact** for the
    polynomial $u_h$: all of the error comes from how well $u_h$ approximates the true solution. Because the
    diagonal is chosen so that each row sums to zero, $D$ differentiates constants exactly even in floating point.
    """)
    return


@app.cell
def _(barycentric_weights, lgl_nodes_weights, mo, np):
    def diff_matrix(nodes):
        """Lagrange differentiation matrix D[i, j] = l_j'(xi_i)."""
        lam = barycentric_weights(nodes)
        diff = nodes[:, None] - nodes[None, :]
        np.fill_diagonal(diff, 1.0)
        D = (lam[None, :] / lam[:, None]) / diff
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        return D


    # Sanity check: D differentiates polynomials of degree <= N exactly.
    _nodes = lgl_nodes_weights(8)[0]
    _max_err = max(
        np.max(np.abs(diff_matrix(_nodes) @ _nodes**p - p * _nodes ** max(p - 1, 0))) for p in range(9)
    )
    mo.md(f"Sanity check: max error of $D$ applied to $\\xi^p$, $p\\le 8$, with $N=8$: **{_max_err:.1e}**")
    return (diff_matrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5 — spectral derivative of an analytic function

    Take $f(x) = e^{\sin x}$ on $[0, 2\pi]$ as a **single element** ($x = \pi(1+\xi)$, $J=\pi$).
    Complete `derivative_errors(Ns)`: for each $N$, compute $f$ at the LGL nodes, differentiate with `diff_matrix`
    (remember the $1/J$), and record the maximum error at the nodes compared with $f'(x) = \cos x\,e^{\sin x}$.

    The cell below then compares this with the 4th-order finite differences of Lecture 1, which *knows* the function is
    periodic, using the same number of points.
    """)
    return


@app.cell
def _(diff_matrix, lgl_nodes_weights, np):
    def derivative_errors(Ns):
        """Max error at the nodes of the LGL derivative of exp(sin x) on one element [0, 2pi]."""
        errs = []
        ### BEGIN SOLUTION
        for N in Ns:
            xi = lgl_nodes_weights(N)[0]
            x = np.pi * (1 + xi)
            dfdx = diff_matrix(xi) @ np.exp(np.sin(x)) / np.pi
            errs.append(np.max(np.abs(dfdx - np.cos(x) * np.exp(np.sin(x)))))
        ### END SOLUTION
        return np.array(errs)

    return (derivative_errors,)


@app.cell(hide_code=True)
def _(check, derivative_errors, np):
    _e = derivative_errors([8, 16, 48])
    ex5_done = _e.size == 3
    _checks = []
    if ex5_done:
        _checks = [
            (_e[0] > _e[1] > _e[2], "The error should decrease with N."),
            (_e[2] < 1e-10, f"At N = 48 the error should be < 1e-10, got {_e[2]:.2e} (did you divide by J?)."),
            (_e[0] > 1e-4, "At N = 8 the error looks too small; are you comparing with the right derivative?"),
        ]
    check("Exercise 5: spectral derivative", _checks)
    return (ex5_done,)


@app.cell
def _(T, derivative_errors, ex5_done, mo, np, plt):
    mo.stop(not ex5_done, mo.callout(mo.md("⬆ Complete **Exercise 5** first."), kind="warn"))


    def _fd4_error(n_points):
        _dx = T / n_points
        _x = np.arange(n_points) * _dx
        _u = np.exp(np.sin(_x))
        _du = (np.roll(_u, 2) - 8 * np.roll(_u, 1) + 8 * np.roll(_u, -1) - np.roll(_u, -2)) / (12 * _dx)
        return np.max(np.abs(_du - np.cos(_x) * _u))


    _Ns = np.arange(4, 61, 4)
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for _ax in (_ax0, _ax1):
        _ax.plot(_Ns + 1, derivative_errors(_Ns), "o-", label="LGL spectral, one element")
        _ax.plot(_Ns + 1, [_fd4_error(n) for n in _Ns + 1], "s-", label="FD4, periodic grid")
        _ax.set_xlabel("number of points")
        _ax.legend()
    _ax0.set_yscale("log")
    _ax0.set_title("semilog: spectral is a straight line")
    _ax1.set_xscale("log")
    _ax1.set_yscale("log")
    _ax1.set_xticks([5, 10, 20, 40, 60], ["5", "10", "20", "40", "60"])
    _ax1.minorticks_off()
    _ax1.set_title("log-log: FD4 is a straight line")
    _ax0.set_ylabel(r"max error in $f'$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why nodal methods make nonlinear terms easy

    Real PDEs have nonlinear fluxes and sources: $u^2$ in Burgers' equation, $\rho v^2 + p(\rho,\epsilon)$ in hydrodynamics,
    products of metric components in general relativity. In a **nodal** method we evaluate them **pointwise at the nodes**:
    $F(u)_j = F(u_j)$. That costs $\mathcal O(N)$ operations and works for *any* function, including $\sqrt{u}$, $e^u$,
    tabulated equations of state, or conditionals.

    In a **modal** method the unknowns are coefficients $a_n$. The product of two degree-$N$ series is a degree-$2N$ series
    whose coefficients require $\mathcal O(N^2)$ work (a "convolution" of Legendre coefficients), and there is no finite
    formula at all for $e^u$ or $\sqrt u$: you would transform to point values, evaluate, and transform back, which is exactly
    what the nodal method does from the start. (Pointwise evaluation of a nonlinear term does introduce a small
    *aliasing* error, which converges away at the same spectral rate for smooth solutions.)
    """)
    return


@app.cell
def _(interp_matrix, legendre, legendre_modal_coeffs, lgl_nodes_weights, mo, np):
    import timeit

    _N = 32
    _xi = lgl_nodes_weights(_N)[0]
    _u = np.exp(np.sin(np.pi * _xi))  # nodal values of u
    _a = legendre_modal_coeffs(_xi, _u)  # modal coefficients of the same polynomial
    _xf = np.linspace(-1, 1, 1001)
    _exact = np.exp(2 * np.sin(np.pi * _xf))

    _nodal_sq = _u * _u  # nodal: pointwise product, O(N)
    _modal_sq = legendre.legmul(_a, _a)[: _N + 1]  # modal: series product, then truncate to degree N

    _err_nodal = np.max(np.abs(interp_matrix(_xi, _xf) @ _nodal_sq - _exact))
    _err_modal = np.max(np.abs(legendre.legval(_xf, _modal_sq) - _exact))
    _t_nodal = timeit.timeit(lambda: _u * _u, number=2000) / 2000
    _t_modal = timeit.timeit(lambda: legendre.legmul(_a, _a)[: _N + 1], number=2000) / 2000

    mo.md(
        f"""
    Computing $u^2$ for $u = e^{{\\sin\\pi x}}$ with $N = {_N}$:

    | method | max error of $u^2$ | time per evaluation |
    |---|---|---|
    | nodal (pointwise `u * u`) | {_err_nodal:.1e} | {_t_nodal * 1e6:.2f} µs |
    | modal (`legmul` + truncation) | {_err_modal:.1e} | {_t_modal * 1e6:.2f} µs |

    Same accuracy, but the modal product is more expensive and more complicated, and it doesn't generalise to $e^u$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 4 — Boundary corrections (numerical fluxes)

    Inside each element the solution is a smooth polynomial; at a face we have two values, $u^{\rm int}$ and $u^{\rm ext}$.
    The numerical flux $G(u^{\rm int}, u^{\rm ext}; n)$ decides how they interact. It must have these properties:

    1. **Consistency:** $G(u, u; n) = n F(u)$. If the solution is continuous the correction $G - nF^{\rm int}$ vanishes,
       and the scheme reduces to the PDE.
    2. **Conservation:** $G(u^{\rm int}, u^{\rm ext}; n) = -G(u^{\rm ext}, u^{\rm int}; -n)$. What leaves one element enters its
       neighbour, so conserved quantities (mass, energy, charge, …) are conserved exactly by the discrete scheme.
    3. **Stability (dissipation):** $G$ should depend on the jump $[u] = u^{\rm ext}-u^{\rm int}$ in a way that removes
       energy from the discontinuities.

    ### Why dissipation: an energy estimate

    Take advection, $\partial_t u + a\,\partial_x u = 0$, with energy $E = \tfrac12\sum_k\int_{\Omega_k}u_h^2\,dx$, and the flux
    family $G = n\left(a\,\{u\} - \tfrac{\alpha}{2}\,n\,[u]\right)$, where $\{u\}$ is the average of the two sides.
    Multiplying the strong form by $u_h$, integrating, and adding the contributions of the two elements that share each face gives

    $$
    \frac{dE}{dt} = -\frac{\alpha}{2}\sum_{\rm faces}[u]^2 .
    $$

    * **Central flux** ($\alpha=0$): energy is conserved exactly, and nothing damps under-resolved or spurious modes.
    * **Upwind flux** ($\alpha=|a|$): energy decreases in proportion to the jumps squared.

    For a smooth, well-resolved solution the jumps are tiny ($\sim\Delta x^{N+1}$), so this dissipation does not hurt accuracy.
    It acts only on under-resolved features, where it is exactly what we want. For nonlinear problems this dissipation is
    essential for stability.

    ### Characteristic (upwind) correction for the scalar wave

    For our system the **characteristic fields** along the outward normal $n$ are

    $$
    w^\Psi = \Psi\ \ (\lambda = 0), \qquad w^\pm = \Pi \pm n\,\Phi - \gamma_2\Psi\ \ (\lambda = \pm 1),
    $$

    with inverse
    $\Psi = w^\Psi,\ \ \Pi = \tfrac12(w^++w^-) + \gamma_2 w^\Psi,\ \ \Phi = \tfrac{n}{2}(w^+-w^-)$.
    Only fields with negative speed ($\lambda<0$) **enter** the element. The upwind correction replaces exactly those fields
    by the neighbour's values:

    $$
    G - nF^{\rm int} = T\,\Lambda^-\,\big(w^{\rm ext} - w^{\rm int}\big),
    $$

    where $\Lambda^-$ keeps only the negative speeds and $T$ is the matrix of the inverse transformation above
    (both $w^{\rm ext}$ and $w^{\rm int}$ are computed with the *same*, interior, normal $n$).

    ### Rusanov (local Lax–Friedrichs) correction

    A simpler, more dissipative choice that needs no characteristic decomposition, only the largest speed $\lambda_{\max}$:

    $$
    G = \tfrac12 n\left(F^{\rm int}+F^{\rm ext}\right) - \tfrac12\lambda_{\max}\left(u^{\rm ext}-u^{\rm int}\right)
    \quad\Longrightarrow\quad
    G - nF^{\rm int} = \tfrac12 n\left(F^{\rm ext}-F^{\rm int}\right) - \tfrac12\lambda_{\max}\left(u^{\rm ext}-u^{\rm int}\right).
    $$

    Here $\lambda_{\max}=1$. For systems where computing the characteristic fields is hard (e.g. relativistic
    magnetohydrodynamics), Rusanov and HLL-type fluxes are the workhorses.

    *(In curved spacetimes or on moving meshes the normals and speeds differ on the two sides of a face; one then uses
    $T^{\rm ext}\Lambda^{-,\rm ext}w^{\rm ext} - T^{\rm int}\Lambda^{-,\rm int}w^{\rm int}$ to keep the conservation property.)*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 5 — A multi-element DG solver for the scalar wave

    **Data layout.** The solution is stored as an array `u` of shape `(3, K, N + 1)`: variable $(\Psi,\Pi,\Phi)$ × element × node.
    Face values are then arrays of shape `(3, K)`: `u[:, :, 0]` is every element's left face and `u[:, :, -1]` its right face.
    Periodicity means the neighbour to the right of element $k$ is element $k+1$ modulo $K$, i.e. `np.roll` along axis 1.

    Provided below: the grid, `flux`, `source`, the **central** correction (as an example of the interface), and `dg_rhs`,
    which assembles the scheme including the lifting. You will write

    * **Exercise 6a** `volume_terms(u, grid, gamma2)`: $-\frac1J\sum_j D_{ij}F(u_j) + S(u_i)$ for all variables, elements and nodes;
    * **Exercise 6b** `boundary_correction_upwind(u_int, u_ext, n, gamma2)`: $G - nF^{\rm int}$ from the characteristic fields;
    * **Exercise 6c** `boundary_correction_rusanov(u_int, u_ext, n, gamma2)`.
    """)
    return


@app.cell
def _(T, diff_matrix, lgl_nodes_weights, np):
    from types import SimpleNamespace


    def make_grid(K, N):
        """K equal elements covering [0, 2pi), each with N + 1 LGL nodes."""
        xi, w = lgl_nodes_weights(N)
        dx = T / K
        x = (np.arange(K) * dx)[:, None] + (1 + xi)[None, :] * dx / 2  # shape (K, N + 1)
        return SimpleNamespace(K=K, N=N, xi=xi, w=w, D=diff_matrix(xi), J=dx / 2, x=x)


    def flux(u, gamma2):
        psi, pi, phi = u
        return np.stack([np.zeros_like(psi), phi, pi - gamma2 * psi])


    def source(u, gamma2):
        psi, pi, phi = u
        return np.stack([-pi, np.zeros_like(pi), -gamma2 * phi])


    def boundary_correction_central(u_int, u_ext, n, gamma2):
        """G - n F_int for the central flux G = n (F_int + F_ext) / 2. Face arrays have shape (3, K)."""
        return 0.5 * n * (flux(u_ext, gamma2) - flux(u_int, gamma2))

    return boundary_correction_central, flux, make_grid, source


@app.cell
def _(hint):
    hint(
        r"`grid.D` has shape `(N+1, N+1)` and `F = flux(u, gamma2)` has shape `(3, K, N+1)`. "
        r"`F @ grid.D.T` computes $\sum_j D_{ij}F_j$ along the last axis for every variable and element at once.",
        r"For the upwind correction, compute $w^-$ on both sides, $w^- = \Pi - n\Phi - \gamma_2\Psi$ (same `n` for both), "
        r"and the jump $\Delta w^- = w^{-,\rm ext} - w^{-,\rm int}$. Only $\lambda_- = -1$ is negative.",
        r"$T\Lambda^-\Delta w$ with only $w^-$ incoming: the $\Psi$ component is $0$; the $\Pi$ component is "
        r"$\tfrac12\lambda_-\Delta w^-$; the $\Phi$ component is $-\tfrac n2\lambda_-\Delta w^-$. "
        r"Stack them with `np.stack([...])` to get shape `(3, K)`.",
    )
    return


@app.cell
def _(flux, source):
    def volume_terms(u, grid, gamma2):
        """-(1/J) D F(u) + S(u), shape (3, K, N + 1)."""
        ### BEGIN SOLUTION
        dt_u_volume = -(flux(u, gamma2) @ grid.D.T) / grid.J + source(u, gamma2)
        ### END SOLUTION
        return dt_u_volume

    return (volume_terms,)


@app.cell
def _(np):
    def boundary_correction_upwind(u_int, u_ext, n, gamma2):
        """Upwind G - n F_int = T Lambda^- (w_ext - w_int). Face arrays have shape (3, K); n = +1 or -1."""
        ### BEGIN SOLUTION
        w_minus_int = u_int[1] - n * u_int[2] - gamma2 * u_int[0]
        w_minus_ext = u_ext[1] - n * u_ext[2] - gamma2 * u_ext[0]
        _lam_minus = -1.0
        _dw = w_minus_ext - w_minus_int
        correction = np.stack([np.zeros_like(_dw), 0.5 * _lam_minus * _dw, -0.5 * n * _lam_minus * _dw])
        ### END SOLUTION
        return correction

    return (boundary_correction_upwind,)


@app.cell
def _(flux):
    def boundary_correction_rusanov(u_int, u_ext, n, gamma2, lambda_max=1.0):
        """Rusanov G - n F_int. Face arrays have shape (3, K); n = +1 or -1."""
        ### BEGIN SOLUTION
        correction = 0.5 * n * (flux(u_ext, gamma2) - flux(u_int, gamma2)) - 0.5 * lambda_max * (u_ext - u_int)
        ### END SOLUTION
        return correction

    return (boundary_correction_rusanov,)


@app.cell
def _(np, volume_terms):
    def dg_rhs(t, y, grid, gamma2, correction):
        """Semi-discrete strong-form nodal DG right-hand side for the scalar wave."""
        u = y.reshape(3, grid.K, grid.N + 1)
        dt_u = np.array(volume_terms(u, grid, gamma2))
        u_right, u_left = u[:, :, -1], u[:, :, 0]
        # Exterior values come from the neighbouring elements (periodic domain).
        corr_right = correction(u_right, np.roll(u_left, -1, axis=1), +1.0, gamma2)
        corr_left = correction(u_left, np.roll(u_right, 1, axis=1), -1.0, gamma2)
        # Lift the corrections: M^{-1} = 1 / (w J), and only the endpoint nodes are touched.
        dt_u[:, :, -1] -= corr_right / (grid.w[-1] * grid.J)
        dt_u[:, :, 0] -= corr_left / (grid.w[0] * grid.J)
        return dt_u.ravel()

    return (dg_rhs,)


@app.cell(hide_code=True)
def _(
    boundary_correction_central,
    boundary_correction_rusanov,
    boundary_correction_upwind,
    check,
    dg_rhs,
    flux,
    incomplete,
    make_grid,
    np,
    volume_terms,
):
    _g = make_grid(4, 8)
    _x = _g.x
    _u = np.stack([np.sin(_x), np.cos(_x), np.cos(_x)])
    _exact_dt = np.stack([-np.cos(_x), np.sin(_x), np.sin(_x)])
    _rng = np.random.default_rng(1)
    _ui, _ue = _rng.normal(size=(2, 3, 5))
    _g2 = 0.7

    vol_done = not incomplete(volume_terms(_u, _g, _g2))
    upwind_done = not incomplete(boundary_correction_upwind(_ui, _ue, 1.0, _g2))
    rusanov_done = not incomplete(boundary_correction_rusanov(_ui, _ue, 1.0, _g2))


    def _conservation(corr):
        # G(int, ext; n) = -G(ext, int; -n) with G = correction + n F_int
        _G1 = corr(_ui, _ue, 1.0, _g2) + flux(_ui, _g2)
        _G2 = corr(_ue, _ui, -1.0, _g2) - flux(_ue, _g2)
        return np.allclose(_G1, -_G2)


    _checks = []
    if vol_done:
        _v = volume_terms(_u, _g, 1.0)
        _checks += [
            (np.shape(_v) == _u.shape, f"volume_terms should return shape {_u.shape}, got {np.shape(_v)}."),
            (
                np.shape(_v) == _u.shape and np.max(np.abs(_v - _exact_dt)) < 1e-6,
                "For smooth continuous data, volume_terms should equal du/dt to spectral accuracy. Check signs and 1/J.",
            ),
        ]
    if upwind_done:
        _c = boundary_correction_upwind
        _wplus_jump = np.stack([np.zeros(5), np.ones(5), np.ones(5)])  # jump only in w^+ (n = +1)
        _checks += [
            (np.allclose(_c(_ui, _ui, 1.0, _g2), 0), "Upwind: the correction must vanish for continuous data (consistency)."),
            (_conservation(_c), "Upwind: the conservation property G(int, ext; n) = -G(ext, int; -n) fails."),
            (np.allclose(_c(_ui, _ui + _wplus_jump, 1.0, _g2), 0), "Upwind: a jump only in the outgoing field w^+ must not produce a correction."),
            (
                np.allclose(_c(_ui, _ui + np.stack([np.zeros(5), np.ones(5), -np.ones(5)]), 1.0, 0.0)[1:], [[-1.0] * 5, [1.0] * 5]),
                "Upwind: a jump of 2 in w^- (n = +1) should give a correction (0, -1, +1).",
            ),
        ]
    if rusanov_done:
        _c = boundary_correction_rusanov
        _jump_psi = np.stack([np.ones(5), np.zeros(5), np.zeros(5)])
        _checks += [
            (np.allclose(_c(_ui, _ui, -1.0, _g2), 0), "Rusanov: the correction must vanish for continuous data."),
            (_conservation(_c), "Rusanov: the conservation property fails."),
            (
                np.allclose(_c(_ui, _ui + _jump_psi, 1.0, _g2), np.stack([-0.5 * np.ones(5), np.zeros(5), -0.5 * _g2 * np.ones(5)])),
                "Rusanov: a unit jump in Psi (n = +1) should give (-1/2, 0, -gamma2/2).",
            ),
        ]
    if vol_done and upwind_done:
        for _corr in [boundary_correction_upwind, boundary_correction_central] + ([boundary_correction_rusanov] if rusanov_done else []):
            _r = dg_rhs(0.0, _u.ravel(), _g, 1.0, _corr).reshape(_u.shape)
            _checks.append((np.max(np.abs(_r - _exact_dt)) < 1e-6, f"The full DG right-hand side with {_corr.__name__} is not accurate."))
    ex6_done = vol_done and upwind_done
    check("Exercise 6: DG volume terms and boundary corrections", _checks)
    return ex6_done, rusanov_done


@app.cell
def _(dg_rhs, make_grid, np, solve_ivp):
    def exact_solution(t, x):
        """Exact (psi, pi, phi) for the right-moving wave Psi = sin(x - t), stacked."""
        return np.stack([np.sin(x - t), np.cos(x - t), np.cos(x - t)])


    def evolve_dg(K, N, t_final, correction, gamma2=1.0, rtol=1e-12, atol=1e-12, data=None):
        """Evolve the DG scalar wave; returns (sol, grid). `data(x)` optionally sets the initial data."""
        grid = make_grid(K, N)
        u0 = exact_solution(0.0, grid.x) if data is None else data(grid.x)
        sol = solve_ivp(
            dg_rhs, (0.0, t_final), u0.ravel(), method="RK45", rtol=rtol, atol=atol,
            dense_output=True, args=(grid, gamma2, correction),
        )
        return sol, grid

    return evolve_dg, exact_solution


@app.cell
def _(
    T,
    boundary_correction_central,
    boundary_correction_rusanov,
    boundary_correction_upwind,
    mo,
    rusanov_done,
):
    corrections = {"upwind": boundary_correction_upwind, "central": boundary_correction_central}
    if rusanov_done:
        corrections["Rusanov"] = boundary_correction_rusanov
    K_slider = mo.ui.slider(1, 16, value=4, label="elements $K$", show_value=True)
    Ndeg_slider = mo.ui.slider(1, 12, value=3, label="degree $N$", show_value=True)
    flux_dropdown = mo.ui.dropdown(options=list(corrections), value="upwind", label="boundary correction")
    dg_gamma2_slider = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label=r"$\gamma_2$", show_value=True)
    dg_t_slider = mo.ui.slider(0.0, 2 * T, step=2 * T / 400, value=0.25 * T, label="output time $t$", show_value=True)
    mo.vstack(
        [
            mo.hstack([K_slider, Ndeg_slider, flux_dropdown], justify="start", gap=2),
            mo.hstack([dg_gamma2_slider, dg_t_slider], justify="start", gap=2),
        ]
    )
    return (
        K_slider,
        Ndeg_slider,
        corrections,
        dg_gamma2_slider,
        dg_t_slider,
        flux_dropdown,
    )


@app.cell
def _(
    K_slider,
    Ndeg_slider,
    T,
    corrections,
    dg_gamma2_slider,
    evolve_dg,
    ex6_done,
    flux_dropdown,
    mo,
):
    mo.stop(not ex6_done, mo.callout(mo.md("⬆ Complete **Exercise 6** (at least 6a and 6b) to run the DG evolution."), kind="warn"))
    dg_sol, dg_grid = evolve_dg(
        K_slider.value, Ndeg_slider.value, 2 * T, corrections[flux_dropdown.value],
        gamma2=dg_gamma2_slider.value, rtol=1e-8, atol=1e-8,
    )
    mo.md(
        f"$K={dg_grid.K}$ elements × $N+1={dg_grid.N + 1}$ nodes $= {dg_grid.K * (dg_grid.N + 1)}$ points per variable; "
        f"**{dg_sol.t.size - 1} accepted steps**, {dg_sol.nfev} RHS evaluations (`rtol = atol = 1e-8`)."
    )
    return dg_grid, dg_sol


@app.cell
def _(T, dg_grid, dg_sol, dg_t_slider, exact_solution, interp_matrix, np, plt):
    _t = dg_t_slider.value
    _u = dg_sol.sol(_t).reshape(3, dg_grid.K, dg_grid.N + 1)
    _xi_f = np.linspace(-1, 1, 40)
    _I = interp_matrix(dg_grid.xi, _xi_f)
    _dx = T / dg_grid.K
    _names = [r"$\Psi$", r"$\Pi$", r"$\Phi$"]

    _fig, (_ax0, _ax1) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for _k in range(dg_grid.K):
        _xf = _k * _dx + (1 + _xi_f) * _dx / 2
        _ex = exact_solution(_t, _xf)
        for _i in range(3):
            _uf = _I @ _u[_i, _k]
            _ax0.plot(_xf, _uf, "-", color=f"C{_i}", lw=1.5, label=_names[_i] if _k == 0 else None)
            _ax0.plot(dg_grid.x[_k], _u[_i, _k], "o", color=f"C{_i}", ms=3)
            _ax1.semilogy(_xf, np.abs(_uf - _ex[_i]) + 1e-17, "-", color=f"C{_i}", lw=1)
        _ax0.axvline(_k * _dx, color="gray", lw=0.5)
        _ax1.axvline(_k * _dx, color="gray", lw=0.5)
    _ax0.set_title(f"t = {_t / T:.3f} T   (lines: DG polynomial in each element, dots: nodes)")
    _ax0.legend(loc="upper right")
    _ax1.set_ylabel("|DG − exact|")
    _ax1.set_xlabel("$x$")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to try with the controls:

    * Fix $K$ and raise $N$ (knobs 2 + 3): the error drops by orders of magnitude per step.
    * Fix $N$ and raise $K$ (knob 1): the error drops algebraically.
    * Compare *upwind* and *central* at low resolution (e.g. $K=4$, $N=2$): watch the jumps between elements in the error
      panel, and the number of time steps.

    ### Demo — dissipation from the boundary correction

    Evolve an **under-resolved** narrow Gaussian pulse ($K=8$, $N=3$, $\gamma_2=0$) and monitor the energy
    $E=\frac12\int(\Pi^2+\Phi^2)\,dx$, which is exactly conserved by the continuum equations.
    """)
    return


@app.cell
def _(T, corrections, evolve_dg, ex6_done, mo, np, plt):
    mo.stop(not ex6_done, mo.callout(mo.md("⬆ Complete **Exercise 6** first."), kind="warn"))


    def _pulse(x, width=0.3):
        _psi = np.exp(-(((x - np.pi) / width) ** 2))
        _dpsi = -2 * (x - np.pi) / width**2 * _psi
        return np.stack([_psi, _dpsi, _dpsi])  # right-moving: Pi = Phi = dPsi/dx


    _ts = np.linspace(0, 3 * T, 300)
    _fig, _ax = plt.subplots(figsize=(6.5, 4))
    _rows = []
    for _name, _corr in corrections.items():
        _sol, _grid = evolve_dg(8, 3, 3 * T, _corr, gamma2=0.0, rtol=1e-10, atol=1e-10, data=_pulse)
        _E = [
            0.5 * np.sum(_grid.w * _grid.J * _sol.sol(_t).reshape(3, _grid.K, _grid.N + 1)[1:] ** 2)
            for _t in _ts
        ]
        _style = {"upwind": dict(lw=4, alpha=0.5), "central": dict(lw=2), "Rusanov": dict(lw=1.5, ls="--", color="k")}
        _ax.plot(_ts / T, np.array(_E) / _E[0], label=_name, **_style[_name])
        _rows.append(f"| {_name} | {_sol.t.size - 1} |")
    _ax.set_xlabel("$t/T$")
    _ax.set_ylabel("$E(t)/E(0)$")
    _ax.set_title("Energy of an under-resolved pulse (K = 8, N = 3)")
    _ax.legend()
    _fig.tight_layout()
    mo.hstack(
        [_fig, mo.md("| correction | accepted steps |\n|---|---|\n" + "\n".join(_rows))],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * The **central** flux conserves the discrete energy to integration accuracy: the under-resolved, wrong parts of the solution
      keep sloshing around forever.
    * **Upwind** (and **Rusanov**) remove energy through the jumps, damping precisely the unresolved content. For this system, where
      every nonzero speed equals $\lambda_{\max}=1$, the two coincide in the $\Pi$ and $\Phi$ equations and differ only in
      $\Psi$, where Rusanov adds extra dissipation.
    * The central flux also needs **more time steps**. Its semi-discrete eigenvalues lie on the imaginary axis, exactly where
      Dormand–Prince is weakest (Lecture 1), whereas the upwind eigenvalues have negative real parts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Part 6 — Convergence of DG: $p$- and $h$-refinement

    We measure the $L_2$ error using the LGL quadrature on each element,
    $\|e\|_2^2 \approx \sum_k\sum_i J\,w_i\,e_{k,i}^2$ (provided below as `dg_l2_errors`). As in Lecture 1 we check all three
    variables at $t=T$ and $t=1.0927\,T$, with `rtol = atol = 1e-12`.

    ### Exercise 7 — convergence test

    Using `evolve_dg` with the **upwind** correction, fill in

    1. **$p$-convergence** (knobs 2 + 3): `p_errors[label]`, shape `(len(p_Ns), 3)`, for $K=4$ and $N$ in `p_Ns`;
    2. **$h$-convergence** (knob 1): `h_errors[N][label]`, shape `(len(h_Ks), 3)`, for each $N$ in `h_Ns` and $K$ in `h_Ks`.

    Expect a straight line on a **semilog** plot for $p$-convergence, and slope $-(N+1)$ on a **log-log** plot for
    $h$-convergence.
    """)
    return


@app.cell
def _(exact_solution, np):
    def dg_l2_errors(sol, grid, t):
        """L2 errors of (psi, pi, phi) at time t, via LGL quadrature. Returns shape (3,)."""
        _e = sol.sol(t).reshape(3, grid.K, grid.N + 1) - exact_solution(t, grid.x)
        return np.sqrt(np.sum(grid.w * grid.J * _e**2, axis=(1, 2)))

    return (dg_l2_errors,)


@app.cell
def _(T, boundary_correction_upwind, dg_l2_errors, evolve_dg, ex6_done, mo, np):
    mo.stop(not ex6_done, mo.callout(mo.md("⬆ Complete **Exercise 6** first."), kind="warn"))
    dg_t_checks = {"t = T": T, "t = 1.0927 T": 1.0927 * T}
    p_K = 4
    p_Ns = np.arange(2, 13)
    h_Ns = [2, 3, 4]
    h_Ks = np.array([2, 4, 8, 16, 32])
    p_errors = {}  # p_errors[label]: shape (len(p_Ns), 3)
    h_errors = {}  # h_errors[N][label]: shape (len(h_Ks), 3)
    # Tip: evolve each (K, N) once to 1.1 T and evaluate dg_l2_errors at both check times.
    ### BEGIN SOLUTION
    _p_runs = [evolve_dg(p_K, _N, 1.1 * T, boundary_correction_upwind) for _N in p_Ns]
    for _label, _t in dg_t_checks.items():
        p_errors[_label] = np.array([dg_l2_errors(_s, _g, _t) for _s, _g in _p_runs])
    for _N in h_Ns:
        _h_runs = [evolve_dg(_K, _N, 1.1 * T, boundary_correction_upwind) for _K in h_Ks]
        h_errors[_N] = {
            _label: np.array([dg_l2_errors(_s, _g, _t) for _s, _g in _h_runs])
            for _label, _t in dg_t_checks.items()
        }
    ### END SOLUTION
    return dg_t_checks, h_Ks, h_Ns, h_errors, p_Ns, p_errors


@app.cell(hide_code=True)
def _(check, dg_t_checks, h_Ks, h_Ns, h_errors, np, p_Ns, p_errors):
    ex7_done = len(p_errors) == len(dg_t_checks) and len(h_errors) == len(h_Ns)
    _checks = []
    if len(p_errors) == len(dg_t_checks):
        for _label in dg_t_checks:
            _e = np.asarray(p_errors[_label])
            _i10 = list(p_Ns).index(10)
            _checks.append(
                (_e.shape == (len(p_Ns), 3) and np.all(_e[_i10] < 1e-9), f"p-convergence at {_label}: errors at N = 10 should be < 1e-9.")
            )
    if len(h_errors) == len(h_Ns):
        for _N in h_Ns:
            for _label in dg_t_checks:
                _e = np.asarray(h_errors[_N][_label])
                _ok = _e.shape == (len(h_Ks), 3)
                _order = np.log2(_e[-2] / _e[-1]) if _ok else np.nan
                _checks.append(
                    (_ok and np.all(np.abs(_order - (_N + 1)) < 0.35), f"h-convergence N = {_N} at {_label}: finest orders {np.round(_order, 2)}, expected about {_N + 1}.")
                )
    check("Exercise 7: DG convergence", _checks)
    return (ex7_done,)


@app.cell
def _(dg_t_checks, ex7_done, h_Ks, h_Ns, h_errors, mo, np, p_K, p_Ns, p_errors, plt):
    mo.stop(not ex7_done, mo.callout(mo.md("⬆ Complete **Exercise 7** to see the convergence plots."), kind="warn"))
    _names = [r"$\Psi$", r"$\Pi$", r"$\Phi$"]
    _styles = {"t = T": "-", "t = 1.0927 T": ":"}
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(11, 4.3))
    for _label in dg_t_checks:
        for _i in range(3):
            _ax0.semilogy(p_Ns, p_errors[_label][:, _i], "o" + _styles[_label], color=f"C{_i}", ms=4,
                          label=f"{_names[_i]}, {_label}")
    _ax0.set_xlabel("polynomial degree $N$")
    _ax0.set_ylabel(r"$L_2$ error")
    _ax0.set_title(f"p-convergence (K = {p_K}): exponential")
    _ax0.legend(ncol=2, fontsize=8)

    _dxs = 2 * np.pi / h_Ks
    for _j, _N in enumerate(h_Ns):
        for _i in range(3):
            _ax1.loglog(_dxs, h_errors[_N]["t = 1.0927 T"][:, _i], "o-", color=f"C{_j}", ms=3, alpha=0.4 + 0.2 * _i,
                        label=f"N = {_N}" if _i == 0 else None)
        _e0 = h_errors[_N]["t = 1.0927 T"][-1, 0]
        _ax1.loglog(_dxs, _e0 * (_dxs / _dxs[-1]) ** (_N + 1), "k--", lw=0.8)
    _ax1.plot([], [], "k--", lw=0.8, label=r"$\propto\Delta x^{N+1}$")
    _ax1.set_xticks(_dxs, [f"{_d:.2f}" for _d in _dxs])
    _ax1.minorticks_off()
    _ax1.set_xlabel(r"element size $\Delta x$")
    _ax1.set_title("h-convergence at t = 1.0927 T (all 3 variables)")
    _ax1.legend(fontsize=8)
    _fig.tight_layout()

    _rows = "\n".join(
        f"| {_N} | {_N + 1} | "
        + " | ".join(
            " / ".join(f"{np.log2(h_errors[_N][_l][-2, _i] / h_errors[_N][_l][-1, _i]):.2f}" for _l in dg_t_checks)
            for _i in range(3)
        )
        + " |"
        for _N in h_Ns
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                "**Observed h-convergence orders** between the two finest grids ($t=T$ / $t=1.0927\\,T$):\n\n"
                "| $N$ | expected | $\\Psi$ | $\\Pi$ | $\\Phi$ |\n|---|---|---|---|---|\n" + _rows
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reading the plots.**

    * $p$-convergence is a straight line on the semilog plot until it reaches the **time-integration floor** set by
      `rtol = atol = 1e-12`. With $K=4$, $N=10$ ($44$ points per variable) the error is $\sim10^{-11}$. Compare Lecture 1, where
      4th-order FD with $256$ points only reached $\sim10^{-7}$.
    * $h$-refinement at fixed $N$ converges as $\Delta x^{N+1}$ (with upwind fluxes), like a high-order finite-difference method.
    * In practice one chooses $N$ as high as the smoothness of the solution warrants and uses elements (and $h$-refinement) to handle
      geometry and to isolate non-smooth regions.

    ---
    ## Summary

    * Nodal DG approximates the **solution** by a polynomial in each element and differentiates that approximation **exactly**.
    * **Three knobs:** element size, number of points, point location. Points + location (LGL) give exponential convergence;
      equally spaced points fail (Runge).
    * Nodal representation: nonlinear fluxes and sources are evaluated **pointwise**, which is cheap, simple and general.
    * Elements couple only through **boundary corrections**, which must be consistent and conservative and should be dissipative
      (upwind, Rusanov, HLL, …). The dissipation acts on the jumps, i.e. only where the solution is under-resolved.
    * With LGL nodes and a lumped (diagonal) mass matrix the scheme is
      $\dot u = -J^{-1}DF + S - \text{lift}(G - nF^{\rm int})$, which is only a few lines of NumPy.

    **Where next?** Non-smooth solutions (shocks) break exponential convergence and produce Gibbs oscillations. That is where
    limiters, finite-volume/finite-difference subcells, and the SSP time integrators mentioned in Lecture 1 come in. Curved
    elements, multiple dimensions (tensor products of the 1d operators), and moving meshes follow the same template.
    """)
    return


if __name__ == "__main__":
    app.run()
