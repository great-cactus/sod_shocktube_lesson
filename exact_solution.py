"""Sod shock tube: exact Riemann solver.

Usage as module::

    from exact_solution import ShockTubeConfig, solve

    cfg = ShockTubeConfig(
        p_L=1e5, T_L=348.24, u_L=0.0,
        p_R=1e4, T_R=278.24, u_R=0.0,
        MW=28.96e-3, gamma=1.4,
    )
    x = np.linspace(-5, 5, 500)
    fields = solve(cfg, x, t=0.005)
    # fields.rho, fields.p, fields.T, fields.u
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

R_UNIVERSAL = 8.314_462_62  # J/(mol·K)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GasState:
    """Primitive gas variables at a single location."""

    rho: float
    p: float
    u: float
    gamma: float

    @property
    def sound_speed(self) -> float:
        return np.sqrt(self.gamma * self.p / self.rho)


@dataclass(frozen=True)
class ShockTubeConfig:
    """Parameters that define a shock-tube problem."""

    p_L: float
    T_L: float
    u_L: float
    p_R: float
    T_R: float
    u_R: float
    MW: float
    gamma: float
    x_center: float = 0.0

    @property
    def rho_L(self) -> float:
        return self.p_L * self.MW / (R_UNIVERSAL * self.T_L)

    @property
    def rho_R(self) -> float:
        return self.p_R * self.MW / (R_UNIVERSAL * self.T_R)

    @property
    def state_L(self) -> GasState:
        return GasState(self.rho_L, self.p_L, self.u_L, self.gamma)

    @property
    def state_R(self) -> GasState:
        return GasState(self.rho_R, self.p_R, self.u_R, self.gamma)


@dataclass(frozen=True)
class FieldArrays:
    """1-D solution fields on a given grid."""

    x: np.ndarray
    rho: np.ndarray
    p: np.ndarray
    T: np.ndarray
    u: np.ndarray


# ---------------------------------------------------------------------------
# Riemann solver internals
# ---------------------------------------------------------------------------
def _pressure_function(p: float, p_k: float, rho_k: float, c_k: float,
                       gamma: float, mu2: float) -> float:
    """Evaluate the pressure function f_k(p) for one side of the Riemann fan.

    Selects the shock (Rankine-Hugoniot) or rarefaction (isentropic) branch
    depending on whether p exceeds the reference pressure p_k.  (Toro, ch. 4)

    Args:
        p:     Trial pressure in the star region [Pa].
        p_k:   Reference (undisturbed) pressure on side k [Pa].
        rho_k: Reference density on side k [kg/m^3].
        c_k:   Reference sound speed on side k [m/s].
        gamma: Ratio of specific heats.
        mu2:   (gamma-1)/(gamma+1), pre-computed for efficiency.

    Returns:
        Scalar value of f_k(p).
    """
    if p > p_k:
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = mu2 * p_k
        return (p - p_k) * np.sqrt(A / (p + B))
    return 2.0 * c_k / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _density_behind(p_star: float, p_k: float, rho_k: float,
                    gamma: float, mu2: float) -> float:
    """Compute the post-wave density on one side of the contact surface.

    Uses Rankine-Hugoniot jump relation for shocks (p_star > p_k)
    or the isentropic relation for rarefactions (p_star <= p_k).

    Args:
        p_star: Star-region pressure [Pa].
        p_k:    Reference (undisturbed) pressure on side k [Pa].
        rho_k:  Reference density on side k [kg/m^3].
        gamma:  Ratio of specific heats.
        mu2:    (gamma-1)/(gamma+1).

    Returns:
        Post-wave density [kg/m^3].
    """
    r = p_star / p_k
    if p_star > p_k:
        return rho_k * (r + mu2) / (mu2 * r + 1.0)
    return rho_k * r ** (1.0 / gamma)


def _solve_star_region(cfg: ShockTubeConfig):
    """Find the star-region state by solving the nonlinear pressure equation.

    Uses Brent's method to find p_star satisfying f_L(p) + f_R(p) + du = 0,
    then derives u_star and the post-wave densities on both sides.

    Args:
        cfg: Shock tube problem configuration.

    Returns:
        (p_star, u_star, rho_star_L, rho_star_R) — star-region pressure [Pa],
        velocity [m/s], and left/right post-wave densities [kg/m^3].
    """
    gamma = cfg.gamma
    mu2 = (gamma - 1.0) / (gamma + 1.0)

    sL, sR = cfg.state_L, cfg.state_R
    p_L, rho_L, c_L, u_L = sL.p, sL.rho, sL.sound_speed, sL.u
    p_R, rho_R, c_R, u_R = sR.p, sR.rho, sR.sound_speed, sR.u

    def equation(p):
        fL = _pressure_function(p, p_L, rho_L, c_L, gamma, mu2)
        fR = _pressure_function(p, p_R, rho_R, c_R, gamma, mu2)
        return fL + fR + (u_R - u_L)

    p_min = min(p_L, p_R)
    p_max = max(p_L, p_R)
    if equation(p_min) * equation(p_max) < 0:
        p_star = brentq(equation, p_min, p_max)
    else:
        p_star = brentq(equation, 1e-6, 10.0 * p_max)

    fL_star = _pressure_function(p_star, p_L, rho_L, c_L, gamma, mu2)
    u_star = u_L - fL_star
    rho_star_L = _density_behind(p_star, p_L, rho_L, gamma, mu2)
    rho_star_R = _density_behind(p_star, p_R, rho_R, gamma, mu2)

    return p_star, u_star, rho_star_L, rho_star_R


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def initial_condition(cfg: ShockTubeConfig, x: np.ndarray) -> FieldArrays:
    """Create piece-wise constant initial fields (t = 0) on grid *x*.

    Args:
        cfg: Shock tube problem configuration.
        x:   1-D spatial grid [m], shape (N,).

    Returns:
        FieldArrays with rho, p, T, u initialised to left/right states.
    """
    left = x < cfg.x_center
    return FieldArrays(
        x=x,
        rho=np.where(left, cfg.rho_L, cfg.rho_R),
        p=np.where(left, cfg.p_L, cfg.p_R),
        T=np.where(left, cfg.T_L, cfg.T_R),
        u=np.where(left, cfg.u_L, cfg.u_R),
    )


def solve(cfg: ShockTubeConfig, x: np.ndarray, t: float) -> FieldArrays:
    """Compute the exact Riemann solution at time *t* on grid *x*.

    Solves for the star region once, then samples all five regions
    (left, rarefaction fan, star-left, star-right, right) using the
    self-similar variable xi = (x - x_center) / t.

    Args:
        cfg: Shock tube problem configuration.
        x:   1-D spatial grid [m], shape (N,).
        t:   Evaluation time [s].  If t <= 0, returns initial_condition.

    Returns:
        New immutable FieldArrays with rho, p, T, u at every grid point.
    """
    if t <= 0.0:
        return initial_condition(cfg, x)

    gamma = cfg.gamma
    sL, sR = cfg.state_L, cfg.state_R
    p_L, rho_L, c_L, u_L = sL.p, sL.rho, sL.sound_speed, sL.u
    p_R, rho_R, c_R, u_R = sR.p, sR.rho, sR.sound_speed, sR.u

    p_star, u_star, rho_star_L, rho_star_R = _solve_star_region(cfg)
    c_star_L = np.sqrt(gamma * p_star / rho_star_L)

    # Wave speeds (in similarity variable xi = (x - x0) / t)
    S_HL = u_L - c_L                                       # left rarefaction head
    S_TL = u_star - c_star_L                                # left rarefaction tail
    if p_star > p_R:                                        # right shock
        S_R = u_R + c_R * np.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * (p_star / p_R - 1.0) + 1.0
        )
    else:                                                   # right rarefaction
        c_star_R = np.sqrt(gamma * p_star / rho_star_R)
        S_R = u_star + c_star_R

    xi = (x - cfg.x_center) / t

    # --- Rarefaction fan values (computed for all x; masked by np.select) ---
    u_fan = (2.0 / (gamma + 1.0)) * (c_L + 0.5 * (gamma - 1.0) * u_L + xi)
    c_fan = c_L - 0.5 * (gamma - 1.0) * (u_fan - u_L)
    rho_fan = rho_L * (c_fan / c_L) ** (2.0 / (gamma - 1.0))
    p_fan = p_L * (rho_fan / rho_L) ** gamma

    # --- Assemble by region (left to right) ---
    # default = Region 5 (undisturbed left)
    conditions = [
        (xi > S_HL) & (xi <= S_TL),    # rarefaction fan
        (xi > S_TL) & (xi <= u_star),   # star-left
        (xi > u_star) & (xi <= S_R),    # star-right
        xi > S_R,                       # undisturbed right
    ]
    rho = np.select(conditions, [rho_fan, rho_star_L, rho_star_R, rho_R], default=rho_L)
    p_out = np.select(conditions, [p_fan, p_star, p_star, p_R], default=p_L)
    u_out = np.select(conditions, [u_fan, u_star, u_star, u_R], default=u_L)
    T_out = p_out * cfg.MW / (rho * R_UNIVERSAL)

    return FieldArrays(x=x, rho=rho, p=p_out, T=T_out, u=u_out)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def main() -> None:
    import matplotlib.pyplot as plt

    cfg = ShockTubeConfig(
        p_L=1e5, T_L=348.24, u_L=0.0,
        p_R=1e4, T_R=278.24, u_R=0.0,
        MW=28.96e-3, gamma=1.4,
    )

    x = np.linspace(-5.0, 5.0, 500)
    fields = solve(cfg, x, t=0.005)

    width = 190.0 / 25.4
    height = width * 3.0 / 4.0
    fig, axes = plt.subplots(2, 2, figsize=(width, height), tight_layout=True)

    for ax, (arr, label) in zip(
        axes.flat,
        [
            (fields.rho, r"$\rho$ [kg/m³]"),
            (fields.p, r"$p$ [Pa]"),
            (fields.T, r"$T$ [K]"),
            (fields.u, r"$u$ [m/s]"),
        ],
    ):
        ax.plot(fields.x, arr, "k-")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(label)
        ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle(r"Exact Riemann Solution ($t = 0.005$ s)", fontsize=12)
    fig.savefig("exact.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
