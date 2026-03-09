"""Sod shock tube: numerical solver using Lax-Friedrichs scheme.

1D Euler equations を Lax-Friedrichs flux + Forward Euler で解く
最小限の実装. 圧縮性流体コードの学習用.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

import exact_solution

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_UNIVERSAL = 8.314_462_62  # J/(mol·K)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Shock tube の計算条件."""

    # 左状態
    p_L: float
    T_L: float
    u_L: float
    # 右状態
    p_R: float
    T_R: float
    u_R: float
    # 気体物性
    MW: float
    gamma: float
    # 計算領域
    x_left: float
    x_right: float
    x_center: float
    dx: float
    n_ghost: int
    # 時間
    cfl: float
    t_max: float
    out_interval: int

    @property
    def rho_L(self) -> float:
        return self.p_L * self.MW / (R_UNIVERSAL * self.T_L)

    @property
    def rho_R(self) -> float:
        return self.p_R * self.MW / (R_UNIVERSAL * self.T_R)


# ---------------------------------------------------------------------------
# 状態変数の変換
# ---------------------------------------------------------------------------
# 保存変数 U = [rho, rho*u, rho*E]
# 基本変数 W = [rho, u, p]


def conservative_to_primitive(U: np.ndarray, gamma: float) -> np.ndarray:
    """保存変数から基本変数へ変換する.

    U = [rho, rho*u, rho*E] から圧力を状態方程式で復元し,
    W = [rho, u, p] を返す.

    Args:
        U:     1セルの保存変数ベクトル (3,).
        gamma: 比熱比.

    Returns:
        基本変数ベクトル [rho, u, p] (3,).
    """
    rho = U[0]
    u = U[1] / U[0]
    p = (gamma - 1.0) * (U[2] - 0.5 * U[1] ** 2 / U[0])
    return np.array([rho, u, p])


def primitive_to_conservative(W: np.ndarray, gamma: float) -> np.ndarray:
    """基本変数から保存変数へ変換する.

    W = [rho, u, p] から運動量とエネルギーを計算し,
    U = [rho, rho*u, rho*E] を返す.

    Args:
        W:     1セルの基本変数ベクトル (3,).
        gamma: 比熱比.

    Returns:
        保存変数ベクトル [rho, rho*u, rho*E] (3,).
    """
    rho, u, p = W[0], W[1], W[2]
    rho_u = rho * u
    rho_E = p / (gamma - 1.0) + 0.5 * rho * u ** 2
    return np.array([rho, rho_u, rho_E])


def compute_flux(U: np.ndarray, gamma: float) -> np.ndarray:
    """保存変数から Euler 方程式のフラックスベクトル F(U) を計算する.

    F = [rho*u, rho*u^2 + p, u*(rho*E + p)]

    Args:
        U:     1セルの保存変数ベクトル (3,).
        gamma: 比熱比.

    Returns:
        フラックスベクトル (3,).
    """
    p = (gamma - 1.0) * (U[2] - 0.5 * U[1] ** 2 / U[0])
    F0 = U[1]                          # rho * u
    F1 = U[1] ** 2 / U[0] + p          # rho * u^2 + p
    F2 = U[1] / U[0] * (U[2] + p)      # u * (rho*E + p)
    return np.array([F0, F1, F2])


# ---------------------------------------------------------------------------
# 音速
# ---------------------------------------------------------------------------
def sound_speed(p: float, rho: float, gamma: float) -> float:
    """等エントロピー音速を計算する.

    a = sqrt(gamma * p / rho)

    Args:
        p:     圧力 [Pa].
        rho:   密度 [kg/m^3].
        gamma: 比熱比.

    Returns:
        音速 [m/s].
    """
    return np.sqrt(gamma * p / rho)


# ---------------------------------------------------------------------------
# Lax-Friedrichs flux
# ---------------------------------------------------------------------------
def lax_friedrichs_flux(U_L: np.ndarray, U_R: np.ndarray,
                        gamma: float) -> np.ndarray:
    """セル界面での Lax-Friedrichs 数値フラックスを計算する.

    F_LF = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)
    alpha = max(|u_L| + a_L, |u_R| + a_R)

    最も単純な Riemann ソルバ. 拡散が大きいが安定.

    Args:
        U_L:   界面左側の保存変数ベクトル (3,).
        U_R:   界面右側の保存変数ベクトル (3,).
        gamma: 比熱比.

    Returns:
        数値フラックスベクトル (3,).
    """
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)

    a_L = sound_speed(W_L[2], W_L[0], gamma)
    a_R = sound_speed(W_R[2], W_R[0], gamma)
    alpha = max(abs(W_L[1]) + a_L, abs(W_R[1]) + a_R)

    return 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)


# ---------------------------------------------------------------------------
# 時間刻み幅
# ---------------------------------------------------------------------------
def compute_dt(U_arr: np.ndarray, dx: float, cfl: float,
               gamma: float) -> float:
    """CFL 条件を満たす時間刻み幅を計算する.

    全セルの最大波速 (|u| + a) を求め, dt = cfl * dx / max_speed とする.

    Args:
        U_arr: 全セルの保存変数配列 (n_points, 3).
        dx:    セル幅 [m].
        cfl:   CFL 数 (0 < cfl <= 1).
        gamma: 比熱比.

    Returns:
        時間刻み幅 dt [s].
    """
    max_speed = 0.0
    for i in range(len(U_arr)):
        W = conservative_to_primitive(U_arr[i], gamma)
        rho, u, p = W[0], W[1], W[2]
        a = sound_speed(p, rho, gamma)
        speed = abs(u) + a
        if speed > max_speed:
            max_speed = speed
    return cfl * dx / max_speed


# ---------------------------------------------------------------------------
# 初期条件
# ---------------------------------------------------------------------------
def create_initial_condition(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """区分一定の初期条件を生成する.

    x_center を境に左右の状態を設定し, 基本変数から保存変数へ変換する.

    Args:
        cfg: 計算条件.

    Returns:
        x:     座標配列 (n_points,). ゴーストセル含む.
        U_arr: 保存変数配列 (n_points, 3).
    """
    n_cells = int((cfg.x_right - cfg.x_left) / cfg.dx) + 1
    n_points = n_cells + 2 * cfg.n_ghost

    # 座標 (ゴーストセル含む)
    x = np.zeros(n_points)
    for i in range(n_points):
        x[i] = cfg.x_left - cfg.n_ghost * cfg.dx + cfg.dx * i

    # 基本変数 → 保存変数
    U_arr = np.zeros((n_points, 3))
    for i in range(n_points):
        if x[i] < cfg.x_center:
            W = np.array([cfg.rho_L, cfg.u_L, cfg.p_L])
        else:
            W = np.array([cfg.rho_R, cfg.u_R, cfg.p_R])
        U_arr[i] = primitive_to_conservative(W, cfg.gamma)

    return x, U_arr


# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------
def get_temperature(p: float, rho: float, MW: float) -> float:
    """理想気体の状態方程式から温度を計算する.

    T = p * MW / (rho * R)

    Args:
        p:   圧力 [Pa].
        rho: 密度 [kg/m^3].
        MW:  分子量 [kg/mol].

    Returns:
        温度 [K].
    """
    return p * MW / (rho * R_UNIVERSAL)


def make_exact_config(cfg: Config) -> exact_solution.ShockTubeConfig:
    """num.Config から exact_solution.ShockTubeConfig を生成する.

    Args:
        cfg: 数値計算用の Config.

    Returns:
        解析解ソルバ用の ShockTubeConfig.
    """
    return exact_solution.ShockTubeConfig(
        p_L=cfg.p_L, T_L=cfg.T_L, u_L=cfg.u_L,
        p_R=cfg.p_R, T_R=cfg.T_R, u_R=cfg.u_R,
        MW=cfg.MW, gamma=cfg.gamma, x_center=cfg.x_center,
    )


def plot_fields(x: np.ndarray, U_arr: np.ndarray, t: float,
                cfg: Config) -> None:
    """数値解と解析解を比較する 4 パネルプロットを PNG に保存する.

    数値解を黒実線, 解析解を赤破線で重ねて描画する.
    ファイル名は "{t:.6f}.png" 形式.

    Args:
        x:     座標配列 (n_points,).
        U_arr: 保存変数配列 (n_points, 3).
        t:     現在時刻 [s].
        cfg:   計算条件 (gamma, MW の参照用).
    """
    n_points = len(x)

    # 数値解: 保存変数 → 基本変数 + 温度
    rho_arr = np.zeros(n_points)
    p_arr = np.zeros(n_points)
    u_arr = np.zeros(n_points)
    T_arr = np.zeros(n_points)

    for i in range(n_points):
        W = conservative_to_primitive(U_arr[i], cfg.gamma)
        rho_arr[i] = W[0]
        u_arr[i] = W[1]
        p_arr[i] = W[2]
        T_arr[i] = get_temperature(W[2], W[0], cfg.MW)

    # 解析解
    exact_cfg = make_exact_config(cfg)
    exact_fields = exact_solution.solve(exact_cfg, x, t)

    width = 190.0 / 25.4
    height = width * 3.0 / 4.0
    fig, axes = plt.subplots(2, 2, figsize=(width, height), tight_layout=True)

    for ax, (num, ex, label) in zip(
        axes.flat,
        [
            (rho_arr, exact_fields.rho, r"$\rho$ [kg/m$^3$]"),
            (p_arr, exact_fields.p, r"$p$ [Pa]"),
            (T_arr, exact_fields.T, r"$T$ [K]"),
            (u_arr, exact_fields.u, r"$u$ [m/s]"),
        ],
    ):
        ax.plot(x, num, "k-", label="Numerical")
        ax.plot(x, ex, "r--", label="Exact")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", alpha=0.4)

    fig.suptitle(rf"$t$ = {t:.4e} s", fontsize=12)
    fig.savefig(f"{t:.6f}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------
def solve(cfg: Config) -> None:
    """Lax-Friedrichs + Forward Euler で shock tube を時間発展させる.

    1. 初期条件を生成
    2. 各タイムステップで:
       - CFL 条件から dt を決定
       - 各セル界面の数値フラックスを計算
       - Forward Euler で保存変数を更新
       - ゼロ勾配境界条件を適用
    3. out_interval ステップごとにプロット出力

    Args:
        cfg: 計算条件.
    """
    x, U_arr = create_initial_condition(cfg)
    plot_fields(x, U_arr, t=0.0, cfg=cfg)

    i_start = cfg.n_ghost
    i_end = len(x) - cfg.n_ghost

    t = 0.0
    step = 0

    while t < cfg.t_max:
        dt = compute_dt(U_arr, cfg.dx, cfg.cfl, cfg.gamma)

        # 内部セルの更新
        U_new = np.copy(U_arr)
        for i in range(i_start, i_end):
            F_left = lax_friedrichs_flux(U_arr[i - 1], U_arr[i], cfg.gamma)
            F_right = lax_friedrichs_flux(U_arr[i], U_arr[i + 1], cfg.gamma)
            L = -(F_right - F_left) / cfg.dx
            U_new[i] = U_arr[i] + dt * L

        # 境界条件 (ゼロ勾配)
        for g in range(cfg.n_ghost):
            U_new[g] = U_new[i_start]
            U_new[-(g + 1)] = U_new[i_end - 1]

        U_arr = U_new
        t += dt
        step += 1
        print(f"step={step:6d}, t={t:.6e}")

        if step % cfg.out_interval == 0:
            plot_fields(x, U_arr, t, cfg)


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = Config(
        p_L=1e5, T_L=348.24, u_L=0.0,
        p_R=1e4, T_R=278.24, u_R=0.0,
        MW=28.96e-3, gamma=1.4,
        x_left=-5.0, x_right=5.0, x_center=0.0,
        dx=1e-3, n_ghost=1,
        cfl=0.9, t_max=0.01, out_interval=100,
    )
    solve(cfg)


if __name__ == "__main__":
    main()
