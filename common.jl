"""
Sod shock tube: 共通定義.

全ステップで共有する定数, Config, 状態変数変換, 初期条件生成, プロット機能.
"""

include("exact_solution.jl")
using .ExactSolution
using Printf
using CairoMakie
using StaticArrays

const Vec3 = SVector{3, Float64}

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
const R_UNIVERSAL = 8.314_462_62  # J/(mol·K)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

"""Shock tube の計算条件."""
struct Config
    # 左状態
    p_L::Float64
    T_L::Float64
    u_L::Float64
    # 右状態
    p_R::Float64
    T_R::Float64
    u_R::Float64
    # 物性
    MW::Float64
    gamma::Float64
    # 計算領域
    x_left::Float64
    x_right::Float64
    x_center::Float64
    dx::Float64
    n_ghost::Int
    # 時間
    cfl::Float64
    t_max::Float64
    out_interval::Int
    # プロット軸範囲 (rho, p, T, u) の (ymin, ymax)
    ylims::Vector{Tuple{Float64, Float64}}
end

"""理想気体の密度 rho = p * MW / (R * T)."""
function get_rho(p::Float64, T::Float64, MW::Float64)::Float64
    return p * MW / (R_UNIVERSAL * T)
end

# ---------------------------------------------------------------------------
# 状態変数の変換
# ---------------------------------------------------------------------------
# 保存変数 U = [rho, rho*u, rho*E]
# 基本変数 W = [rho, u, p]

"""
保存変数から基本変数へ変換する.

U = [rho, rho*u, rho*E] から圧力を状態方程式で復元し,
W = [rho, u, p] を返す.

# Args
- `U`:     1セルの保存変数ベクトル (長さ3).
- `gamma`: 比熱比.

# Returns
基本変数ベクトル [rho, u, p] (長さ3).
"""
function conservative_to_primitive(U::Vec3, gamma::Float64)::Vec3
    rho = U[1]
    u = U[2] / U[1]
    p = (gamma - 1.0) * (U[3] - 0.5 * U[2]^2 / U[1])
    return Vec3(rho, u, p)
end

"""
基本変数から保存変数へ変換する.

W = [rho, u, p] から運動量とエネルギーを計算し,
U = [rho, rho*u, rho*E] を返す.

# Args
- `W`:     1セルの基本変数ベクトル (長さ3).
- `gamma`: 比熱比.

# Returns
保存変数ベクトル [rho, rho*u, rho*E] (長さ3).
"""
function primitive_to_conservative(W::Vec3, gamma::Float64)::Vec3
    rho, u, p = W[1], W[2], W[3]
    rho_u = rho * u
    rho_E = p / (gamma - 1.0) + 0.5 * rho * u^2
    return Vec3(rho, rho_u, rho_E)
end

"""
保存変数から Euler 方程式のフラックスベクトル F(U) を計算する.

F = [rho*u, rho*u^2 + p, u*(rho*E + p)]

# Args
- `U`:     1セルの保存変数ベクトル (長さ3).
- `gamma`: 比熱比.

# Returns
フラックスベクトル (長さ3).
"""
function compute_flux(U::Vec3, gamma::Float64)::Vec3
    p = (gamma - 1.0) * (U[3] - 0.5 * U[2]^2 / U[1])
    F1 = U[2]                          # rho * u
    F2 = U[2]^2 / U[1] + p            # rho * u^2 + p
    F3 = U[2] / U[1] * (U[3] + p)     # u * (rho*E + p)
    return Vec3(F1, F2, F3)
end

# ---------------------------------------------------------------------------
# 音速
# ---------------------------------------------------------------------------

"""
等エントロピー音速を計算する.

a = sqrt(gamma * p / rho)

# Args
- `p`:     圧力 [Pa].
- `rho`:   密度 [kg/m^3].
- `gamma`: 比熱比.

# Returns
音速 [m/s].
"""
function sound_speed(p::Float64, rho::Float64, gamma::Float64)::Float64
    return sqrt(gamma * p / rho)
end

# ---------------------------------------------------------------------------
# 時間刻み幅
# ---------------------------------------------------------------------------

"""
CFL 条件を満たす時間刻み幅を計算する.

全セルの最大波速 (|u| + a) を求め, dt = cfl * dx / max_speed とする.

# Args
- `U_arr`: 全セルの保存変数配列 (n_points × 3 の Vector{Vec3}).
- `dx`:    セル幅 [m].
- `cfl`:   CFL 数 (0 < cfl <= 1).
- `gamma`: 比熱比.

# Returns
時間刻み幅 dt [s].
"""
function compute_dt(U_arr::Vector{Vec3}, dx::Float64,
                    cfl::Float64, gamma::Float64)::Float64
    max_speed = 0.0
    @inbounds for i in eachindex(U_arr)
        W = conservative_to_primitive(U_arr[i], gamma)
        rho, u, p = W[1], W[2], W[3]
        a = sound_speed(p, rho, gamma)
        speed = abs(u) + a
        if speed > max_speed
            max_speed = speed
        end
    end
    return cfl * dx / max_speed
end

# ---------------------------------------------------------------------------
# 初期条件
# ---------------------------------------------------------------------------

"""
初期条件を生成する.

# Args
- `cfg`: 計算条件.

# Returns
- `x`:     座標配列. ゴーストセル含む.
- `U_arr`: 保存変数の配列 (Vector{Vec3}).
"""
function create_initial_condition(cfg::Config)
    n_cells = round(Int, (cfg.x_right - cfg.x_left) / cfg.dx) + 1
    n_points = n_cells + 2 * cfg.n_ghost

    rho_L = get_rho(cfg.p_L, cfg.T_L, cfg.MW)
    rho_R = get_rho(cfg.p_R, cfg.T_R, cfg.MW)

    # 座標 (ゴーストセル含む)
    x = Vector{Float64}(undef, n_points)
    for i in 1:n_points
        x[i] = cfg.x_left - cfg.n_ghost * cfg.dx + cfg.dx * (i - 1)
    end

    # 基本変数 → 保存変数
    U_arr = Vector{Vec3}(undef, n_points)
    for i in 1:n_points
        if x[i] < cfg.x_center
            W = Vec3(rho_L, cfg.u_L, cfg.p_L)
        else
            W = Vec3(rho_R, cfg.u_R, cfg.p_R)
        end
        U_arr[i] = primitive_to_conservative(W, cfg.gamma)
    end

    return x, U_arr
end

include("plotting.jl")
