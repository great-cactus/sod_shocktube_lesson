"""
1D Euler equations を Riemann solver + Forward Euler で解く
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
    for i in eachindex(U_arr)
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
- `U_arr`: 保存変数の配列 (Vector{Vector{Float64}}).
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

# ---------------------------------------------------------------------------
# Riemman Solver
# ---------------------------------------------------------------------------

function HLL(U_L::Vec3, U_R::Vec3, gamma::Float64)::Vec3
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)
    a_L = sound_speed(W_L[3], W_L[1], gamma)
    a_R = sound_speed(W_R[3], W_R[1], gamma)
    u_L = W_L[2]
    u_R = W_R[2]

    S_L = min(u_L, u_R) - max(a_L, a_R)
    S_R = max(u_L, u_R) + max(a_L, a_R)

    if S_L > 0.0
        F_HLL = F_L
    elseif S_R < 0.0
        F_HLL = F_R
    else
        F_HLL = (S_R * F_L - S_L * F_R + S_R * S_L * (U_R - U_L)) / (S_R - S_L)
    end

    return F_HLL
end

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

"""
Lax-Friedrichs + Forward Euler で shock tube を時間発展させ, MP4 動画を出力する.

1. 初期条件を生成
2. 各タイムステップで:
   - CFL 条件から dt を決定
   - 各セル界面の数値フラックスを計算
   - Forward Euler で保存変数を更新
   - ゼロ勾配境界条件を適用
3. out_interval ステップごとに動画フレームを記録

# Args
- `cfg`:      計算条件.
- `filename`: 出力ファイル名 (デフォルト "movie.mp4").
- `fps`:      フレームレート (デフォルト 30).
"""
function solve(cfg::Config; filename::String="movie.mp4", fps::Int=30)
    x, U_arr = create_initial_condition(cfg)

    fig, obs_num, obs_exact, title_obs = create_figure(x, cfg)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    # フレームデータを収集 (時刻, U_arr のスナップショット)
    frames = Tuple{Float64, Vector{Vec3}}[]
    push!(frames, (0.0, copy(U_arr)))

    U_buf = copy(U_arr)
    t = 0.0
    step = 0

    while t < cfg.t_max
        dt = compute_dt(U_arr, cfg.dx, cfg.cfl, cfg.gamma)

        # 内部セルの更新
        copyto!(U_buf, U_arr)
        @inbounds for i in i_start:i_end
            F_left = HLL(U_arr[i-1], U_arr[i], cfg.gamma)
            F_right = HLL(U_arr[i], U_arr[i+1], cfg.gamma)
            L = -(F_right - F_left) / cfg.dx

            # Forward Euler method
            U_buf[i] = U_arr[i] + dt * L
        end

        # 境界条件 (ゼロ勾配)
        for g in 1:cfg.n_ghost
            U_buf[g] = U_buf[i_start]
            U_buf[end - g + 1] = U_buf[i_end]
        end

        U_arr, U_buf = U_buf, U_arr
        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, copy(U_arr)))
        end
    end

    # 動画を記録
    println("Recording $(length(frames)) frames to $filename ...")
    record(fig, filename, frames; framerate=fps) do (t_frame, U_frame)
        update_observables!(obs_num, obs_exact, title_obs, x, U_frame, t_frame, cfg)
    end
    println("Done: $filename")
end

# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
function main()
    cfg = Config(
        1e5, 348.24, 0.0,       # p_L, T_L, u_L
        1e4, 278.24, 0.0,       # p_R, T_R, u_R
        28.96e-3, 1.4,          # MW, gamma
        -5.0, 5.0, 0.0,         # x_left, x_right, x_center
        1e-2, 1,                 # dx, n_ghost
        0.9, 0.01, 10,         # cfl, t_max, out_interval
        [(0.0, 1.1), (0.0, 1.2e5), (200.0, 450.0), (-0.5, 10.0)],  # ylims (rho, p, T, u)
    )
    solve(cfg; filename="step2.mp4")
end

main()
