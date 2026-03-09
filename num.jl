"""
Sod shock tube: numerical solver using Lax-Friedrichs scheme (Julia版).

1D Euler equations を Lax-Friedrichs flux + Forward Euler で解く
最小限の実装. 圧縮性流体コードの学習用.
"""

include("exact_solution.jl")
using .ExactSolution
using Printf
using CairoMakie

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
    # 気体物性
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
end

"""理想気体の密度 rho = p * MW / (R * T)."""
function get_rho_L(cfg::Config)::Float64
    return cfg.p_L * cfg.MW / (R_UNIVERSAL * cfg.T_L)
end

"""理想気体の密度 rho = p * MW / (R * T)."""
function get_rho_R(cfg::Config)::Float64
    return cfg.p_R * cfg.MW / (R_UNIVERSAL * cfg.T_R)
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
function conservative_to_primitive(U::Vector{Float64}, gamma::Float64)::Vector{Float64}
    rho = U[1]
    u = U[2] / U[1]
    p = (gamma - 1.0) * (U[3] - 0.5 * U[2]^2 / U[1])
    return [rho, u, p]
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
function primitive_to_conservative(W::Vector{Float64}, gamma::Float64)::Vector{Float64}
    rho, u, p = W[1], W[2], W[3]
    rho_u = rho * u
    rho_E = p / (gamma - 1.0) + 0.5 * rho * u^2
    return [rho, rho_u, rho_E]
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
function compute_flux(U::Vector{Float64}, gamma::Float64)::Vector{Float64}
    p = (gamma - 1.0) * (U[3] - 0.5 * U[2]^2 / U[1])
    F1 = U[2]                          # rho * u
    F2 = U[2]^2 / U[1] + p            # rho * u^2 + p
    F3 = U[2] / U[1] * (U[3] + p)     # u * (rho*E + p)
    return [F1, F2, F3]
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
# Lax-Friedrichs flux
# ---------------------------------------------------------------------------

"""
セル界面での Lax-Friedrichs 数値フラックスを計算する.

F_LF = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)
alpha = max(|u_L| + a_L, |u_R| + a_R)

# Args
- `U_L`:   界面左側の保存変数ベクトル (長さ3).
- `U_R`:   界面右側の保存変数ベクトル (長さ3).
- `gamma`: 比熱比.

# Returns
数値フラックスベクトル (長さ3).
"""
function lax_friedrichs_flux(U_L::Vector{Float64}, U_R::Vector{Float64},
                             gamma::Float64)::Vector{Float64}
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)

    a_L = sound_speed(W_L[3], W_L[1], gamma)
    a_R = sound_speed(W_R[3], W_R[1], gamma)
    alpha = max(abs(W_L[2]) + a_L, abs(W_R[2]) + a_R)

    return 0.5 .* (F_L .+ F_R) .- 0.5 * alpha .* (U_R .- U_L)
end

# ---------------------------------------------------------------------------
# 時間刻み幅
# ---------------------------------------------------------------------------

"""
CFL 条件を満たす時間刻み幅を計算する.

全セルの最大波速 (|u| + a) を求め, dt = cfl * dx / max_speed とする.

# Args
- `U_arr`: 全セルの保存変数配列 (n_points × 3 の Vector{Vector}).
- `dx`:    セル幅 [m].
- `cfl`:   CFL 数 (0 < cfl <= 1).
- `gamma`: 比熱比.

# Returns
時間刻み幅 dt [s].
"""
function compute_dt(U_arr::Vector{Vector{Float64}}, dx::Float64,
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

    rho_L = get_rho_L(cfg)
    rho_R = get_rho_R(cfg)

    # 座標 (ゴーストセル含む)
    x = Vector{Float64}(undef, n_points)
    for i in 1:n_points
        x[i] = cfg.x_left - cfg.n_ghost * cfg.dx + cfg.dx * (i - 1)
    end

    # 基本変数 → 保存変数
    U_arr = Vector{Vector{Float64}}(undef, n_points)
    for i in 1:n_points
        if x[i] < cfg.x_center
            W = [rho_L, cfg.u_L, cfg.p_L]
        else
            W = [rho_R, cfg.u_R, cfg.p_R]
        end
        U_arr[i] = primitive_to_conservative(W, cfg.gamma)
    end

    return x, U_arr
end

# ---------------------------------------------------------------------------
# プロット
# ---------------------------------------------------------------------------

"""
理想気体の状態方程式から温度を計算する.

T = p * MW / (rho * R)

# Args
- `p`:   圧力 [Pa].
- `rho`: 密度 [kg/m^3].
- `MW`:  分子量 [kg/mol].

# Returns
温度 [K].
"""
function get_temperature(p::Float64, rho::Float64, MW::Float64)::Float64
    return p * MW / (rho * R_UNIVERSAL)
end

"""
Config から ExactSolution.ShockTubeConfig を生成する.

# Args
- `cfg`: 数値計算用の Config.

# Returns
解析解ソルバ用の ShockTubeConfig.
"""
function make_exact_config(cfg::Config)::ExactSolution.ShockTubeConfig
    return ExactSolution.ShockTubeConfig(
        cfg.p_L, cfg.T_L, cfg.u_L,
        cfg.p_R, cfg.T_R, cfg.u_R,
        cfg.MW, cfg.gamma, cfg.x_center,
    )
end

"""
保存変数配列から基本変数 + 温度の配列を抽出する.

# Args
- `U_arr`: 保存変数の配列.
- `cfg`:   計算条件.

# Returns
`(rho, p, T, u)` の各 Vector{Float64}.
"""
function extract_primitives(U_arr::Vector{Vector{Float64}}, cfg::Config)
    n = length(U_arr)
    rho_arr = Vector{Float64}(undef, n)
    p_arr = Vector{Float64}(undef, n)
    u_arr = Vector{Float64}(undef, n)
    T_arr = Vector{Float64}(undef, n)

    for i in 1:n
        W = conservative_to_primitive(U_arr[i], cfg.gamma)
        rho_arr[i] = W[1]
        u_arr[i] = W[2]
        p_arr[i] = W[3]
        T_arr[i] = get_temperature(W[3], W[1], cfg.MW)
    end

    return rho_arr, p_arr, T_arr, u_arr
end

"""
4 パネルの Figure と Observable を作成する.

数値解 (黒実線) と解析解 (赤破線) を重ねて描画するレイアウトを構築.
Observable を更新するとアニメーションフレームが変化する.

# Args
- `x`:   座標配列.
- `cfg`: 計算条件.

# Returns
`(fig, obs_num, obs_exact, title_obs)` — Figure, 数値解 Observable (4つ),
解析解 Observable (4つ), タイトル Observable.
"""
function create_figure(x::Vector{Float64}, cfg::Config)
    labels = [L"\rho~\mathrm{[kg/m^3]}", L"p~\mathrm{[Pa]}",
              L"T~\mathrm{[K]}", L"u~\mathrm{[m/s]}"]

    # 初期値で Observable を作成
    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, 0.0)
    rho0, p0, T0, u0 = extract_primitives(
        [primitive_to_conservative([exact_fields.rho[i], exact_fields.u[i], exact_fields.p[i]], cfg.gamma)
         for i in eachindex(x)], cfg)

    obs_num = [Observable(copy(d)) for d in [rho0, p0, T0, u0]]
    obs_exact = [Observable(copy(d)) for d in [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u]]
    title_obs = Observable("t = 0.0000e+00 s")

    fig = Figure(size=(800, 600))

    for (idx, label) in enumerate(labels)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = Axis(fig[row, col], xlabel="x [m]", ylabel=label)
        lines!(ax, x, obs_num[idx], color=:black, label="Numerical")
        lines!(ax, x, obs_exact[idx], color=:red, linestyle=:dash, label="Exact")
        axislegend(ax, position=:lt, labelsize=10)
    end

    Label(fig[0, :], title_obs, fontsize=14)

    return fig, obs_num, obs_exact, title_obs
end

"""
Observable を現在の U_arr と時刻 t で更新する.

# Args
- `obs_num`:   数値解の Observable 配列 (4つ).
- `obs_exact`: 解析解の Observable 配列 (4つ).
- `title_obs`: タイトルの Observable.
- `x`:         座標配列.
- `U_arr`:     保存変数の配列.
- `t`:         現在時刻 [s].
- `cfg`:       計算条件.
"""
function update_observables!(obs_num, obs_exact, title_obs,
                             x::Vector{Float64}, U_arr::Vector{Vector{Float64}},
                             t::Float64, cfg::Config)
    rho_arr, p_arr, T_arr, u_arr = extract_primitives(U_arr, cfg)

    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, t)

    for (obs, data) in zip(obs_num, [rho_arr, p_arr, T_arr, u_arr])
        obs[] = data
    end
    for (obs, data) in zip(obs_exact, [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u])
        obs[] = data
    end
    title_obs[] = @sprintf("t = %.4e s", t)
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
- `filename`: 出力ファイル名 (デフォルト "shock_tube.mp4").
- `fps`:      フレームレート (デフォルト 30).
"""
function solve(cfg::Config; filename::String="shock_tube.mp4", fps::Int=30)
    x, U_arr = create_initial_condition(cfg)

    fig, obs_num, obs_exact, title_obs = create_figure(x, cfg)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    # フレームデータを収集 (時刻, U_arr のスナップショット)
    frames = Tuple{Float64, Vector{Vector{Float64}}}[]
    push!(frames, (0.0, deepcopy(U_arr)))

    t = 0.0
    step = 0

    while t < cfg.t_max
        dt = compute_dt(U_arr, cfg.dx, cfg.cfl, cfg.gamma)

        # 内部セルの更新
        U_new = deepcopy(U_arr)
        for i in i_start:i_end
            F_left = lax_friedrichs_flux(U_arr[i-1], U_arr[i], cfg.gamma)
            F_right = lax_friedrichs_flux(U_arr[i], U_arr[i+1], cfg.gamma)
            L = -(F_right .- F_left) ./ cfg.dx
            U_new[i] = U_arr[i] .+ dt .* L
        end

        # 境界条件 (ゼロ勾配)
        for g in 1:cfg.n_ghost
            U_new[g] = copy(U_new[i_start])
            U_new[end - g + 1] = copy(U_new[i_end])
        end

        U_arr = U_new
        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, deepcopy(U_arr)))
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
        1e-3, 1,                 # dx, n_ghost
        0.9, 0.01, 100,         # cfl, t_max, out_interval
    )
    solve(cfg)
end

main()
