"""
1D Euler equations を HLL + Runge-Kutta 時間積分で解く.

RungeKutta モジュールの Butcher tableau ベース汎用ソルバを利用し,
SSPRK3 と RK4 を同時に時間発展させ, 比較動画を出力する.
"""

include("common.jl")
include("riemann_solvers.jl")
include("runge-kutta.jl")
using .RungeKutta

# ---------------------------------------------------------------------------
# 空間離散化 (HLL フラックス → RHS)
# ---------------------------------------------------------------------------

"""
HLL フラックスから空間 RHS L(U) を計算する.
ゴーストセルの L は 0 のまま.
"""
function compute_rhs(U_arr::Vector{Vec3}, cfg::Config, i_start::Int, i_end::Int)
    L = fill(Vec3(0.0, 0.0, 0.0), length(U_arr))
    @inbounds for i in i_start:i_end
        F_left  = HLL(U_arr[i-1], U_arr[i],   cfg.gamma)
        F_right = HLL(U_arr[i],   U_arr[i+1], cfg.gamma)
        L[i] = -(F_right - F_left) / cfg.dx
    end
    return L
end

"""
RungeKutta.rk_step に渡すための RHS 関数を生成する.
境界条件を適用してから compute_rhs を呼ぶクロージャを返す.
"""
function make_rhs_func(cfg::Config, i_start::Int, i_end::Int)
    return function(_t, U_arr)
        U_bc = copy(U_arr)
        apply_bc!(U_bc, cfg, i_start, i_end)
        return compute_rhs(U_bc, cfg, i_start, i_end)
    end
end

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

"""
SSPRK3 と RK4 を同時に時間発展させ, 比較動画を出力する.

RungeKutta モジュールの汎用 rk_step を使用.

# Args
- `cfg`:      計算条件.
- `filename`: 出力ファイル名 (デフォルト "step3.mp4").
- `fps`:      フレームレート (デフォルト 30).
"""
function solve(cfg::Config; filename::String="step3.mp4", fps::Int=30)
    x, U_ssprk3 = create_initial_condition(cfg)
    _, U_rk4    = create_initial_condition(cfg)

    fig, obs_ssprk3, obs_rk4, obs_exact, title_obs = create_figure_compare(
        x, cfg; label1="SSPRK3", label2="RK4", color1=:blue, color2=:green)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    tab_ssprk3 = RungeKutta.SSPRK3()
    tab_rk4    = RungeKutta.RK4()
    rhs_func   = make_rhs_func(cfg, i_start, i_end)

    # フレームデータを収集
    frames = Tuple{Float64, Vector{Vec3}, Vector{Vec3}}[]
    push!(frames, (0.0, copy(U_ssprk3), copy(U_rk4)))

    t = 0.0
    step = 0

    while t < cfg.t_max
        dt_ssprk3 = compute_dt(U_ssprk3, cfg.dx, cfg.cfl, cfg.gamma)
        dt_rk4    = compute_dt(U_rk4,    cfg.dx, cfg.cfl, cfg.gamma)
        dt = min(dt_ssprk3, dt_rk4)

        U_ssprk3 = RungeKutta.rk_step(rhs_func, t, U_ssprk3, dt, tab_ssprk3)
        apply_bc!(U_ssprk3, cfg, i_start, i_end)

        U_rk4 = RungeKutta.rk_step(rhs_func, t, U_rk4, dt, tab_rk4)
        apply_bc!(U_rk4, cfg, i_start, i_end)

        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, copy(U_ssprk3), copy(U_rk4)))
        end
    end

    # 動画を記録
    println("Recording $(length(frames)) frames to $filename ...")
    record(fig, filename, frames; framerate=fps) do (t_frame, U_ssprk3_frame, U_rk4_frame)
        update_observables_compare!(obs_ssprk3, obs_rk4, obs_exact, title_obs,
                                    x, U_ssprk3_frame, U_rk4_frame, t_frame, cfg)
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
        1e-1, 1,                 # dx, n_ghost
        0.5, 0.008, 1,         # cfl, t_max, out_interval
        [(0.0, 1.1), (0.0, 1.2e5), (200.0, 450.0), (-0.5, 330.0)],  # ylims (rho, p, T, u)
    )

    solve(cfg; filename="step3.mp4")
end

main()
