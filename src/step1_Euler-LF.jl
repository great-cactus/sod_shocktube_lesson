"""
Sod shock tube: numerical solver using Lax-Friedrichs scheme.

1D Euler equations を Lax-Friedrichs flux + Forward Euler で解く
"""

include(joinpath(@__DIR__, "common.jl"))

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
function lax_friedrichs_flux(U_L::Vec3, U_R::Vec3,
                             gamma::Float64)::Vec3
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    F_L = compute_flux(U_L, gamma)
    F_R = compute_flux(U_R, gamma)

    a_L = sound_speed(W_L[3], W_L[1], gamma)
    a_R = sound_speed(W_R[3], W_R[1], gamma)
    alpha = max(abs(W_L[2]) + a_L, abs(W_R[2]) + a_R)

    return 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)
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

    # 保存量の履歴を収集
    cons_history = Tuple{Float64, Vector{Vec3}}[]
    Q0 = compute_total_conserved(U_arr, cfg.dx, i_start, i_end)
    push!(cons_history, (0.0, [Q0]))

    U_buf = copy(U_arr)
    t = 0.0
    step = 0

    while t < cfg.t_max
        dt = compute_dt(U_arr, cfg.dx, cfg.cfl, cfg.gamma)

        # 内部セルの更新
        copyto!(U_buf, U_arr)
        @inbounds for i in i_start:i_end
            F_left = lax_friedrichs_flux(U_arr[i-1], U_arr[i], cfg.gamma)
            F_right = lax_friedrichs_flux(U_arr[i], U_arr[i+1], cfg.gamma)
            L = -(F_right - F_left) / cfg.dx
            U_buf[i] = U_arr[i] + dt * L
        end

        # 境界条件 (ゼロ勾配)
        apply_bc!(U_buf, cfg, i_start, i_end)

        U_arr, U_buf = U_buf, U_arr
        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, copy(U_arr)))
            Q = compute_total_conserved(U_arr, cfg.dx, i_start, i_end)
            push!(cons_history, (t, [Q]))
        end
    end

    # 動画を記録
    println("Recording $(length(frames)) frames to $filename ...")
    record(fig, filename, frames; framerate=fps) do (t_frame, U_frame)
        update_observables!(obs_num, obs_exact, title_obs, x, U_frame, t_frame, cfg)
    end
    println("Done: $filename")

    # 保存量の時間変化動画
    cons_filename = replace(filename, ".mp4" => "_conservation.mp4")
    record_conservation(cons_filename, cons_history, cfg;
                         solver_names=["Lax-Friedrichs"],
                         solver_colors=[:black], fps=fps)
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
    solve(cfg; filename="step1.mp4")
end

main()
