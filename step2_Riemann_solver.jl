"""
1D Euler equations を Riemann solver + Forward Euler で解く
"""

include("common.jl")
include("riemann_solvers.jl")

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

"""
HLL と HLLC を同時に時間発展させ, 比較動画を出力する.

1. 初期条件を生成 (両ソルバで共有)
2. 各タイムステップで:
   - CFL 条件から dt を決定 (両ソルバの最小値)
   - HLL / HLLC それぞれの数値フラックスを計算
   - Forward Euler で保存変数を更新
   - ゼロ勾配境界条件を適用
3. out_interval ステップごとに動画フレームを記録

# Args
- `cfg`:      計算条件.
- `filename`: 出力ファイル名 (デフォルト "movie.mp4").
- `fps`:      フレームレート (デフォルト 30).
"""
function solve(cfg::Config; filename::String="movie.mp4", fps::Int=30)
    x, U_hll = create_initial_condition(cfg)
    _, U_hllc = create_initial_condition(cfg)

    fig, obs_hll, obs_hllc, obs_exact, title_obs = create_figure_compare(x, cfg)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    # フレームデータを収集 (時刻, HLL の U, HLLC の U)
    frames = Tuple{Float64, Vector{Vec3}, Vector{Vec3}}[]
    push!(frames, (0.0, copy(U_hll), copy(U_hllc)))

    buf_hll  = copy(U_hll)
    buf_hllc = copy(U_hllc)
    t = 0.0
    step = 0

    while t < cfg.t_max
        dt_hll  = compute_dt(U_hll,  cfg.dx, cfg.cfl, cfg.gamma)
        dt_hllc = compute_dt(U_hllc, cfg.dx, cfg.cfl, cfg.gamma)
        dt = min(dt_hll, dt_hllc)

        # HLL の更新
        copyto!(buf_hll, U_hll)
        for i in i_start:i_end
            F_left  = HLL(U_hll[i-1], U_hll[i],   cfg.gamma)
            F_right = HLL(U_hll[i],   U_hll[i+1], cfg.gamma)
            L = -(F_right - F_left) / cfg.dx
            buf_hll[i] = U_hll[i] + dt * L
        end

        # HLLC の更新
        copyto!(buf_hllc, U_hllc)
        for i in i_start:i_end
            F_left  = HLLC(U_hllc[i-1], U_hllc[i],   cfg.gamma)
            F_right = HLLC(U_hllc[i],   U_hllc[i+1], cfg.gamma)
            L = -(F_right - F_left) / cfg.dx
            buf_hllc[i] = U_hllc[i] + dt * L
        end

        # 境界条件 (ゼロ勾配)
        apply_bc!(buf_hll,  cfg, i_start, i_end)
        apply_bc!(buf_hllc, cfg, i_start, i_end)

        U_hll,  buf_hll  = buf_hll,  U_hll
        U_hllc, buf_hllc = buf_hllc, U_hllc
        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, copy(U_hll), copy(U_hllc)))
        end
    end

    # 動画を記録
    println("Recording $(length(frames)) frames to $filename ...")
    record(fig, filename, frames; framerate=fps) do (t_frame, U_hll_frame, U_hllc_frame)
        update_observables_compare!(obs_hll, obs_hllc, obs_exact, title_obs,
                                    x, U_hll_frame, U_hllc_frame, t_frame, cfg)
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

    solve(cfg; filename="step2.mp4")
end

main()
