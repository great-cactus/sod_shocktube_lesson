"""
1D Euler equations を Riemann solver + Forward Euler で解く
"""

include("common.jl")
include("riemann_solvers.jl")

# ---------------------------------------------------------------------------
# MUSCLスキーム
# ---------------------------------------------------------------------------

function MUSCL(cfg::Config; U_minus::Vec3, U_i::Vec3, U_plus::Vec3, U_plus2:: Vec3, kappa::Float64 = 1/3)
    W_minus = conservative_to_primitive(U_minus, cfg.gamma)
    W_i     = conservative_to_primitive(U_i, cfg.gamma)
    W_plus  = conservative_to_primitive(U_plus, cfg.gamma)
    W_plus2 = conservative_to_primitive(U_plus2, cfg.gamma)

    # rho
    D_rho_minus = W_i[1] - W_minus[1]
    D_rho_i     = W_plus[1] - W_i[1]
    D_rho_plus  = W_plus2[1] - W_plus[1]

    rho_L = W_i[1] + 0.25 * ( (1-kappa)*D_rho_minus + (1+kappa)*D_rho_i)
    rho_R = W_plus[1] + 0.25 * ( (1+kappa)*D_rho_i + (1-kappa)*D_rho_plus )

    # u
    D_u_minus = W_i[2] - W_minus[2]
    D_u_i     = W_plus[2] - W_i[2]
    D_u_plus  = W_plus2[2] - W_plus[2]

    u_L = W_i[2] + 0.25 * ( (1-kappa)*D_u_minus + (1+kappa)*D_u_i)
    u_R = W_plus[2] + 0.25 * ( (1+kappa)*D_u_i + (1-kappa)*D_u_plus )

    # p
    D_p_minus = W_i[3] - W_minus[3]
    D_p_i     = W_plus[3] - W_i[3]
    D_p_plus  = W_plus2[3] - W_plus[3]

    p_L = W_i[3] + 0.25 * ( (1-kappa)*D_p_minus + (1+kappa)*D_p_i)
    p_R = W_plus[3] + 0.25 * ( (1+kappa)*D_p_i + (1-kappa)*D_p_plus )

    # 負密度・負圧力が出たら1次精度にフォールバック
    if rho_L <= 0.0 || p_L <= 0.0
        rho_L, u_L, p_L = W_i[1], W_i[2], W_i[3]
    end
    if rho_R <= 0.0 || p_R <= 0.0
        rho_R, u_R, p_R = W_plus[1], W_plus[2], W_plus[3]
    end

    W_L = Vec3(rho_L, u_L, p_L)
    W_R = Vec3(rho_R, u_R, p_R)
    U_L = primitive_to_conservative(W_L, cfg.gamma)
    U_R = primitive_to_conservative(W_R, cfg.gamma)

    return U_L, U_R
end

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

"""
MUSCL + HLL で shock tube を時間発展させ, MP4 動画を出力する.

1. 初期条件を生成
2. 各タイムステップで:
   - CFL 条件から dt を決定
   - MUSCL 再構成で界面左右状態を求め, HLL フラックスを計算
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

    fig, obs_num, obs_exact, title_obs = create_figure(x, cfg)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    frames = Tuple{Float64, Vector{Vec3}}[]
    push!(frames, (0.0, copy(U_hll)))

    U_buf = copy(U_hll)
    t = 0.0
    step = 0

    while t < cfg.t_max
        dt = compute_dt(U_hll, cfg.dx, cfg.cfl, cfg.gamma)

        copyto!(U_buf, U_hll)
        for i in i_start:i_end
            # 界面 i-1/2
            U_L_left, U_R_left = MUSCL(cfg;
                U_minus=U_hll[i-2], U_i=U_hll[i-1], U_plus=U_hll[i], U_plus2=U_hll[i+1])
            F_left = HLL(U_L_left, U_R_left, cfg.gamma)

            # 界面 i+1/2
            U_L_right, U_R_right = MUSCL(cfg;
                U_minus=U_hll[i-1], U_i=U_hll[i], U_plus=U_hll[i+1], U_plus2=U_hll[i+2])
            F_right = HLL(U_L_right, U_R_right, cfg.gamma)

            L = -(F_right - F_left) / cfg.dx
            U_buf[i] = U_hll[i] + dt * L
        end

        # 境界条件 (ゼロ勾配)
        for g in 1:cfg.n_ghost
            U_buf[g]            = U_buf[i_start]
            U_buf[end - g + 1]  = U_buf[i_end]
        end

        U_hll, U_buf = U_buf, U_hll
        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, copy(U_hll)))
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
        1e-1, 2,                 # dx, n_ghost
        0.5, 0.008, 1,         # cfl, t_max, out_interval
        [(0.0, 1.1), (0.0, 1.2e5), (200.0, 450.0), (-0.5, 330.0)],  # ylims (rho, p, T, u)
    )

    solve(cfg; filename="step3.mp4")
end

main()
