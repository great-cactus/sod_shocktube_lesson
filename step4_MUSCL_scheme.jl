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
# リミッタ関数 (すべて double 形式: x, y の2引数)
# ---------------------------------------------------------------------------

"""minmod(x, y): 同符号なら絶対値の小さい方, 異符号なら 0."""
function minmod(x::Float64, y::Float64)::Float64
    if x * y <= 0.0
        return 0.0
    else
        return sign(x) * min(abs(x), abs(y))
    end
end

"""van Leer(x, y): 同符号なら調和平均, 異符号なら 0."""
function van_leer(x::Float64, y::Float64)::Float64
    if x * y <= 0.0
        return 0.0
    else
        return 2.0 * x * y / (x + y)
    end
end

"""superbee(x, y): minmod の max をとる."""
function superbee(x::Float64, y::Float64)::Float64
    if x * y <= 0.0
        return 0.0
    else
        s = sign(x)
        ax, ay = abs(x), abs(y)
        return s * max(min(2.0 * ax, ay), min(ax, 2.0 * ay))
    end
end

"""
MUSCL 再構成の共通ロジック. リミタ関数 `limiter(x,y)` を引数で受け取る.
"""
function MUSCL_limited(cfg::Config, limiter::Function;
                       U_minus::Vec3, U_i::Vec3, U_plus::Vec3, U_plus2::Vec3,
                       kappa::Float64 = 1.0/3.0)
    W_minus = conservative_to_primitive(U_minus, cfg.gamma)
    W_i     = conservative_to_primitive(U_i, cfg.gamma)
    W_plus  = conservative_to_primitive(U_plus, cfg.gamma)
    W_plus2 = conservative_to_primitive(U_plus2, cfg.gamma)

    beta = (3.0 - kappa) / (1.0 - kappa)

    wL = MVector{3, Float64}(0.0, 0.0, 0.0)
    wR = MVector{3, Float64}(0.0, 0.0, 0.0)
    for k in 1:3
        D_minus = W_i[k] - W_minus[k]
        D_i     = W_plus[k] - W_i[k]
        D_plus  = W_plus2[k] - W_plus[k]

        D_back_L = limiter(D_minus, beta * D_i)
        D_fwd_L  = limiter(D_i, beta * D_minus)
        D_back_R = limiter(D_i, beta * D_plus)
        D_fwd_R  = limiter(D_plus, beta * D_i)

        wL[k] = W_i[k]    + 0.25 * ((1.0 - kappa) * D_back_L + (1.0 + kappa) * D_fwd_L)
        wR[k] = W_plus[k] - 0.25 * ((1.0 + kappa) * D_back_R + (1.0 - kappa) * D_fwd_R)
    end

    W_L = Vec3(wL[1], wL[2], wL[3])
    W_R = Vec3(wR[1], wR[2], wR[3])
    U_L = primitive_to_conservative(W_L, cfg.gamma)
    U_R = primitive_to_conservative(W_R, cfg.gamma)

    return U_L, U_R
end

MUSCL_minmod(cfg::Config; kwargs...)   = MUSCL_limited(cfg, minmod; kwargs...)
MUSCL_van_leer(cfg::Config; kwargs...) = MUSCL_limited(cfg, van_leer; kwargs...)
MUSCL_superbee(cfg::Config; kwargs...) = MUSCL_limited(cfg, superbee; kwargs...)

"""
再構成結果が非物理的 (負密度・負圧力) なら1次精度 (セル平均値) にフォールバック.
"""
function safe_reconstruct(U_L::Vec3, U_R::Vec3, U_i::Vec3, U_ip1::Vec3, gamma::Float64)
    W_L = conservative_to_primitive(U_L, gamma)
    W_R = conservative_to_primitive(U_R, gamma)
    out_L = (W_L[1] <= 0.0 || W_L[3] <= 0.0) ? U_i   : U_L
    out_R = (W_R[1] <= 0.0 || W_R[3] <= 0.0) ? U_ip1 : U_R
    return out_L, out_R
end

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

"""
4種の MUSCL (no limiter, minmod, van Leer, superbee) + HLL を同時に
時間発展させ, 比較動画を出力する.

# Args
- `cfg`:      計算条件.
- `filename`: 出力ファイル名 (デフォルト "movie.mp4").
- `fps`:      フレームレート (デフォルト 30).
"""
function solve(cfg::Config; filename::String="movie.mp4", fps::Int=30)
    muscl_funcs = [MUSCL, MUSCL_minmod, MUSCL_van_leer, MUSCL_superbee]
    n_solvers = length(muscl_funcs)

    x, U0 = create_initial_condition(cfg)
    U_arr = [copy(U0) for _ in 1:n_solvers]
    U_buf = [copy(U0) for _ in 1:n_solvers]

    fig, obs_solvers, obs_exact, title_obs = create_figure_4solvers(x, cfg)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    frames = Tuple{Float64, Vector{Vector{Vec3}}}[]
    push!(frames, (0.0, [copy(u) for u in U_arr]))

    t = 0.0
    step = 0

    while t < cfg.t_max
        dt = minimum(compute_dt(U_arr[s], cfg.dx, cfg.cfl, cfg.gamma) for s in 1:n_solvers)

        for s in 1:n_solvers
            copyto!(U_buf[s], U_arr[s])
            for i in i_start:i_end
                # 界面 i-1/2
                U_L_left, U_R_left = muscl_funcs[s](cfg;
                    U_minus=U_arr[s][i-2], U_i=U_arr[s][i-1], U_plus=U_arr[s][i], U_plus2=U_arr[s][i+1])
                U_L_left, U_R_left = safe_reconstruct(U_L_left, U_R_left, U_arr[s][i-1], U_arr[s][i], cfg.gamma)
                F_left = HLL(U_L_left, U_R_left, cfg.gamma)

                # 界面 i+1/2
                U_L_right, U_R_right = muscl_funcs[s](cfg;
                    U_minus=U_arr[s][i-1], U_i=U_arr[s][i], U_plus=U_arr[s][i+1], U_plus2=U_arr[s][i+2])
                U_L_right, U_R_right = safe_reconstruct(U_L_right, U_R_right, U_arr[s][i], U_arr[s][i+1], cfg.gamma)
                F_right = HLL(U_L_right, U_R_right, cfg.gamma)

                L = -(F_right - F_left) / cfg.dx
                U_buf[s][i] = U_arr[s][i] + dt * L
            end

            # 境界条件 (ゼロ勾配)
            for g in 1:cfg.n_ghost
                U_buf[s][g]            = U_buf[s][i_start]
                U_buf[s][end - g + 1]  = U_buf[s][i_end]
            end
        end

        for s in 1:n_solvers
            U_arr[s], U_buf[s] = U_buf[s], U_arr[s]

            # 更新後に非物理的なセルがあれば前ステップの値に戻す
            for i in eachindex(U_arr[s])
                W = conservative_to_primitive(U_arr[s][i], cfg.gamma)
                if W[1] <= 0.0 || W[3] <= 0.0
                    U_arr[s][i] = U_buf[s][i]
                end
            end
        end
        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, [copy(U_arr[s]) for s in 1:n_solvers]))
        end
    end

    # 動画を記録
    println("Recording $(length(frames)) frames to $filename ...")
    record(fig, filename, frames; framerate=fps) do (t_frame, U_list)
        update_observables_4solvers!(obs_solvers, obs_exact, title_obs, x, U_list, t_frame, cfg)
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
