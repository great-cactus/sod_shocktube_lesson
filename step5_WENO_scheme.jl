"""
1D Euler equations を MUSCL + HLL + SSPRK2 で解く
"""

include("common.jl")
include("riemann_solvers.jl")
include("runge-kutta.jl")
using .RungeKutta

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
    rho_R = W_plus[1] - 0.25 * ( (1+kappa)*D_rho_i + (1-kappa)*D_rho_plus )

    # u
    D_u_minus = W_i[2] - W_minus[2]
    D_u_i     = W_plus[2] - W_i[2]
    D_u_plus  = W_plus2[2] - W_plus[2]

    u_L = W_i[2] + 0.25 * ( (1-kappa)*D_u_minus + (1+kappa)*D_u_i)
    u_R = W_plus[2] - 0.25 * ( (1+kappa)*D_u_i + (1-kappa)*D_u_plus )

    # p
    D_p_minus = W_i[3] - W_minus[3]
    D_p_i     = W_plus[3] - W_i[3]
    D_p_plus  = W_plus2[3] - W_plus[3]

    p_L = W_i[3] + 0.25 * ( (1-kappa)*D_p_minus + (1+kappa)*D_p_i)
    p_R = W_plus[3] - 0.25 * ( (1+kappa)*D_p_i + (1-kappa)*D_p_plus )

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
# リミタ関数 (ratio 形式: 勾配比 r を受け取りスカラー φ(r) を返す)
# ---------------------------------------------------------------------------

"""minmod(r): 最も拡散的なリミタ."""
function limiter_minmod(r::Float64)::Float64
    return max(0.0, min(r, 1.0))
end

"""van Leer(r): バランス型リミタ."""
function limiter_van_leer(r::Float64)::Float64
    return (r + abs(r)) / (1.0 + abs(r))
end

"""superbee(r): 最も圧縮的なリミタ."""
function limiter_superbee(r::Float64)::Float64
    return max(0.0, min(2.0 * r, 1.0), min(r, 2.0))
end

"""
単一リミタ MUSCL 再構成.

セルごとに勾配比 r から φ(r) を求め, 制限された勾配で界面値を再構成する.
W_L = W_i    + 0.5 · φ(r_i)    · Δ_{i-1/2}
W_R = W_{i+1} - 0.5 · φ(r_{i+1}) · Δ_{i+3/2}
"""
function MUSCL_limited(cfg::Config, limiter::Function;
                       U_minus::Vec3, U_i::Vec3, U_plus::Vec3, U_plus2::Vec3,
                       kappa::Float64 = 0.0)
    W_minus = conservative_to_primitive(U_minus, cfg.gamma)
    W_i     = conservative_to_primitive(U_i, cfg.gamma)
    W_plus  = conservative_to_primitive(U_plus, cfg.gamma)
    W_plus2 = conservative_to_primitive(U_plus2, cfg.gamma)

    wL = MVector{3, Float64}(0.0, 0.0, 0.0)
    wR = MVector{3, Float64}(0.0, 0.0, 0.0)
    for k in 1:3
        D_minus = W_i[k] - W_minus[k]       # Δ_{i-1/2}
        D_i     = W_plus[k] - W_i[k]        # Δ_{i+1/2}
        D_plus  = W_plus2[k] - W_plus[k]    # Δ_{i+3/2}

        # 左状態: セル i から右界面へ外挿
        r_L = (abs(D_i) > 1.0e-30) ? D_minus / D_i : 0.0
        wL[k] = W_i[k] + 0.5 * limiter(r_L) * D_i

        # 右状態: セル i+1 から左界面へ外挿
        r_R = (abs(D_i) > 1.0e-30) ? D_plus / D_i : 0.0
        wR[k] = W_plus[k] - 0.5 * limiter(r_R) * D_i
    end

    W_L = Vec3(wL[1], wL[2], wL[3])
    W_R = Vec3(wR[1], wR[2], wR[3])
    U_L = primitive_to_conservative(W_L, cfg.gamma)
    U_R = primitive_to_conservative(W_R, cfg.gamma)

    return U_L, U_R
end

MUSCL_minmod(cfg::Config; kwargs...)   = MUSCL_limited(cfg, limiter_minmod; kwargs...)
MUSCL_van_leer(cfg::Config; kwargs...) = MUSCL_limited(cfg, limiter_van_leer; kwargs...)
MUSCL_superbee(cfg::Config; kwargs...) = MUSCL_limited(cfg, limiter_superbee; kwargs...)

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
# 空間離散化 (MUSCL + HLL フラックス → RHS)
# ---------------------------------------------------------------------------

"""
MUSCL 再構成 + HLL フラックスから空間 RHS L(U) を計算する.
"""
function compute_rhs(U_arr::Vector{Vec3}, muscl_func::Function,
                     cfg::Config, i_start::Int, i_end::Int)
    L = fill(Vec3(0.0, 0.0, 0.0), length(U_arr))
    @inbounds for i in i_start:i_end
        # 界面 i-1/2
        U_L_left, U_R_left = muscl_func(cfg;
            U_minus=U_arr[i-2], U_i=U_arr[i-1], U_plus=U_arr[i], U_plus2=U_arr[i+1])
        U_L_left, U_R_left = safe_reconstruct(U_L_left, U_R_left, U_arr[i-1], U_arr[i], cfg.gamma)
        F_left = HLL(U_L_left, U_R_left, cfg.gamma)

        # 界面 i+1/2
        U_L_right, U_R_right = muscl_func(cfg;
            U_minus=U_arr[i-1], U_i=U_arr[i], U_plus=U_arr[i+1], U_plus2=U_arr[i+2])
        U_L_right, U_R_right = safe_reconstruct(U_L_right, U_R_right, U_arr[i], U_arr[i+1], cfg.gamma)
        F_right = HLL(U_L_right, U_R_right, cfg.gamma)

        L[i] = -(F_right - F_left) / cfg.dx
    end
    return L
end

"""
RK中間段階で発生しうる非物理的な状態 (負密度・負圧力) をクランプする.
"""
function clamp_nonphysical!(U::Vector{Vec3}, gamma::Float64)
    EPS_RHO = 1.0e-10
    EPS_P   = 1.0e-10
    @inbounds for i in eachindex(U)
        rho = U[i][1]
        u   = U[i][2] / U[i][1]
        p   = (gamma - 1.0) * (U[i][3] - 0.5 * U[i][2]^2 / U[i][1])
        if rho <= 0.0 || p <= 0.0
            rho_c = max(rho, EPS_RHO)
            p_c   = max(p, EPS_P)
            U[i] = primitive_to_conservative(Vec3(rho_c, u, p_c), gamma)
        end
    end
end

"""
RungeKutta.rk_step に渡すための RHS 関数を生成する.
境界条件を適用してから compute_rhs を呼ぶクロージャを返す.
"""
function make_rhs_func(muscl_func::Function, cfg::Config, i_start::Int, i_end::Int)
    return function(_t, U_arr)
        U_bc = copy(U_arr)
        clamp_nonphysical!(U_bc, cfg.gamma)
        apply_bc!(U_bc, cfg, i_start, i_end)
        return compute_rhs(U_bc, muscl_func, cfg, i_start, i_end)
    end
end

# ---------------------------------------------------------------------------
# メインループ
# ---------------------------------------------------------------------------

"""
4種の MUSCL (no limiter, minmod, van Leer, superbee) + HLL + SSPRK2 を
同時に時間発展させ, 比較動画を出力する.

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

    fig, obs_solvers, obs_exact, title_obs = create_figure_4solvers(x, cfg)

    i_start = cfg.n_ghost + 1
    i_end = length(x) - cfg.n_ghost

    tab = RungeKutta.SSPRK2()
    rhs_funcs = [make_rhs_func(mf, cfg, i_start, i_end) for mf in muscl_funcs]

    frames = Tuple{Float64, Vector{Vector{Vec3}}}[]
    push!(frames, (0.0, [copy(u) for u in U_arr]))

    # 保存量の履歴を収集
    cons_history = Tuple{Float64, Vector{Vec3}}[]
    Qs0 = [compute_total_conserved(U_arr[s], cfg.dx, i_start, i_end) for s in 1:n_solvers]
    push!(cons_history, (0.0, Qs0))

    t = 0.0
    step = 0

    while t < cfg.t_max
        dt = minimum(compute_dt(U_arr[s], cfg.dx, cfg.cfl, cfg.gamma) for s in 1:n_solvers)

        for s in 1:n_solvers
            U_arr[s] = RungeKutta.rk_step(rhs_funcs[s], t, U_arr[s], dt, tab)
            apply_bc!(U_arr[s], cfg, i_start, i_end)

            # 更新後に非物理的なセルがあれば前ステップの値に戻す
            for i in eachindex(U_arr[s])
                W = conservative_to_primitive(U_arr[s][i], cfg.gamma)
                if W[1] <= 0.0 || W[3] <= 0.0
                    U_arr[s][i] = U0[i]
                end
            end
        end

        t += dt
        step += 1
        @printf("step=%6d, t=%.6e\n", step, t)

        if step % cfg.out_interval == 0
            push!(frames, (t, [copy(U_arr[s]) for s in 1:n_solvers]))
            Qs = [compute_total_conserved(U_arr[s], cfg.dx, i_start, i_end) for s in 1:n_solvers]
            push!(cons_history, (t, Qs))
        end
    end

    # 動画を記録
    println("Recording $(length(frames)) frames to $filename ...")
    record(fig, filename, frames; framerate=fps) do (t_frame, U_list)
        update_observables_4solvers!(obs_solvers, obs_exact, title_obs, x, U_list, t_frame, cfg)
    end
    println("Done: $filename")

    # 保存量の時間変化動画
    cons_filename = replace(filename, ".mp4" => "_conservation.mp4")
    record_conservation(cons_filename, cons_history, cfg;
                         solver_names=["No limiter", "Minmod", "Van Leer", "Superbee"],
                         solver_colors=[:gray, :blue, :green, :orange], fps=fps)
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

    solve(cfg; filename="step4.mp4")
end

main()
