"""
1D Euler equations を WENO5 / WENO5-Z+ + HLL + SSPRK3 で解く
"""

include("common.jl")
include("riemann_solvers.jl")
include("runge-kutta.jl")
using .RungeKutta

# ---------------------------------------------------------------------------
# WENO5 再構成
# ---------------------------------------------------------------------------

"""
WENO5 左バイアス再構成.

セル i+1/2 の左状態 f_{i+1/2}^- をスティンシル {i-2, ..., i+2} から計算する.

呼び出し: WENO5_left(f[i-2], f[i-1], f[i], f[i+1], f[i+2])
"""
function WENO5_left(F_mm::Vec3, F_m::Vec3, F_i::Vec3, F_p::Vec3, F_pp::Vec3)::Vec3
    d_0, d_1, d_2 = 1/10, 6/10, 3/10
    SMALL = 1e-6

    results = MVector{3, Float64}(undef)
    for j in 1:3
        f_mm, f_m, f_i, f_p, f_pp = F_mm[j], F_m[j], F_i[j], F_p[j], F_pp[j]

        f_0 = (1/6) * ( 2*f_mm - 7*f_m + 11*f_i )
        f_1 = (1/6) * (   -f_m + 5*f_i  + 2*f_p )
        f_2 = (1/6) * ( 2*f_i + 5*f_p   -  f_pp )

        b_0 = (13/12)*(f_mm - 2*f_m + f_i)^2 + (1/4)*(f_mm - 4*f_m + 3*f_i)^2
        b_1 = (13/12)*(f_m  - 2*f_i + f_p)^2 + (1/4)*(f_m - f_p)^2
        b_2 = (13/12)*(f_i  - 2*f_p + f_pp)^2 + (1/4)*(3*f_i - 4*f_p + f_pp)^2

        a_0, a_1, a_2 = d_0/(b_0+SMALL)^2, d_1/(b_1+SMALL)^2, d_2/(b_2+SMALL)^2
        a_tot = a_0 + a_1 + a_2

        results[j] = (a_0*f_0 + a_1*f_1 + a_2*f_2) / a_tot
    end
    return Vec3(results[1], results[2], results[3])
end

"""
WENO5 右バイアス再構成.

セル i+1/2 の右状態 f_{i+1/2}^+ をスティンシル {i-1, ..., i+3} から計算する.

引数は右から左へ逆順に渡す:
呼び出し: WENO5_right(f[i+3], f[i+2], f[i+1], f[i], f[i-1])

逆順にすることでサブスティンシルの多項式は左バイアスと共通になり,
理想重み d_0=3/10, d_2=1/10 が右バイアスを実現する.
"""
function WENO5_right(F_mm::Vec3, F_m::Vec3, F_i::Vec3, F_p::Vec3, F_pp::Vec3)::Vec3
    d_0, d_1, d_2 = 3/10, 6/10, 1/10   # 左バイアスと d_0, d_2 を交換
    SMALL = 1e-6

    results = MVector{3, Float64}(undef)
    for j in 1:3
        f_mm, f_m, f_i, f_p, f_pp = F_mm[j], F_m[j], F_i[j], F_p[j], F_pp[j]

        f_0 = (1/6) * ( 2*f_mm - 7*f_m + 11*f_i )
        f_1 = (1/6) * (   -f_m + 5*f_i  + 2*f_p )
        f_2 = (1/6) * ( 2*f_i + 5*f_p   -  f_pp )

        b_0 = (13/12)*(f_mm - 2*f_m + f_i)^2 + (1/4)*(f_mm - 4*f_m + 3*f_i)^2
        b_1 = (13/12)*(f_m  - 2*f_i + f_p)^2 + (1/4)*(f_m - f_p)^2
        b_2 = (13/12)*(f_i  - 2*f_p + f_pp)^2 + (1/4)*(3*f_i - 4*f_p + f_pp)^2

        a_0, a_1, a_2 = d_0/(b_0+SMALL)^2, d_1/(b_1+SMALL)^2, d_2/(b_2+SMALL)^2
        a_tot = a_0 + a_1 + a_2

        results[j] = (a_0*f_0 + a_1*f_1 + a_2*f_2) / a_tot
    end
    return Vec3(results[1], results[2], results[3])
end

# ---------------------------------------------------------------------------
# WENO5-Z+ 再構成 (Borges et al. 2008)
#
# 通常の WENO5-JS の重み  a_k = d_k / (b_k + ε)^2  を改良し,
# グローバル平滑度指標 τ = |β_0 - β_2| を使って
#   a_k = d_k · (1 + (τ / (b_k + ε))^p)
# とすることで滑らかな領域での理想重みへの収束を加速する.
# ---------------------------------------------------------------------------

"""
WENO5-Z+ 左バイアス再構成.

呼び出し: WENO5zp_left(f[i-2], f[i-1], f[i], f[i+1], f[i+2])
"""
function WENO5zp_left(F_mm::Vec3, F_m::Vec3, F_i::Vec3, F_p::Vec3, F_pp::Vec3)::Vec3
    p = 2
    d_0, d_1, d_2 = 1/10, 6/10, 3/10
    SMALL = 1e-6

    results = MVector{3, Float64}(undef)
    for j in 1:3
        f_mm, f_m, f_i, f_p, f_pp = F_mm[j], F_m[j], F_i[j], F_p[j], F_pp[j]

        f_0 = (1/6) * ( 2*f_mm - 7*f_m + 11*f_i )
        f_1 = (1/6) * (   -f_m + 5*f_i  + 2*f_p )
        f_2 = (1/6) * ( 2*f_i + 5*f_p   -  f_pp )

        b_0 = (13/12)*(f_mm - 2*f_m + f_i)^2 + (1/4)*(f_mm - 4*f_m + 3*f_i)^2
        b_1 = (13/12)*(f_m  - 2*f_i + f_p)^2 + (1/4)*(f_m - f_p)^2
        b_2 = (13/12)*(f_i  - 2*f_p + f_pp)^2 + (1/4)*(3*f_i - 4*f_p + f_pp)^2

        tau = abs(b_0 - b_2)

        # WENO-Z 重み: a_k = d_k * (1 + (τ/(β_k+ε))^p)
        a_0 = d_0 * (1 + (tau / (b_0 + SMALL))^p)
        a_1 = d_1 * (1 + (tau / (b_1 + SMALL))^p)
        a_2 = d_2 * (1 + (tau / (b_2 + SMALL))^p)
        a_tot = a_0 + a_1 + a_2

        results[j] = (a_0*f_0 + a_1*f_1 + a_2*f_2) / a_tot
    end
    return Vec3(results[1], results[2], results[3])
end

"""
WENO5-Z+ 右バイアス再構成.

呼び出し: WENO5zp_right(f[i+3], f[i+2], f[i+1], f[i], f[i-1])
"""
function WENO5zp_right(F_mm::Vec3, F_m::Vec3, F_i::Vec3, F_p::Vec3, F_pp::Vec3)::Vec3
    p = 2
    d_0, d_1, d_2 = 3/10, 6/10, 1/10   # 左バイアスと d_0, d_2 を交換
    SMALL = 1e-6

    results = MVector{3, Float64}(undef)
    for j in 1:3
        f_mm, f_m, f_i, f_p, f_pp = F_mm[j], F_m[j], F_i[j], F_p[j], F_pp[j]

        f_0 = (1/6) * ( 2*f_mm - 7*f_m + 11*f_i )
        f_1 = (1/6) * (   -f_m + 5*f_i  + 2*f_p )
        f_2 = (1/6) * ( 2*f_i + 5*f_p   -  f_pp )

        b_0 = (13/12)*(f_mm - 2*f_m + f_i)^2 + (1/4)*(f_mm - 4*f_m + 3*f_i)^2
        b_1 = (13/12)*(f_m  - 2*f_i + f_p)^2 + (1/4)*(f_m - f_p)^2
        b_2 = (13/12)*(f_i  - 2*f_p + f_pp)^2 + (1/4)*(3*f_i - 4*f_p + f_pp)^2

        tau = abs(b_0 - b_2)

        # WENO-Z 重み: a_k = d_k * (1 + (τ/(β_k+ε))^p)
        a_0 = d_0 * (1 + (tau / (b_0 + SMALL))^p)
        a_1 = d_1 * (1 + (tau / (b_1 + SMALL))^p)
        a_2 = d_2 * (1 + (tau / (b_2 + SMALL))^p)
        a_tot = a_0 + a_1 + a_2

        results[j] = (a_0*f_0 + a_1*f_1 + a_2*f_2) / a_tot
    end
    return Vec3(results[1], results[2], results[3])
end

# ---------------------------------------------------------------------------
# WENO5-Z+M 再構成 (Adaptive-λ variant)
#
# WENO-Z+ の λ を適応的に決める改良版.
#
#   ξ_k  = τ / (β_k + ε)          # WENO-Z と同じ比率
#   η_k  = sqrt(1 / ξ_k)          # β_k/τ の平方根
#   z    = (1 + ξ_min²) /          # 理想重み和で正規化
#           Σ_k d_k (1 + ξ_k²)
#   λ    = z * Λ                   # Λ = 579.0
#   α_k  = d_k * (1 + ξ_k² + λ η_k)
# ---------------------------------------------------------------------------

const ΛWENO = 579.0

"""
WENO5-Z+M 左バイアス再構成.

呼び出し: WENO5zpm_left(f[i-2], f[i-1], f[i], f[i+1], f[i+2])
"""
function WENO5zpm_left(F_mm::Vec3, F_m::Vec3, F_i::Vec3, F_p::Vec3, F_pp::Vec3)::Vec3
    d_0, d_1, d_2 = 1/10, 6/10, 3/10
    SMALL = 1e-6

    results = MVector{3, Float64}(undef)
    for j in 1:3
        f_mm, f_m, f_i, f_p, f_pp = F_mm[j], F_m[j], F_i[j], F_p[j], F_pp[j]

        f_0 = (1/6) * ( 2*f_mm - 7*f_m + 11*f_i )
        f_1 = (1/6) * (   -f_m + 5*f_i  + 2*f_p )
        f_2 = (1/6) * ( 2*f_i + 5*f_p   -  f_pp )

        b_0 = (13/12)*(f_mm - 2*f_m + f_i)^2 + (1/4)*(f_mm - 4*f_m + 3*f_i)^2
        b_1 = (13/12)*(f_m  - 2*f_i + f_p)^2 + (1/4)*(f_m - f_p)^2
        b_2 = (13/12)*(f_i  - 2*f_p + f_pp)^2 + (1/4)*(3*f_i - 4*f_p + f_pp)^2

        tau = abs(b_0 - b_2) + SMALL

        xi_0 = tau / (b_0 + SMALL)
        xi_1 = tau / (b_1 + SMALL)
        xi_2 = tau / (b_2 + SMALL)

        xi_min = min(xi_0, xi_1, xi_2)
        z = (1 + xi_min^2) / (d_0*(1 + xi_0^2) + d_1*(1 + xi_1^2) + d_2*(1 + xi_2^2))
        lam = z * ΛWENO

        eta_0 = sqrt(1 / xi_0)
        eta_1 = sqrt(1 / xi_1)
        eta_2 = sqrt(1 / xi_2)

        a_0 = d_0 * (1 + xi_0^2 + lam * eta_0)
        a_1 = d_1 * (1 + xi_1^2 + lam * eta_1)
        a_2 = d_2 * (1 + xi_2^2 + lam * eta_2)
        a_tot = a_0 + a_1 + a_2

        results[j] = (a_0*f_0 + a_1*f_1 + a_2*f_2) / a_tot
    end
    return Vec3(results[1], results[2], results[3])
end

"""
WENO5-Z+M 右バイアス再構成.

呼び出し: WENO5zpm_right(f[i+3], f[i+2], f[i+1], f[i], f[i-1])
"""
function WENO5zpm_right(F_mm::Vec3, F_m::Vec3, F_i::Vec3, F_p::Vec3, F_pp::Vec3)::Vec3
    d_0, d_1, d_2 = 3/10, 6/10, 1/10   # 左バイアスと d_0, d_2 を交換
    SMALL = 1e-6

    results = MVector{3, Float64}(undef)
    for j in 1:3
        f_mm, f_m, f_i, f_p, f_pp = F_mm[j], F_m[j], F_i[j], F_p[j], F_pp[j]

        f_0 = (1/6) * ( 2*f_mm - 7*f_m + 11*f_i )
        f_1 = (1/6) * (   -f_m + 5*f_i  + 2*f_p )
        f_2 = (1/6) * ( 2*f_i + 5*f_p   -  f_pp )

        b_0 = (13/12)*(f_mm - 2*f_m + f_i)^2 + (1/4)*(f_mm - 4*f_m + 3*f_i)^2
        b_1 = (13/12)*(f_m  - 2*f_i + f_p)^2 + (1/4)*(f_m - f_p)^2
        b_2 = (13/12)*(f_i  - 2*f_p + f_pp)^2 + (1/4)*(3*f_i - 4*f_p + f_pp)^2

        tau = abs(b_0 - b_2) + SMALL

        xi_0 = tau / (b_0 + SMALL)
        xi_1 = tau / (b_1 + SMALL)
        xi_2 = tau / (b_2 + SMALL)

        xi_min = min(xi_0, xi_1, xi_2)
        z = (1 + xi_min^2) / (d_0*(1 + xi_0^2) + d_1*(1 + xi_1^2) + d_2*(1 + xi_2^2))
        lam = z * ΛWENO

        eta_0 = sqrt(1 / xi_0)
        eta_1 = sqrt(1 / xi_1)
        eta_2 = sqrt(1 / xi_2)

        a_0 = d_0 * (1 + xi_0^2 + lam * eta_0)
        a_1 = d_1 * (1 + xi_1^2 + lam * eta_1)
        a_2 = d_2 * (1 + xi_2^2 + lam * eta_2)
        a_tot = a_0 + a_1 + a_2

        results[j] = (a_0*f_0 + a_1*f_1 + a_2*f_2) / a_tot
    end
    return Vec3(results[1], results[2], results[3])
end

# ---------------------------------------------------------------------------
# 空間離散化 (WENO 再構成 + HLL フラックス → RHS)
# ---------------------------------------------------------------------------

"""
WENO 再構成 + HLL フラックスから空間 RHS L(U) を計算する.

`weno_left`, `weno_right` に渡す再構成関数を切り替えることで
WENO5-JS / WENO5-Z+ を統一インタフェースで使える.

n_ghost ≥ 3 が必要.
"""
function compute_rhs(U_arr::Vector{Vec3}, cfg::Config, i_start::Int, i_end::Int,
                     weno_left::Function, weno_right::Function)
    L = fill(Vec3(0.0, 0.0, 0.0), length(U_arr))
    @inbounds for i in i_start:i_end
        # 界面 i-1/2
        U_L_left  = weno_left( U_arr[i-3], U_arr[i-2], U_arr[i-1], U_arr[i],   U_arr[i+1])
        U_R_left  = weno_right(U_arr[i+2], U_arr[i+1], U_arr[i],   U_arr[i-1], U_arr[i-2])
        F_left    = HLL(U_L_left, U_R_left, cfg.gamma)

        # 界面 i+1/2
        U_L_right = weno_left( U_arr[i-2], U_arr[i-1], U_arr[i],   U_arr[i+1], U_arr[i+2])
        U_R_right = weno_right(U_arr[i+3], U_arr[i+2], U_arr[i+1], U_arr[i],   U_arr[i-1])
        F_right   = HLL(U_L_right, U_R_right, cfg.gamma)

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
    for i in eachindex(U)
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
function make_rhs_func(cfg::Config, i_start::Int, i_end::Int,
                       weno_left::Function, weno_right::Function)
    return function(_t, U_arr)
        U_bc = copy(U_arr)
        clamp_nonphysical!(U_bc, cfg.gamma)
        apply_bc!(U_bc, cfg, i_start, i_end)
        return compute_rhs(U_bc, cfg, i_start, i_end, weno_left, weno_right)
    end
end

# ---------------------------------------------------------------------------
# メインループ (WENO5 vs WENO5-Z+ 比較)
# ---------------------------------------------------------------------------

"""
WENO5-JS と WENO5-Z+ を同時に時間発展させ, 比較動画を出力する.

# Args
- `cfg`:      計算条件.
- `filename`: 出力ファイル名 (デフォルト "movie.mp4").
- `fps`:      フレームレート (デフォルト 30).
"""
function solve(cfg::Config; filename::String="movie.mp4", fps::Int=30)
    weno_pairs = [
        (WENO5_left,    WENO5_right,    "WENO5",     :blue),
        (WENO5zp_left,  WENO5zp_right,  "WENO5-Z+",  :orange),
        (WENO5zpm_left, WENO5zpm_right, "WENO5-Z+M", :green),
    ]
    n_solvers = length(weno_pairs)

    x, U0 = create_initial_condition(cfg)
    U_arr = [copy(U0) for _ in 1:n_solvers]

    fig, obs_solvers, obs_exact, title_obs = create_figure_nsolvers(
        x, cfg;
        solver_names=[wp[3] for wp in weno_pairs],
        solver_colors=[wp[4] for wp in weno_pairs],
    )

    i_start = cfg.n_ghost + 1
    i_end   = length(x) - cfg.n_ghost

    tab = RungeKutta.SSPRK3()
    rhs_funcs = [make_rhs_func(cfg, i_start, i_end, wp[1], wp[2]) for wp in weno_pairs]

    frames = Tuple{Float64, Vector{Vector{Vec3}}}[]
    push!(frames, (0.0, [copy(u) for u in U_arr]))

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
        update_observables_nsolvers!(obs_solvers, obs_exact, title_obs, x, U_list, t_frame, cfg)
    end
    println("Done: $filename")

    # 保存量の時間変化動画
    cons_filename = replace(filename, ".mp4" => "_conservation.mp4")
    record_conservation(cons_filename, cons_history, cfg;
                         solver_names=[wp[3] for wp in weno_pairs],
                         solver_colors=[wp[4] for wp in weno_pairs], fps=fps)
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
        1e-1, 3,                 # dx, n_ghost
        0.5, 0.008, 1,          # cfl, t_max, out_interval
        [(0.0, 1.1), (0.0, 1.2e5), (200.0, 450.0), (-0.5, 330.0)],  # ylims (rho, p, T, u)
    )

    solve(cfg; filename="step5.mp4")
end

main()
