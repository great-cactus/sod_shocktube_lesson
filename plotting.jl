# ---------------------------------------------------------------------------
# グラフ描画・動画生成
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
function extract_primitives(U_arr::Vector{Vec3}, cfg::Config)
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
function create_figure(x::Vector{Float64}, cfg::Config;
                       ylims_list::Vector{Tuple{Float64, Float64}}=cfg.ylims)
    labels = [L"\rho~\mathrm{[kg/m^3]}", L"p~\mathrm{[Pa]}",
              L"T~\mathrm{[K]}", L"u~\mathrm{[m/s]}"]

    # 初期値で Observable を作成
    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, 0.0)
    rho0, p0, T0, u0 = extract_primitives(
        [primitive_to_conservative(Vec3(exact_fields.rho[i], exact_fields.u[i], exact_fields.p[i]), cfg.gamma)
         for i in eachindex(x)], cfg)

    obs_num = [Observable(copy(d)) for d in [rho0, p0, T0, u0]]
    obs_exact = [Observable(copy(d)) for d in [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u]]
    title_obs = Observable("t = 0.0000e+00 s")

    fig = Figure(size=(800, 600))

    legend_positions = [:rt, :rt, :lt, :lt]  # rho, p, T, u

    for (idx, label) in enumerate(labels)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = Axis(fig[row, col], xlabel="x [m]", ylabel=label)
        ylims!(ax, ylims_list[idx])
        lines!(ax, x, obs_num[idx], color=:black, label="Numerical")
        lines!(ax, x, obs_exact[idx], color=:red, linestyle=:dash, label="Exact")
        axislegend(ax, position=legend_positions[idx], labelsize=10)
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
- `x`:         座標の配列.
- `U_arr`:     保存変数の配列.
- `t`:         現在時刻 [s].
- `cfg`:       計算条件.
"""
function update_observables!(obs_num, obs_exact, title_obs,
                             x::Vector{Float64}, U_arr::Vector{Vec3},
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
# 2ソルバ比較用 (HLL vs HLLC)
# ---------------------------------------------------------------------------

"""
4 パネルの Figure と Observable を作成する (2ソルバ比較版).

HLL (青), HLLC (黒), 解析解 (赤破線) を重ねて描画.

# Args
- `x`:   座標配列.
- `cfg`: 計算条件.

# Returns
`(fig, obs_hll, obs_hllc, obs_exact, title_obs)`
"""
function create_figure_compare(x::Vector{Float64}, cfg::Config;
                               ylims_list::Vector{Tuple{Float64, Float64}}=cfg.ylims)
    labels = [L"\rho~\mathrm{[kg/m^3]}", L"p~\mathrm{[Pa]}",
              L"T~\mathrm{[K]}", L"u~\mathrm{[m/s]}"]

    # 初期値で Observable を作成
    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, 0.0)
    rho0, p0, T0, u0 = extract_primitives(
        [primitive_to_conservative(Vec3(exact_fields.rho[i], exact_fields.u[i], exact_fields.p[i]), cfg.gamma)
         for i in eachindex(x)], cfg)

    obs_hll  = [Observable(copy(d)) for d in [rho0, p0, T0, u0]]
    obs_hllc = [Observable(copy(d)) for d in [rho0, p0, T0, u0]]
    obs_exact = [Observable(copy(d)) for d in [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u]]
    title_obs = Observable("t = 0.0000e+00 s")

    fig = Figure(size=(800, 600))

    legend_positions = [:rt, :rt, :lt, :lt]

    for (idx, label) in enumerate(labels)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = Axis(fig[row, col], xlabel="x [m]", ylabel=label)
        ylims!(ax, ylims_list[idx])
        lines!(ax, x, obs_hll[idx],   color=:blue,  label="HLL")
        lines!(ax, x, obs_hllc[idx],  color=:black,  label="HLLC")
        lines!(ax, x, obs_exact[idx], color=:red, linestyle=:dash, label="Exact")
        axislegend(ax, position=legend_positions[idx], labelsize=10)
    end

    Label(fig[0, :], title_obs, fontsize=14)

    return fig, obs_hll, obs_hllc, obs_exact, title_obs
end

"""
Observable を現在の 2 つの U_arr と時刻 t で更新する (2ソルバ比較版).

# Args
- `obs_hll`:   HLL 数値解の Observable 配列 (4つ).
- `obs_hllc`:  HLLC 数値解の Observable 配列 (4つ).
- `obs_exact`: 解析解の Observable 配列 (4つ).
- `title_obs`: タイトルの Observable.
- `x`:         座標の配列.
- `U_hll`:     HLL の保存変数配列.
- `U_hllc`:    HLLC の保存変数配列.
- `t`:         現在時刻 [s].
- `cfg`:       計算条件.
"""
function update_observables_compare!(obs_hll, obs_hllc, obs_exact, title_obs,
                                     x::Vector{Float64},
                                     U_hll::Vector{Vec3}, U_hllc::Vector{Vec3},
                                     t::Float64, cfg::Config)
    rho_hll, p_hll, T_hll, u_hll = extract_primitives(U_hll, cfg)
    rho_hllc, p_hllc, T_hllc, u_hllc = extract_primitives(U_hllc, cfg)

    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, t)

    for (obs, data) in zip(obs_hll, [rho_hll, p_hll, T_hll, u_hll])
        obs[] = data
    end
    for (obs, data) in zip(obs_hllc, [rho_hllc, p_hllc, T_hllc, u_hllc])
        obs[] = data
    end
    for (obs, data) in zip(obs_exact, [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u])
        obs[] = data
    end
    title_obs[] = @sprintf("t = %.4e s", t)
end
