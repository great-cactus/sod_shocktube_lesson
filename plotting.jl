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
                               ylims_list::Vector{Tuple{Float64, Float64}}=cfg.ylims,
                               label1::String="HLL", label2::String="HLLC",
                               color1::Symbol=:blue, color2::Symbol=:black)
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
        lines!(ax, x, obs_hll[idx],   color=color1,  label=label1)
        lines!(ax, x, obs_hllc[idx],  color=color2,  label=label2)
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

# ---------------------------------------------------------------------------
# 4ソルバ比較用 (no limiter, minmod, van Leer, superbee)
# ---------------------------------------------------------------------------

"""
4 パネルの Figure と Observable を作成する (4ソルバ比較版).

no limiter (灰), minmod (青), van Leer (緑), superbee (橙), 解析解 (赤破線).
"""
function create_figure_4solvers(x::Vector{Float64}, cfg::Config;
                                ylims_list::Vector{Tuple{Float64, Float64}}=cfg.ylims)
    labels = [L"\rho~\mathrm{[kg/m^3]}", L"p~\mathrm{[Pa]}",
              L"T~\mathrm{[K]}", L"u~\mathrm{[m/s]}"]
    solver_names  = ["No limiter", "Minmod", "Van Leer", "Superbee"]
    solver_colors = [:gray, :blue, :green, :orange]

    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, 0.0)
    rho0, p0, T0, u0 = extract_primitives(
        [primitive_to_conservative(Vec3(exact_fields.rho[i], exact_fields.u[i], exact_fields.p[i]), cfg.gamma)
         for i in eachindex(x)], cfg)

    obs_solvers = [[Observable(copy(d)) for d in [rho0, p0, T0, u0]] for _ in 1:4]
    obs_exact = [Observable(copy(d)) for d in [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u]]
    title_obs = Observable("t = 0.0000e+00 s")

    fig = Figure(size=(900, 700))
    legend_positions = [:rt, :rt, :lt, :lt]

    for (idx, label) in enumerate(labels)
        row = (idx - 1) ÷ 2 + 1
        col = (idx - 1) % 2 + 1
        ax = Axis(fig[row, col], xlabel="x [m]", ylabel=label)
        ylims!(ax, ylims_list[idx])
        for s in 1:4
            lines!(ax, x, obs_solvers[s][idx], color=solver_colors[s], label=solver_names[s])
        end
        lines!(ax, x, obs_exact[idx], color=:red, linestyle=:dash, label="Exact")
        axislegend(ax, position=legend_positions[idx], labelsize=8)
    end

    Label(fig[0, :], title_obs, fontsize=14)

    return fig, obs_solvers, obs_exact, title_obs
end

"""
Observable を 4 つの U_arr と時刻 t で更新する (4ソルバ比較版).
"""
function update_observables_4solvers!(obs_solvers, obs_exact, title_obs,
                                      x::Vector{Float64},
                                      U_list::Vector{Vector{Vec3}},
                                      t::Float64, cfg::Config)
    for (s, U_arr) in enumerate(U_list)
        rho, p, T, u = extract_primitives(U_arr, cfg)
        for (obs, data) in zip(obs_solvers[s], [rho, p, T, u])
            obs[] = data
        end
    end

    exact_cfg = make_exact_config(cfg)
    exact_fields = ExactSolution.solve(exact_cfg, x, t)
    for (obs, data) in zip(obs_exact, [exact_fields.rho, exact_fields.p, exact_fields.T, exact_fields.u])
        obs[] = data
    end
    title_obs[] = @sprintf("t = %.4e s", t)
end

# ---------------------------------------------------------------------------
# 保存量の時間変化動画
# ---------------------------------------------------------------------------

"""
保存量の時間変化を動画として記録する.

各保存量 (質量, 運動量, エネルギー) の初期値からの絶対変化量を時系列で表示する.

# Args
- `filename`:      出力ファイル名.
- `cons_history`:  保存量の履歴. 各要素は (時刻, ソルバごとの Vec3 の配列).
- `cfg`:           計算条件.
- `solver_names`:  ソルバ名の配列.
- `solver_colors`: ソルバの色の配列.
- `fps`:           フレームレート.
"""
function record_conservation(filename::String,
                              cons_history::Vector{Tuple{Float64, Vector{Vec3}}},
                              cfg::Config;
                              solver_names::Vector{String}=["Numerical"],
                              solver_colors::Vector{Symbol}=[:black],
                              fps::Int=30)
    n_solvers = length(solver_names)
    n_frames = length(cons_history)

    # 全フレームのデータを事前計算
    t_all = [cons_history[i][1] for i in 1:n_frames]
    init_vals = cons_history[1][2]

    # 各ソルバの差分時系列: diffs[solver][quantity][frame] = U_i - U_0
    diffs = [[Vector{Float64}(undef, n_frames) for _ in 1:3] for _ in 1:n_solvers]
    for i in 1:n_frames
        for s in 1:n_solvers
            curr = cons_history[i][2][s]
            init = init_vals[s]
            for k in 1:3
                diffs[s][k][i] = curr[k] - init[k]
            end
        end
    end

    quantity_labels = [
        L"\Delta(\Sigma\rho\cdot\Delta x)~\mathrm{[kg/m^2]}",
        L"\Delta(\Sigma\rho u\cdot\Delta x)~\mathrm{[kg/(m{\cdot}s)]}",
        L"\Delta(\Sigma\rho E\cdot\Delta x)~\mathrm{[J/m^2]}",
    ]

    fig = Figure(size=(800, 600))
    title_obs = Observable(@sprintf("Conservation — t = %.4e s", 0.0))
    Label(fig[0, :], title_obs, fontsize=14)

    # 全データから y 軸範囲を事前計算 (マージン 10%)
    ylims_cons = Vector{Tuple{Float64, Float64}}(undef, 3)
    for k in 1:3
        all_vals = Float64[]
        for s in 1:n_solvers
            append!(all_vals, diffs[s][k])
        end
        ymin, ymax = extrema(all_vals)
        margin = max(abs(ymax - ymin) * 0.1, 1.0e-15)
        ylims_cons[k] = (ymin - margin, ymax + margin)
    end

    obs_t = Observable(t_all[1:1])
    obs_diffs = [[Observable(diffs[s][k][1:1]) for k in 1:3] for s in 1:n_solvers]

    for k in 1:3
        ax = Axis(fig[k, 1],
                  xlabel=(k == 3 ? "t [s]" : ""),
                  ylabel=quantity_labels[k])
        xlims!(ax, (0.0, cfg.t_max * 1.02))
        ylims!(ax, ylims_cons[k])
        for s in 1:n_solvers
            lines!(ax, obs_t, obs_diffs[s][k],
                   color=solver_colors[s], label=solver_names[s])
        end
        if n_solvers > 1
            axislegend(ax, position=:lt, labelsize=8)
        end
    end

    println("Recording $n_frames conservation frames to $filename ...")
    record(fig, filename, 1:n_frames; framerate=fps) do idx
        obs_t[] = t_all[1:idx]
        for s in 1:n_solvers
            for k in 1:3
                obs_diffs[s][k][] = diffs[s][k][1:idx]
            end
        end
        title_obs[] = @sprintf("Conservation — t = %.4e s", t_all[idx])
    end
    println("Done: $filename")
end
