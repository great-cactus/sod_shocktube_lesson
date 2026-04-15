"""
Sod shock tube: exact Riemann solver (Julia版).

Toro ch.4 に基づく厳密 Riemann ソルバ.
Brent 法で star 領域の圧力を求め, 自己相似変数で各領域をサンプリングする.
"""
module ExactSolution

using Roots: find_zero, Brent

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
const R_UNIVERSAL = 8.314_462_62  # J/(mol·K)

# ---------------------------------------------------------------------------
# データ構造
# ---------------------------------------------------------------------------

"""Shock tube の物理条件."""
struct ShockTubeConfig
  p_L::Float64
  T_L::Float64
  u_L::Float64
  p_R::Float64
  T_R::Float64
  u_R::Float64
  MW::Float64
  gamma::Float64
  x_center::Float64
end

"""1次元場の配列."""
struct FieldArrays
  x::Vector{Float64}
  rho::Vector{Float64}
  p::Vector{Float64}
  T::Vector{Float64}
  u::Vector{Float64}
end

# ---------------------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------------------

"""理想気体の密度 rho = p * MW / (R * T)."""
function density(p::Float64, T::Float64, MW::Float64)::Float64
  return p * MW / (R_UNIVERSAL * T)
end

"""等エントロピー音速 a = sqrt(gamma * p / rho)."""
function sound_speed(p::Float64, rho::Float64, gamma::Float64)::Float64
  return sqrt(gamma * p / rho)
end

# ---------------------------------------------------------------------------
# Riemann ソルバ内部
# ---------------------------------------------------------------------------

"""
圧力関数 f_k(p) を評価する.

衝撃波 (p > p_k) または膨張波 (p <= p_k) の分岐を選択.

# Args
- `p`:     star 領域の試行圧力 [Pa].
- `p_k`:   片側の基準圧力 [Pa].
- `rho_k`: 片側の基準密度 [kg/m^3].
- `c_k`:   片側の基準音速 [m/s].
- `gamma`: 比熱比.
- `mu2`:   (gamma-1)/(gamma+1).

# Returns
f_k(p) のスカラ値.
"""
function pressure_function(p, p_k, rho_k, c_k, gamma, mu2)
  if p > p_k
    A = 2.0 / ((gamma + 1.0) * rho_k)
    B = mu2 * p_k
    return (p - p_k) * sqrt(A / (p + B))
  else
    return 2.0 * c_k / (gamma - 1.0) * ((p / p_k)^((gamma - 1.0) / (2.0 * gamma)) - 1.0)
  end
end

"""
波の背後の密度を計算する.

衝撃波には Rankine-Hugoniot 関係式,
膨張波には等エントロピー関係式を使用.

# Args
- `p_star`: star 領域の圧力 [Pa].
- `p_k`:    片側の基準圧力 [Pa].
- `rho_k`:  片側の基準密度 [kg/m^3].
- `gamma`:  比熱比.
- `mu2`:    (gamma-1)/(gamma+1).

# Returns
波の背後の密度 [kg/m^3].
"""
function density_behind(p_star, p_k, rho_k, gamma, mu2)
  r = p_star / p_k
  if p_star > p_k
    return rho_k * (r + mu2) / (mu2 * r + 1.0)
  else
    return rho_k * r^(1.0 / gamma)
  end
end

"""
非線形圧力方程式を解いて star 領域の状態を求める.

Brent 法で f_L(p) + f_R(p) + du = 0 を満たす p_star を求め,
u_star と両側の密度を導出する.

# Args
- `cfg`: Shock tube の物理条件.

# Returns
`(p_star, u_star, rho_star_L, rho_star_R)`.
"""
function solve_star_region(cfg::ShockTubeConfig)
  gamma = cfg.gamma
  mu2 = (gamma - 1.0) / (gamma + 1.0)

  rho_L = density(cfg.p_L, cfg.T_L, cfg.MW)
  rho_R = density(cfg.p_R, cfg.T_R, cfg.MW)
  c_L = sound_speed(cfg.p_L, rho_L, gamma)
  c_R = sound_speed(cfg.p_R, rho_R, gamma)

  equation(p) = (pressure_function(p, cfg.p_L, rho_L, c_L, gamma, mu2)
        + pressure_function(p, cfg.p_R, rho_R, c_R, gamma, mu2)
        + (cfg.u_R - cfg.u_L))

  p_min = min(cfg.p_L, cfg.p_R)
  p_max = max(cfg.p_L, cfg.p_R)

  if equation(p_min) * equation(p_max) < 0
    p_star = find_zero(equation, (p_min, p_max), Brent())
  else
    p_star = find_zero(equation, (1e-6, 10.0 * p_max), Brent())
  end

  fL_star = pressure_function(p_star, cfg.p_L, rho_L, c_L, gamma, mu2)
  u_star = cfg.u_L - fL_star
  rho_star_L = density_behind(p_star, cfg.p_L, rho_L, gamma, mu2)
  rho_star_R = density_behind(p_star, cfg.p_R, rho_R, gamma, mu2)

  return p_star, u_star, rho_star_L, rho_star_R
end

# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

"""
区分一定の初期条件 (t=0) を生成する.

# Args
- `cfg`: Shock tube の物理条件.
- `x`:   1次元空間格子 [m].

# Returns
`FieldArrays` (rho, p, T, u を左右状態で初期化).
"""
function initial_condition(cfg::ShockTubeConfig, x::Vector{Float64})::FieldArrays
  n = length(x)
  rho = Vector{Float64}(undef, n)
  p = Vector{Float64}(undef, n)
  T = Vector{Float64}(undef, n)
  u = Vector{Float64}(undef, n)

  rho_L = density(cfg.p_L, cfg.T_L, cfg.MW)
  rho_R = density(cfg.p_R, cfg.T_R, cfg.MW)

  for i in 1:n
    if x[i] < cfg.x_center
      rho[i] = rho_L
      p[i] = cfg.p_L
      T[i] = cfg.T_L
      u[i] = cfg.u_L
    else
      rho[i] = rho_R
      p[i] = cfg.p_R
      T[i] = cfg.T_R
      u[i] = cfg.u_R
    end
  end

  return FieldArrays(x, rho, p, T, u)
end

"""
時刻 t における厳密 Riemann 解を格子 x 上で計算する.

star 領域を1回解いた後, 自己相似変数 xi = (x - x_center) / t を用いて
5つの領域 (左, 膨張扇, star左, star右, 右) をサンプリングする.

# Args
- `cfg`: Shock tube の物理条件.
- `x`:   1次元空間格子 [m].
- `t`:   評価時刻 [s]. t <= 0 の場合は初期条件を返す.

# Returns
`FieldArrays` (rho, p, T, u).
"""
function solve(cfg::ShockTubeConfig, x::Vector{Float64}, t::Float64)::FieldArrays
  if t <= 0.0
    return initial_condition(cfg, x)
  end

  gamma = cfg.gamma
  rho_L = density(cfg.p_L, cfg.T_L, cfg.MW)
  rho_R = density(cfg.p_R, cfg.T_R, cfg.MW)
  c_L = sound_speed(cfg.p_L, rho_L, gamma)
  c_R = sound_speed(cfg.p_R, rho_R, gamma)

  p_star, u_star, rho_star_L, rho_star_R = solve_star_region(cfg)
  c_star_L = sqrt(gamma * p_star / rho_star_L)

  # 波速 (相似変数 xi = (x - x0) / t)
  S_HL = cfg.u_L - c_L                                  # 左膨張波 head
  S_TL = u_star - c_star_L                               # 左膨張波 tail

  if p_star > cfg.p_R                                    # 右衝撃波
    S_R = cfg.u_R + c_R * sqrt(
      (gamma + 1.0) / (2.0 * gamma) * (p_star / cfg.p_R - 1.0) + 1.0
    )
  else                                                   # 右膨張波
    c_star_R = sqrt(gamma * p_star / rho_star_R)
    S_R = u_star + c_star_R
  end

  n = length(x)
  rho_out = Vector{Float64}(undef, n)
  p_out = Vector{Float64}(undef, n)
  u_out = Vector{Float64}(undef, n)
  T_out = Vector{Float64}(undef, n)

  for i in 1:n
    xi = (x[i] - cfg.x_center) / t

    if xi <= S_HL
      # 非擾乱左領域
      rho_out[i] = rho_L
      p_out[i] = cfg.p_L
      u_out[i] = cfg.u_L
    elseif xi <= S_TL
      # 膨張扇内部
      u_fan = (2.0 / (gamma + 1.0)) * (c_L + 0.5 * (gamma - 1.0) * cfg.u_L + xi)
      c_fan = c_L - 0.5 * (gamma - 1.0) * (u_fan - cfg.u_L)
      rho_fan = rho_L * (c_fan / c_L)^(2.0 / (gamma - 1.0))
      p_fan = cfg.p_L * (rho_fan / rho_L)^gamma
      rho_out[i] = rho_fan
      p_out[i] = p_fan
      u_out[i] = u_fan
    elseif xi <= u_star
      # star 左領域
      rho_out[i] = rho_star_L
      p_out[i] = p_star
      u_out[i] = u_star
    elseif xi <= S_R
      # star 右領域
      rho_out[i] = rho_star_R
      p_out[i] = p_star
      u_out[i] = u_star
    else
      # 非擾乱右領域
      rho_out[i] = rho_R
      p_out[i] = cfg.p_R
      u_out[i] = cfg.u_R
    end

    T_out[i] = p_out[i] * cfg.MW / (rho_out[i] * R_UNIVERSAL)
  end

  return FieldArrays(x, rho_out, p_out, T_out, u_out)
end

end # module
