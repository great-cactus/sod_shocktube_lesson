## 1. はじめに

これは一次元圧縮性オイラー方程式系の数値解法を，非反応系（SOD shock tube）から一段総括反応デトネーション計算へ段階的に習得するための手引書である．

前提知識として，偏微分方程式の基礎的な取り扱い（保存則の概念，特性線の考え方）および熱力学の基礎（理想気体の状態方程式，エンタルピ，内部エネルギなど）を仮定する．

### 本書が提供しないもの
- ソルバやスキームがなぜそのような式展開になっているかの，物理数学的背景の説明
- 数値計算にまつわるテクニック（メモリ節約，高速化...etc.）

---

## 2. 支配方程式

### 2.1 オイラー方程式

一次元圧縮性非反応オイラー方程式は，質量・運動量・エネルギーの保存則からなり，以下の保存形で書ける：

$$
\frac{\partial \boldsymbol{U}}{\partial t} + \frac{\partial \boldsymbol{F}}{\partial x} = \boldsymbol{0},
$$

ここで保存変数ベクトル $\boldsymbol{U}$ と流束ベクトル $\boldsymbol{F}$ はそれぞれ

$$
\boldsymbol{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho E \end{pmatrix}, \qquad
\boldsymbol{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ (\rho E + p) u \end{pmatrix},
$$

である．$\rho$ は密度，$u$ は速度，$p$ は圧力，$E$ は単位質量あたりの全エネルギで，

$$
E := e + \frac{1}{2} u^2,
$$

と定義する．$e$ は単位質量あたりの内部エネルギである．右辺第二項は運動エネルギである．

系を閉じるために理想気体の状態方程式を用いる：

$$
p = (\gamma - 1) \rho e,
$$

ここで $\gamma:=c_\mathrm{p}/c_\mathrm{v}$ は比熱比（定圧比熱と定積比熱の比）である．保存変数 $(\rho,\, \rho u,\, \rho E)$ から圧力を

$$
p = (\gamma - 1) \left( \rho E - \frac{1}{2} \rho u^2 \right),
$$

として復元できる．

### 2.2 保存変数と原始変数

二種類の変数表現を場面に応じて使い分ける：

- **保存変数（conservative variables）**：$\boldsymbol{U} = (\rho,\, \rho u,\, \rho E)^\top$
- **原始変数（primitive variables）**：$\boldsymbol{W} = (\rho,\, u,\, p)^\top$

保存変数は，有限体積法における保存則の離散化と直接対応するため，解の更新（時間発展）に用いる．
一方，原始変数は物理的な解釈が明瞭なので，後処理などで使われる．
（保存変数でない変数を解いても，保存は保証されない）

**保存変数 → 原始変数の変換**：

$$
\rho = U_1, \qquad u = \frac{U_2}{U_1}, \qquad p = (\gamma - 1) \left( U_3 - \frac{U_2^2}{2 U_1} \right)
$$

**原始変数 → 保存変数の変換**：

$$
U_1 = \rho, \qquad U_2 = \rho u, \qquad U_3 = \frac{p}{\gamma - 1} + \frac{1}{2} \rho u^2
$$
### 2.3 反応性オイラー方程式への拡張

一段総括反応を扱うために，反応進行度 $\lambda$（$\lambda = 0$：未反応，$\lambda = 1$：完全反応）を導入し，その輸送方程式を追加する．反応性オイラー方程式は

$$
\frac{\partial \boldsymbol{U}}{\partial t} + \frac{\partial \boldsymbol{F}}{\partial x} = \boldsymbol{S},
$$

と書ける．保存変数・流束・ソース項はそれぞれ

$$
\boldsymbol{U} = \begin{pmatrix} \rho \\ \rho u \\ \rho E \\ \rho \lambda \end{pmatrix}, \qquad
\boldsymbol{F} = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ (\rho E + p) u \\ \rho \lambda u \end{pmatrix}, \qquad
\boldsymbol{S} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ \rho \dot{\omega} \end{pmatrix},
$$

である．ここで $\dot{\omega}$ は反応速度であり，Arrhenius 型の一段総括反応モデルでは

$$
\dot{\omega} = B (1 - \lambda) \exp\left( -E/RT \right),
$$

で与えられる．$B$ はPre-exponential factor，$E$ は活性化エネルギー，$R$ は気体定数，$T$ は温度である．

エネルギー方程式に対する反応の効果は，内部エネルギーの定義を修正することで取り込む：

$$
e = \frac{p}{(\gamma - 1)\rho} - \lambda q,
$$

ここで $q$ は単位質量あたりの発熱量である．あるいは全エネルギーを

$$
E = \frac{p}{(\gamma - 1)\rho} + \frac{1}{2} u^2 - \lambda q,
$$

と定義する．$\lambda$ が増加する（反応が進行する）につれて，圧力・温度が上昇する．

温度は

$$
T = \frac{p}{\rho R / M},
$$

から求める（$M$ は混合気の平均分子量）．

---

## 3. 数値解法

### 3.1 有限体積法

計算領域 $[x_L,\, x_R]$ を $N$ 個のセルに分割する．$i$ 番目のセルの中心を $x_i$，セル幅を $\Delta x$ とする．有限体積法では，セル平均値

$$
\bar{\boldsymbol{U}}_i(t) = \frac{1}{\Delta x} \int_{x_{i-1/2}}^{x_{i+1/2}} \boldsymbol{U}(x,t) \, dx
$$

の時間発展を追跡する．保存則を各セルに対して積分すると，

$$
\frac{d \bar{\boldsymbol{U}}_i}{dt} = -\frac{1}{\Delta x} \left( \hat{\boldsymbol{F}}_{i+1/2} - \hat{\boldsymbol{F}}_{i-1/2} \right) + \boldsymbol{S}_i
$$

が得られる．ここで $\hat{\boldsymbol{F}}_{i+1/2}$ はセル界面 $x_{i+1/2}$ における**数値流束**であり，これをいかに構成するか（いかに離散化するか）が数値計算において重要である（第4章で詳述）．$\boldsymbol{S}_i$ はソース項のセル平均で，非反応系では $\boldsymbol{0}$ である．

### 3.2 時間積分

有限体積法で空間離散化した後，各セルの保存変数は以下の半離散 ODE に従う：

$$
\frac{\mathrm{d}\boldsymbol{U}_i}{\mathrm{d}t} = \boldsymbol{L}_i(\boldsymbol{U})
$$

ここで $\boldsymbol{L}_i = -\dfrac{1}{\Delta x}(\hat{\boldsymbol{F}}_{i+1/2} - \hat{\boldsymbol{F}}_{i-1/2}) + \boldsymbol{S}_i$ は空間離散化の右辺（流束差分＋ソース項）である．この ODE を陽的に時間積分する手法を以下に示す．

#### Butcher tableau による統一表現

陽的 Runge--Kutta 法は $s$ ステージの Butcher tableau $(A, \boldsymbol{b}, \boldsymbol{c})$ で統一的に記述できる．各ステップは

$$
\boldsymbol{k}_i = \boldsymbol{L}\!\left(\boldsymbol{U}^n + \Delta t \sum_{j=1}^{i-1} A_{ij}\,\boldsymbol{k}_j\right), \qquad i = 1, \ldots, s
$$

$$
\boldsymbol{U}^{n+1} = \boldsymbol{U}^n + \Delta t \sum_{i=1}^{s} b_i\,\boldsymbol{k}_i
$$

で与えられる（$t$ 依存項は省略）．

#### Euler陽解法（1次精度）

$$
\boldsymbol{U}^{n+1} = \boldsymbol{U}^n + \Delta t \, \boldsymbol{L}(\boldsymbol{U}^n)
$$

最も単純な1ステージ法．Butcher tableau は $A=[0]$，$b=[1]$，$c=[0]$．

#### 古典的 RK2（2次精度，中点法）

$$
\begin{aligned}
\boldsymbol{k}_1 &= \boldsymbol{L}(\boldsymbol{U}^n) \\
\boldsymbol{k}_2 &= \boldsymbol{L}\!\left(\boldsymbol{U}^n + \tfrac{\Delta t}{2}\boldsymbol{k}_1\right) \\
\boldsymbol{U}^{n+1} &= \boldsymbol{U}^n + \Delta t\, \boldsymbol{k}_2
\end{aligned}
$$

Butcher tableau：

$$
\begin{array}{c|cc} 0 & 0 & 0 \\ 1/2 & 1/2 & 0 \\ \hline & 0 & 1 \end{array}
$$

#### SSPRK2（2次精度，Heun 法）

SSP（Strong Stability Preserving）条件を満たすため，各ステージが Euler 法の凸結合で書ける：

$$
\begin{aligned}
\boldsymbol{U}^{(1)} &= \boldsymbol{U}^n + \Delta t \, \boldsymbol{L}(\boldsymbol{U}^n) \\
\boldsymbol{U}^{n+1} &= \frac{1}{2} \boldsymbol{U}^n + \frac{1}{2} \left[ \boldsymbol{U}^{(1)} + \Delta t \, \boldsymbol{L}(\boldsymbol{U}^{(1)}) \right]
\end{aligned}
$$

Butcher tableau：

$$
\begin{array}{c|cc} 0 & 0 & 0 \\ 1 & 1 & 0 \\ \hline & 1/2 & 1/2 \end{array}
$$

#### SSPRK3（3次精度，Gottlieb--Shu）

$$
\begin{aligned}
\boldsymbol{U}^{(1)} &= \boldsymbol{U}^n + \Delta t \, \boldsymbol{L}(\boldsymbol{U}^n) \\
\boldsymbol{U}^{(2)} &= \frac{3}{4} \boldsymbol{U}^n + \frac{1}{4} \left[ \boldsymbol{U}^{(1)} + \Delta t \, \boldsymbol{L}(\boldsymbol{U}^{(1)}) \right] \\
\boldsymbol{U}^{n+1} &= \frac{1}{3} \boldsymbol{U}^n + \frac{2}{3} \left[ \boldsymbol{U}^{(2)} + \Delta t \, \boldsymbol{L}(\boldsymbol{U}^{(2)}) \right]
\end{aligned}
$$

Butcher tableau：

$$
\begin{array}{c|ccc} 0 & 0 & 0 & 0 \\ 1 & 1 & 0 & 0 \\ 1/2 & 1/4 & 1/4 & 0 \\ \hline & 1/6 & 1/6 & 2/3 \end{array}
$$

step5（WENO5）で使用する．CFL 数の上限は $\text{CFL} \leq 1$．

#### 古典的 RK4（4次精度）

$$
\begin{aligned}
\boldsymbol{k}_1 &= \boldsymbol{L}(\boldsymbol{U}^n) \\
\boldsymbol{k}_2 &= \boldsymbol{L}\!\left(\boldsymbol{U}^n + \tfrac{\Delta t}{2}\boldsymbol{k}_1\right) \\
\boldsymbol{k}_3 &= \boldsymbol{L}\!\left(\boldsymbol{U}^n + \tfrac{\Delta t}{2}\boldsymbol{k}_2\right) \\
\boldsymbol{k}_4 &= \boldsymbol{L}\!\left(\boldsymbol{U}^n + \Delta t\,\boldsymbol{k}_3\right) \\
\boldsymbol{U}^{n+1} &= \boldsymbol{U}^n + \frac{\Delta t}{6}(\boldsymbol{k}_1 + 2\boldsymbol{k}_2 + 2\boldsymbol{k}_3 + \boldsymbol{k}_4)
\end{aligned}
$$

Butcher tableau：

$$
\begin{array}{c|cccc}
0   & 0   & 0   & 0 & 0 \\
1/2 & 1/2 & 0   & 0 & 0 \\
1/2 & 0   & 1/2 & 0 & 0 \\
1   & 0   & 0   & 1 & 0 \\
\hline
    & 1/6 & 1/3 & 1/3 & 1/6
\end{array}
$$

ただし古典的 RK4 は SSP 条件を満たさない．

#### SSP 性について

WENO や MUSCL のような高次空間スキームと組み合わせる場合，時間積分が Total Variation の増大を招かないよう **SSP（Strong Stability Preserving）条件**を満たす手法が推奨される．SSPRK2・SSPRK3 はEuler 法で安定な CFL 条件下では TVD 安定性が保証される．

### 3.3 CFL 条件と時間刻み幅の決定

十分短い時間刻み$\Delta t$を設定しないと，解が不安定化するため適切な$\Delta t$を設定するための指標が存在する．
安定性のために CFL（Courant--Friedrichs--Lewy）条件を満たす必要がある：

$$
\Delta t = \text{CFL} \cdot \frac{\Delta x}{\max_i \left( |u_i| + a_i \right)}
$$

ここで $a_i = \sqrt{\gamma p_i / \rho_i}$ は局所的な音速，CFL は CFL 数（典型的には $0 < \text{CFL} \leq 1$）である．

反応性の場合は，化学反応の時間スケールも考慮する必要がある場合がある（3.4 節参照）．

### 3.4 ソース項の取り扱い

To be done

---

## 4. リーマン問題と数値流束

### 4.1 リーマン問題の物理的意味

有限体積法のセル界面 $x_{i+1/2}$ では，一般に左右で異なる状態 $(\boldsymbol{U}_L,\, \boldsymbol{U}_R)$ が隣接する．この不連続を初期条件とする初期値問題が**リーマン問題**である．

Euler方程式のリーマン問題の解は，膨張波（rarefaction wave），接触不連続（contact discontinuity），衝撃波（shock wave）の3つの波で構成される自己相似解を持つ．数値流束関数は，このリーマン問題の解（厳密解または近似解）に基づいてセル界面での流束を評価する．

### 4.2 数値流束関数の選択肢

以下に，実装の容易さと精度のバランスが異なる代表的な数値流束を示す．いずれも左状態 $\boldsymbol{U}_L$ と右状態 $\boldsymbol{U}_R$（および対応する流束 $\boldsymbol{F}_L$，$\boldsymbol{F}_R$）から界面流束 $\hat{\boldsymbol{F}}$ を計算する関数である．

**Local Lax--Friedrichs流束**：

$$
\hat{\boldsymbol{F}}_{i+1/2} = \frac{1}{2} \left( \boldsymbol{F}_L + \boldsymbol{F}_R \right) - \frac{1}{2} \alpha \left( \boldsymbol{U}_R - \boldsymbol{U}_L \right)
$$

ここで $\alpha = \max(|u_L| + a_L,\, |u_R| + a_R)$ は界面における最大波速である．最も実装が簡単であり，デバッグの基準として有用である．ただし数値拡散が大きい．

**HLL 流束**：

$$
\hat{\boldsymbol{F}}_{i+1/2} =
\begin{cases}
\boldsymbol{F}_L & \text{if } S_L \geq 0 \\[4pt]
\displaystyle \frac{S_R \boldsymbol{F}_L - S_L \boldsymbol{F}_R + S_L S_R (\boldsymbol{U}_R - \boldsymbol{U}_L)}{S_R - S_L} & \text{if } S_L < 0 < S_R \\[4pt]
\boldsymbol{F}_R & \text{if } S_R \leq 0
\end{cases}
$$

波速の推定値として，例えば

$$
S_L = \min(u_L - a_L,\, u_R - a_R), \qquad S_R = \max(u_L + a_L,\, u_R + a_R)
$$

を用いる（Davis の推定）．

**HLLC 流束**：

HLL に加えて中間波（接触不連続）を考慮する．$S_L,\, S_R$ は HLL と同様に推定し，中間波速を

$$
S^* = \frac{p_R - p_L + \rho_L u_L(S_L - u_L) - \rho_R u_R(S_R - u_R)}{\rho_L(S_L - u_L) - \rho_R(S_R - u_R)}
$$

で求める．中間状態 $\boldsymbol{U}_{*K}$（$K = L, R$）は

$$
\boldsymbol{U}_{*K} = \rho_K \frac{S_K - u_K}{S_K - S^*} \begin{pmatrix} 1 \\ S^* \\ E_K / \rho_K + (S^* - u_K)\left[S^* + p_K / (\rho_K(S_K - u_K))\right] \end{pmatrix}
$$

であり，流束は

$$
\hat{\boldsymbol{F}} = \begin{cases} \boldsymbol{F}_L & S_L \geq 0 \\ \boldsymbol{F}_L + S_L(\boldsymbol{U}_{*L} - \boldsymbol{U}_L) & S_L < 0 \leq S^* \\ \boldsymbol{F}_R + S_R(\boldsymbol{U}_{*R} - \boldsymbol{U}_R) & S^* < 0 < S_R \\ \boldsymbol{F}_R & S_R \leq 0 \end{cases}
$$


### 4.3 高次精度化：MUSCL 再構築と傾き制限関数

4.2 節のスキームをそのまま適用すると，空間1次精度（界面でのセル平均値をそのまま左右状態とする）となる．空間2次精度を得るためには，セル内の分布を線形に再構築（MUSCL: Monotone Upstream-centered Schemes for Conservation Laws）する必要がある．

#### 4.3.1 κ-MUSCL 再構築（リミタなし）

パラメータ $\kappa \in [-1, 1]$ を用いた3点スティンシル再構築（原始変数 $\boldsymbol{W}$ に対して成分ごとに適用）：

$$
W_{i,R} = W_i + \frac{1}{4}\bigl[(1-\kappa)\Delta_{i-1/2} + (1+\kappa)\Delta_{i+1/2}\bigr]
$$
$$
W_{i+1,L} = W_{i+1} - \frac{1}{4}\bigl[(1+\kappa)\Delta_{i+1/2} + (1-\kappa)\Delta_{i+3/2}\bigr]
$$

ここで $\Delta_{i+1/2} = W_{i+1} - W_i$ である．$\kappa = 1/3$ のとき 3次精度の再構築が得られる．
#### 4.3.2 リミタ付き MUSCL

勾配比（flux ratio） $r$ を用いた比形式リミタ $\phi(r)$ を定義し，界面値を

$$
W_{i,R} = W_i + \frac{1}{2}\,\phi(r_i)\,\Delta_{i+1/2}, \qquad r_i = \frac{\Delta_{i-1/2}}{\Delta_{i+1/2}}
$$
$$
W_{i+1,L} = W_{i+1} - \frac{1}{2}\,\phi(r_{i+1})\,\Delta_{i+1/2}, \qquad r_{i+1} = \frac{\Delta_{i+3/2}}{\Delta_{i+1/2}}
$$

と再構築する（$\Delta_{i+1/2} = 0$ のときは $\phi = 0$ とする）．界面 $x_{i+1/2}$ のリーマン問題の左右状態は $(W_{i,R},\, W_{i+1,L})$ となる．

**代表的なリミタ関数** $\phi(r)$：

- **minmod**：$\phi(r) = \max(0,\, \min(r,\, 1))$  
- **van Leer**：$\phi(r) = \dfrac{r + |r|}{1 + |r|}$  
- **superbee**：$\phi(r) = \max\!\bigl(0,\, \min(2r,\, 1),\, \min(r,\, 2)\bigr)$  

### 4.4 高次精度化：WENO5 再構築

2次精度のMUSCLよりも高次精度の離散化を実現する方法の一つがWENO（Weighted Essentially Non-Oscillatory）スキームである．
#### 4.4.1 WENO5-JS（Jiang & Shu 1996）

界面 $x_{i+1/2}$ の**左バイアス再構築** $\hat{f}^-_{i+1/2}$ をスティンシル $\{i-2, i-1, i, i+1, i+2\}$ から構成する．

3つのサブスティンシル候補値：

$$
\hat{f}_0 = \frac{1}{6}(2f_{i-2} - 7f_{i-1} + 11f_i), \quad
\hat{f}_1 = \frac{1}{6}(-f_{i-1} + 5f_i + 2f_{i+1}), \quad
\hat{f}_2 = \frac{1}{6}(2f_i + 5f_{i+1} - f_{i+2})
$$

滑らかさ指標（smoothness indicator）：

$$
\beta_0 = \frac{13}{12}(f_{i-2} - 2f_{i-1} + f_i)^2 + \frac{1}{4}(f_{i-2} - 4f_{i-1} + 3f_i)^2
$$
$$
\beta_1 = \frac{13}{12}(f_{i-1} - 2f_i + f_{i+1})^2 + \frac{1}{4}(f_{i-1} - f_{i+1})^2
$$
$$
\beta_2 = \frac{13}{12}(f_i - 2f_{i+1} + f_{i+2})^2 + \frac{1}{4}(3f_i - 4f_{i+1} + f_{i+2})^2
$$

理想重み $d_0 = 1/10$，$d_1 = 6/10$，$d_2 = 3/10$ を用いた WENO-JS 重み：

$$
\alpha_k = \frac{d_k}{(\beta_k + \varepsilon)^2}, \qquad \omega_k = \frac{\alpha_k}{\alpha_0 + \alpha_1 + \alpha_2}
$$

最終的な再構築値：$\hat{f}^-_{i+1/2} = \omega_0 \hat{f}_0 + \omega_1 \hat{f}_1 + \omega_2 \hat{f}_2$

**右バイアス再構築** $\hat{f}^+_{i+1/2}$ は，スティンシルを1セル右にシフトする．
#### 4.4.2 WENO5-Z+（Borges et al. 2008）

WENO-JS の重み計算を改良し，滑らかな領域での理想重みへの収束を加速する．グローバル平滑度指標

$$
\tau = |\beta_0 - \beta_2|
$$

を用いて重みを

$$
\alpha_k = d_k \left(1 + \left(\frac{\tau}{\beta_k + \varepsilon}\right)^p\right), \qquad p = 2
$$

と定義する（WENO-JS と同じ正規化で$\omega$を求める）．滑らかな領域では $\tau \to 0$ により $\alpha_k \to d_k$（理想重み）への収束が加速され，不連続近傍では $\beta_k$ の差が大きくなるため自動的に低次スティンシルの重みが下がる．

#### 4.4.3 WENO5-Z+M（Adaptive-λ variant）

WENO-Z+ をさらに改良したもので，適応的なパラメータ $\lambda$ を用いて接触不連続の解像度を高める．

$$
\xi_k = \frac{\tau}{\beta_k + \varepsilon}, \qquad \eta_k = \sqrt{\frac{1}{\xi_k}}
$$

$$
z = \frac{1 + \xi_{\min}^2}{\sum_k d_k(1 + \xi_k^2)}, \qquad \lambda = z \Lambda \quad (\Lambda = 579)
$$

$$
\alpha_k = d_k\left(1 + \xi_k^2 + \lambda\, \eta_k\right)
$$

$\lambda$ が適応的に変化することで，滑らかな領域では理想重みに近く，不連続近傍では振動抑制が維持される．

---

## 5. 検証問題 I：SOD Shock Tube

### 5.1 問題設定

SOD shock tube問題をベンチマークとして解く．

計算領域：$x \in [0,\, 1]$．初期条件は隔膜の位置 $x_0 = 0.5$ を境に

$$
(\rho,\, u,\, p) = \begin{cases} (1.0,\, 0,\, 1.0) & \text{if } x < 0.5 \\ (0.125,\, 0,\, 0.1) & \text{if } x \geq 0.5 \end{cases}
$$

とする．比熱比は $\gamma = 1.4$ である．典型的な計算終了時刻は $t = 0.2$ である．

詳細は[[【Pythonプログラム】衝撃波菅の理論とOpenFOAMの結果比較｜宇宙に入ったカマキリ]]を参照せよ．

### 5.2 境界条件

計算領域の両端を**反射壁（closed-end）境界条件**として扱う．ゴーストセルに隣接内部セルの値をミラーリングし，運動量の符号のみ反転する：

$$
(\rho,\, \rho u,\, \rho E)_\mathrm{ghost} = \bigl(\rho_\mathrm{inner},\; -(\rho u)_\mathrm{inner},\; (\rho E)_\mathrm{inner}\bigr)
$$

これは壁面で $u = 0$ を強制するノイマン型条件であり，質量・エネルギーフラックスはゼロ，運動量フラックスは圧力のみとなる．

計算終了時刻 $t = t_\mathrm{max}$ までに波が領域端に到達しない設定であれば，この反射壁条件は内部の解に実質的な影響を与えない（ゼロ勾配条件と同等）．

### 5.3 厳密解の構造

$t > 0$ で隔膜が取り除かれると，解は左から右に向かって以下の3つの波で構成される：

1. **左向き膨張波（rarefaction fan）**：高圧側から中間状態へ連続的に遷移する波
2. **接触不連続（contact discontinuity）**：密度が不連続だが，圧力と速度は連続
3. **右向き衝撃波（shock wave）**：低圧側に伝播する不連続的な圧縮波

厳密解は Rankine--Hugoniot 条件から解析的に求まる．

---

## 6. 検証問題 II：一段総括反応デトネーション

To be done

---

## 7. 補遺

### A. ヤコビアン行列と特性速度の導出

非反応オイラー方程式を準線形形式

$$
\frac{\partial \boldsymbol{U}}{\partial t} + \boldsymbol{A} \frac{\partial \boldsymbol{U}}{\partial x} = \boldsymbol{0}, \qquad \boldsymbol{A} = \frac{\partial \boldsymbol{F}}{\partial \boldsymbol{U}}
$$

と書いたとき，ヤコビアン $\boldsymbol{A}$ の固有値が特性速度であり，

$$
\lambda_1 = u - a, \qquad \lambda_2 = u, \qquad \lambda_3 = u + a
$$

で与えられる．ここで $a = \sqrt{\gamma p / \rho}$ は音速である．

$\boldsymbol{A}$ の具体的な表式は

$$
\boldsymbol{A} = \begin{pmatrix}
0 & 1 & 0 \\
\frac{1}{2}(\gamma - 3) u^2 & (3 - \gamma) u & \gamma - 1 \\
\left[\frac{1}{2}(\gamma - 1) u^2 - H\right] u & H - (\gamma - 1) u^2 & \gamma u
\end{pmatrix}
$$

ただし $H = (E + p/\rho) = a^2/(\gamma - 1) + u^2/2$ は全エンタルピーである．

反応性の場合は4成分系となり，特性速度は $u - a,\, u,\, u,\, u + a$（$\lambda$ の移流は流速 $u$ で行われるため固有値 $u$ が二重根となる）となる．

### 参考文献

- E.F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer, 2009.（リーマンソルバ・有限体積法・SOD shock tube の厳密解に関する標準的教科書）
- R.J. LeVeque, *Finite Volume Methods for Hyperbolic Problems*, Cambridge University Press, 2002.
- W. Fickett and W.C. Davis, *Detonation: Theory and Experiment*, Dover, 2000.（デトネーション理論の古典的参考書）
- J.H.S. Lee, *The Detonation Phenomenon*, Cambridge University Press, 2008.（デトネーションの物理に関する包括的参考書）
