# SOD Shock Tube から学ぶ圧縮性オイラー方程式の数値計算

一次元圧縮性オイラー方程式の数値解法を，非反応 SOD ショックチューブ問題から
一段総括反応デトネーション計算へと段階的に習得するための Julia 実装集．

数値計算に関する背景は [`docs/`](docs/SODショックチューブから始める圧縮性反応オイラー方程式の数値計算.md) を参照．

---

## 必要環境

| 依存 | 用途 |
|------|------|
| Julia ≥ 1.10 | 実行環境 |
| [CairoMakie](https://github.com/MakieOrg/Makie.jl) | グラフ描画・mp4 出力 |
| [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) | 状態ベクトルの高速演算 |
| FFmpeg (システム) | CairoMakie が mp4 エンコードに使用 |

パッケージのインストール：

```julia
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

---

## 構成

```
.
├── src/
│   ├── step1_Euler-LF.jl       # Lax-Friedrichs 流束 + 前進 Euler 法
│   ├── step2_Riemann_solver.jl  # HLL / HLLC リーマンソルバ
│   ├── step3_Runge-Kutta.jl     # HLL + SSPRK3 / RK4 比較
│   ├── step4_MUSCL_scheme.jl    # MUSCL 再構築 + HLL + SSPRK2
│   ├── step5_WENO_scheme.jl     # WENO5-JS / Z+ / Z+M + HLL + SSPRK3
│   ├── step6_detonation.jl      # 反応性オイラー方程式（デトネーション）[WIP]
│   ├── common.jl                # Config・状態変数変換・初期条件（非反応系）
│   ├── det_common.jl            # Config・状態変数変換・初期条件（反応系）
│   ├── exact_solution.jl        # SOD 問題の厳密解
│   ├── riemann_solvers.jl       # HLL / HLLC 実装
│   ├── runge-kutta.jl           # Butcher tableau ベース汎用 RK ソルバ
│   └── plotting.jl              # プロット・動画生成ユーティリティ
└── docs/
    └── SODショックチューブから始める圧縮性反応オイラー方程式の数値計算.md
```

各ステップは独立したスクリプトで，`julia src/stepN_*.jl` で直接実行できる．
出力は `stepN.mp4`（物理量の時間発展）と `stepN_conservation.mp4`（保存量誤差）の 2 本．

---

## 実行例

```bash
julia --project=. src/step1_Euler-LF.jl   # Lax-Friedrichs
julia --project=. src/step5_WENO_scheme.jl # WENO5 比較
```

---

## TODO

- [ ] 反応性オイラー方程式を完成させる
- [ ] 検証問題 II（一段総括反応デトネーション）を記述
- [ ] 数値計算スキームやソルバの物理数学的背景の記述
