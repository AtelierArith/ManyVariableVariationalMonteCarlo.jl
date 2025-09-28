ManyVariableVariationalMonteCarlo.jl — TODO（mVMC との差分）

目的: mVMC（C 実装）を参照し、本パッケージに未実装/不足している機能のみを簡潔に列挙します。

最終更新: 2025-09-28

未実装（または部分実装）の機能一覧
- Wannier90 連携（StdFace_W90）
  - *_hr.dat 読み込み、格子/軌道情報の統合、二重カウント補正（Hartree/Hartree-U/Full）
  - XSF 等の補助出力

- UHF/ComplexUHF 初期化
  - UHF ソルバの呼び出し、UHF 結果からの初期波動関数生成/読み込み

- MPI/スケーラブル線形代数
  - MPI 分散サンプリング/通信最適化（現在は Threads/Distributed ベースの簡易分散のみ）
  - ScaLAPACK を用いた大規模 SR/固有値処理パス

- zvo_cisajs 系出力の完全化
  - `zvo_cisajs_*.dat` の各バリアント（スナップショット/ビン/後処理対応）
  - `zvo_cisajscktaltex_*.dat`（4 体）と `zvo_cisajscktalt_*.dat`（4 体 DC）の本実装（現状はプレースホルダ/制限付き）

- Lanczos の本格統合
  - VMC サンプルからの Krylov 構築、物理量投影、スペクトル/動的相関の出力整備
  - zvo_ls_* 出力の物理的内容とフォーマットの完全互換

- モンテカルロ更新の精緻化
  - ブロック・パフィアン更新、2 電子同時更新（FSZ 含む）の物理的比（現状は比=1 等の簡易化）
  - 大規模ローカルグリーン関数の効率的更新（lslocgrn 系）

- 1 体/4 体グリーン関数の厳密化
  - 等時 1 体 G をスレーター逆行列等から厳密計算（現状は対角近似の経路あり）
  - 4 体 G と DC 項の計算/平均化/出力の mVMC 互換

- 反周期境界/ツイスト角の一般化
  - 任意格子での APBC/Twist、トポロジカル量の計測支援

- SR/最適化制御の高度化
  - 固有値カット、対角カット、パラメータ縮退方向の除去、数値安定化オプション
  - 実数/複素数混在パラメータ最適化の統合

- 出力/I/O の拡張
  - バイナリ出力（`FlagBinary` 系）、チェックポイント/リスタート（RNG 以外のフル状態）
  - 見出し/単位/整合性の厳密化、後方互換を保った schema テスト

- 多軌道・実材料系
  - 一般軌道モデル、軌道間相互作用、Wannier90 との端到端ワークフロー

- タイマー/プロファイリング
  - 詳細タイマー、段階別集計（サンプリング/計算/最適化/出力）、メモリ監視

- RBM 拡張/連携
  - ニューロン分類（General/Charge/Spin）、階層 RBM、RBM-Lanczos 連携

テスト観点（追加が必要）
- zvo_* ファイルのスキーマ/ヘッダ/最小内容の検証（特に cisajs/ls 系）
- 小系での物理量再現（既知の基底エネルギー/相関）
- FSZ 更新/グリーン関数の整合性テスト
- Lanczos の固有値回帰/スペクトル形状のスモークテスト
- APBC/Twist の幾何別サニティチェック

作業の指針
- 作業場所: `ManyVariableVariationalMonteCarlo.jl/`
- 依存関係: `julia --project -e 'using Pkg; Pkg.instantiate()'`
- テスト: `julia --project -e 'using Pkg; Pkg.test()'`
- 絞り込み: `julia --project -e 'using ReTestItems; ReTestItems.runtests(filter="lanczos|output|greens")'`
