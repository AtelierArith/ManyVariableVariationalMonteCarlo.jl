# TODO

## 完了済み
- Julia パッケージ骨格とモジュール構成を整備し、`types.jl`, `config.jl`, `parameters.jl`, `io.jl` を組み込んだ。
- `StdFace.def` の読み込みとシミュレーション設定 (`SimulationConfig`) を Julia で再現。
- Variational パラメータ初期化ロジック（RBM/Slater/OptTrans）を Julia へ移植し、初期化テストを追加。
- Green 関数初期データ (`initial.def`) の読み込みサポートを実装。
- ReTestItems ベースのテストスイートを `@testitem` のみに整理し、`Pkg.test()` が完走することを確認。
- `Project.toml` に ReTestItems / StableRNGs 依存関係を追加し、`AGENTS.md` で貢献者向け指針を記述。

## 今後の作業
- 残りの `.def` ファイル（相互作用・投影・RBM 設定など）を Julia 側で構造化読み込みできるように実装。
- C 実装のコアアルゴリズム（スレーター更新、Pfaffian 更新、SR 解法等）を Julia に移植し、ポータブルなデータ構造へ統合。
- メモリアロケーションを抑えたホットパス最適化と `@inbounds` 等の適用条件を整理し、性能回帰テストを追加。
- サンプル入力セットに対する Julia 実装と C 実装の数値一致フィクスチャを整備し、自動比較テストを作成。
- CI もしくはスクリプトで ReTestItems 実行と性能チェックを自動化。
