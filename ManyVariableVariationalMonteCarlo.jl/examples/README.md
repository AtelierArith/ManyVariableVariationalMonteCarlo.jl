# ManyVariableVariationalMonteCarlo.jl Examples

mVMC-tutorial のハンズオン資料を参考に、Julia 実装で最小限の VMC ワークフローを試せるサンプルを用意しました。

- 01_1D_Hubbard.jl: 1 次元 Hubbard 鎖（簡素化デモ）
- 02_2D_Heisenberg.jl: 2 次元 Heisenberg 格子（簡素化デモ）
- 10_from_namelist.jl: mVMC が出力した `namelist.def` から読み込み（mVMC-tutorial の生成物を利用）

注意: 現段階の Julia 実装は C 参照実装の I/O と流れを模した骨格であり、ハミルトニアンや波動関数構成の物理的厳密さは最小限です。サンプルは API の使い方の雰囲気とワークフローを掴むことを目的にしています。

## 前提

- パッケージディレクトリに移動して依存関係を準備
  ```bash
  cd ManyVariableVariationalMonteCarlo.jl
  julia --project -e 'using Pkg; Pkg.instantiate()'
  ```
- 実行
  ```bash
  julia --project examples/01_1D_Hubbard.jl
  julia --project examples/02_2D_Heisenberg.jl
  ```

## mVMC-tutorial の設定ファイルを使う (10_from_namelist)

mVMC-tutorial の各サンプルディレクトリには `input.toml` と `MakeInput.py` があり、Python で mVMC 用の `namelist.def` 等を生成します。`namelist.def` が生成済みであれば、Julia 側の設定ローダ `load_vmc_configuration` で読み込めます。

- 例: 1D Hubbard サンプルで `namelist.def` を生成後に実行
  ```bash
  # 例: mVMC-tutorial/HandsOn/2022_1128/Samples/1D_Hubbard 内で
  # python MakeInput.py などで mVMC 入力一式を生成（環境に応じてコマンドは異なります）
  # 生成後、次のように Julia サンプルを実行
  # 実行はパッケージディレクトリで
  cd ManyVariableVariationalMonteCarlo.jl
  julia --project examples/10_from_namelist.jl \
    ../mVMC-tutorial/HandsOn/2022_1128/Samples/1D_Hubbard/namelist.def
  ```

- `namelist.def` が未生成の場合、本サンプルはエラーメッセージとともに生成手順のヒントを表示します。

## 出力

各サンプルは以下を表示します:
- 初期設定の要約（サイト数/電子数 など）
- VMC サンプリングの統計（エネルギー平均/分散、受理率 など）

## 参考
- 物理モデルや入力ファイル仕様は `mVMC` と `mVMC-tutorial` を参照してください。
- Julia 実装の型や関数は `src/` と `test/` を見ると把握しやすいです（`VMCConfig`, `VMCState`, `run_vmc_sampling!`, `load_vmc_configuration` 等）。
