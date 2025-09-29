# mVMC C実装 開発・使用ガイド

このドキュメントでは、C実装のmVMC（many-variable Variational Monte Carlo）の使い方について説明します。

## 概要

mVMCは量子格子模型に対する高精度な変分モンテカルロ計算を実行するためのソフトウェアです。Hubbard模型、Heisenberg模型、近藤格子模型などの強相関電子系の従来模型に対応しており、シンプルで柔軟なユーザーインターフェースと大規模並列化をサポートしています。

## ドキュメント構成

本プロジェクトには以下のドキュメントが含まれています：

### 公式マニュアル
- **英語版マニュアル**: `doc/mVMC-1.3.0_en.pdf` - 最新版の英語マニュアル
- **日本語版マニュアル**: `doc/mVMC-1.3.0_ja.pdf` - 最新版の日本語マニュアル
- **HTML版マニュアル**: `doc/userguide.html` - ブラウザで閲覧可能なマニュアル

### ソース形式ドキュメント
- **英語版ソース**: `doc/en/source/` - reStructuredText形式の英語ドキュメント
- **日本語版ソース**: `doc/ja/source/` - reStructuredText形式の日本語ドキュメント

### 主要セクション
- `intro.rst` - mVMCの概要と特徴
- `start.rst` - インストールと基本的な使用方法
- `tutorial.rst` - チュートリアル
- `standard.rst` - 標準モードの使用方法
- `expert.rst` - エキスパートモードの詳細
- `output.rst` - 出力ファイルの説明
- `algorithm.rst` - アルゴリズムの詳細
- `fourier/` - フーリエ変換関連の機能
- `wannier/` - Wannier関数との連携

## 必要な環境

mVMCのコンパイル・使用には以下が必要です：

- Cコンパイラ（Intel、Fujitsu、GNU など）
- MPIライブラリ
- LAPACKライブラリ（Intel MKL、Fujitsu、ATLAS など）
- オプション：ScaLAPACKライブラリ

## インストール方法

### 方法1: `mVMCconfig.sh` スクリプトを使用

プロジェクトのmVMCディレクトリで、環境に応じて設定スクリプトを実行します：

```bash
cd mVMC
bash mVMCconfig.sh [system_name]
```

利用可能なシステム設定：

- `gcc-fujitsu`: GCC/FCC混合コンパイル（富岳向け）
- `intel-impi`: Intel Compiler + IntelMPI（物性研システムC向け）
- `intel-mpi`: Intel Compiler + OpenMPI/MPICH2
- `aocc-aocl`: AMD AOCC + AOCL + OpenMPI/MPICH2
- `gcc-aocl`: GCC + AOCL + OpenMPI/MPICH2（物性研システムB向け）
- `gcc-mkl-mpi`: GCC + MKL + OpenMPI/MPICH2
- `gcc-x86-mpi`: GCC + OpenMPI/MPICH2（一般的なx86_64）
- `gcc-arm-mpi`: GCC + OpenMPI/MPICH2（Armv8-A）

設定後、コンパイルを実行：

```bash
make mvmc
```

実行ファイル `vmc.out` と `vmcdry.out` が `src/` ディレクトリに生成されます。

### 方法2: CMakeを使用

```bash
mkdir build
cd build
cmake -DCONFIG=gcc ../
make
```

利用可能なCONFIGオプション：
- `gcc`: GCC compiler + Linux PC
- `intel`: Intel compiler + Linux PC
- `fujitsu`: Fujitsu compiler
- `sekirei`: 物性研システムB "sekirei"

## プロジェクト構造

```
mVMC/
├── src/                    # ソースコード
│   ├── mVMC/              # メインのmVMCソースコード
│   ├── StdFace/           # 標準格子・模型定義
│   ├── pfapack/           # Pfaffian計算ライブラリ
│   └── ...
├── samples/               # サンプル計算
│   ├── tutorial_1.1/      # 基本チュートリアル
│   ├── tutorial_1.2/      # 相関関数計算
│   ├── tutorial_1.3/      # UHF計算との比較
│   └── ...
├── doc/                   # ドキュメント
├── tool/                  # 解析ツール
└── CMakeLists.txt         # CMake設定ファイル
```

## 基本的な使用方法

### 1. 入力ファイルの準備

最もシンプルな例（1次元Hubbard模型）：

```
# stan.in
L = 2
Lsub = 2
model = "FermionHubbard"
lattice = "chain"
t = 0.5
U = 4.0
ncond = 2
NSPGaussLeg = 1
2Sz = 0
NSROptItrStep =  100
NSROptItrSmp  =  10
NVMCSample    =  1000
DSROptRedCut  = 1e-8
DSROptStaDel  = 1e-2
DSROptStepDt  = 1e-2
```

### 2. 計算の実行

#### 基本的な実行手順

1. **最適化計算**：
```bash
vmcdry.out stan.in          # 入力ファイルの準備
mpirun -np 4 vmc.out namelist.def  # 最適化実行
```

2. **物理量計算**：
```bash
vmcdry.out stan.in          # 物理量計算用設定
cp green1 greenone.def      # Green関数設定
cp green2 greentwo.def      # Green関数設定
mpirun -np 4 vmc.out namelist.def zqp_opt.dat  # 計算実行
```

#### サンプル計算の実行例

チュートリアル1.2の実行：

```bash
cd samples/tutorial_1.2

# Python環境での入力ファイル生成
python3 MakeInput.py input.toml

# 最適化計算
vmcdry.out ./stan_opt.in
mpirun vmc.out namelist.def
cp ./output/zqp_opt.dat .
mv output opt

# 物理量計算
vmcdry.out ./stan_aft.in
cp green1 greenone.def
cp green2 greentwo.def
mpirun vmc.out namelist.def ./zqp_opt.dat
mv output aft

# 後処理
python3 VMClocal.py input.toml
python3 VMCcor.py input.toml
```

### 3. 出力ファイル

主要な出力ファイル：

- `output/`: 計算結果の出力ディレクトリ
- `zqp_opt.dat`: 最適化された変分パラメータ
- `Ene.dat`: エネルギーの時系列データ
- `greenone.def`, `greentwo.def`: Green関数の設定ファイル

## サンプル計算とチュートリアル

### 利用可能なサンプル

`mVMC/samples/` ディレクトリには以下のサンプル計算が含まれています：

#### 基本チュートリアル
- **tutorial_1.1**: 基本的なmVMC計算の入門
- **tutorial_1.2**: 相関関数計算の実例
- **tutorial_1.3**: UHF（非制限Hartree-Fock）計算との比較

#### 応用チュートリアル
- **tutorial_2.1**: より高度な計算手法
- **tutorial_2.2**: 専門的な解析手法

#### 標準模型とWannier関数
- **Standard/**: 標準的な格子模型のサンプル
  - `Hubbard/`: Hubbard模型
  - `Kondo/`: 近藤格子模型
  - `Spin/`: スピン模型
- **Wannier/**: Wannier関数を用いた実材料計算
  - `Sr2CuO3/`: Sr2CuO3の計算例
  - `Sr2VO4/`: Sr2VO4の計算例

### チュートリアルの実行例

詳細なチュートリアル手順については、各サンプルディレクトリ内のREADMEファイルを参照してください。

## 高度な使用方法

### 並列計算

MPIを使用した並列計算：

```bash
# 4プロセスでの実行
mpirun -np 4 vmc.out namelist.def

# SLURMを使用したジョブ投入例
#SBATCH -N 4
#SBATCH -n 64
#SBATCH -c 8
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun vmc.out namelist.def
```

### 専門家モード

より詳細な制御が必要な場合は、`namelist.def`ファイルを直接編集して計算パラメータを調整できます。

### 解析ツール

`tool/` ディレクトリには以下の解析ツールが含まれています：

- `greenr2k`: 実空間Green関数から運動量空間への変換
- `respack2wan90.py`: RESPACK形式からWannier90形式への変換

## トラブルシューティング

### よくある問題

1. **コンパイルエラー**：
   - 必要なライブラリ（MPI、LAPACK）が正しくインストールされているか確認
   - `mVMCconfig.sh`で適切なシステム設定を選択

2. **実行時エラー**：
   - 入力ファイルの形式を確認
   - MPI環境が正しく設定されているか確認

3. **収束しない**：
   - 最適化パラメータ（`NSROptItrStep`、`DSROptStepDt`など）を調整
   - より多くのサンプル数（`NVMCSample`）を設定

## 参考資料

### 公式リソース
- [mVMC公式リポジトリ](https://github.com/issp-center-dev/mVMC)
- [mVMCチュートリアル](https://github.com/issp-center-dev/mVMC-tutorial)

### ローカルドキュメント
- **マニュアル（PDF）**: `doc/mVMC-1.3.0_en.pdf` (英語) / `doc/mVMC-1.3.0_ja.pdf` (日本語)
- **ソースドキュメント**: `doc/en/source/` (英語) / `doc/ja/source/` (日本語)
- **HTML版**: `doc/userguide.html` からブラウザでアクセス可能

### 専門トピック
- **フーリエ変換機能**: `doc/en/source/fourier/` または `doc/ja/source/fourier/`
- **Wannier関数連携**: `doc/en/source/wannier/`
- **アルゴリズム詳細**: `doc/en/source/algorithm.rst` または `doc/ja/source/algorithm.rst`

## 引用

mVMCを研究で使用する場合は、以下の論文を引用してください：

> mVMC - Open-source software for many-variable variational Monte Carlo method, Takahiro Misawa, Satoshi Morita, Kazuyoshi Yoshimi, Mitsuaki Kawamura, Yuichi Motoyama, Kota Ido, Takahiro Ohgoe, Masatoshi Imada, Takeo Kato, Computer Physics Communications, 235, 447-462 (2019)
