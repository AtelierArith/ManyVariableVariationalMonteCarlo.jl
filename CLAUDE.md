# TODO: mVMC Julia移植計画

C 実装 mVMC を Julia に移植したいです．その際，mVMC が持っている機能をいくつかのパッケージに分割して実装します。

## パッケージ分割提案

| パッケージ名 | 主な機能 | 対応するmVMCのCファイル/Julia実装 |
|------------|---------|----------------------------|
| **QuantumLatticeHamiltonians.jl** | 量子格子模型のハミルトニアン定義 | `hamiltonian.jl`, `calham*.c`, StdFace関連(`stdface.jl`, `StdFace/`), `readdef.c` |
| **VariationalWavefunctions.jl** | 変分波動関数の実装 | Slater行列式(`slater*.jl`, `slater*.c`), RBM(`rbm.jl`, `rbm.c`), Jastrow因子(`jastrow.jl`), バックフロー補正(`backflow.jl`) |
| **QuantumProjections.jl** | 量子射影演算子 | `projections.jl`, `projection.c`, `qp*.c`, `quantum_projection_*.jl` |
| **PfaffianLinearAlgebra.jl** | Pfaffian計算と線形代数 | `linalg*.jl`, `sherman_morrison.jl`, `matrix.c`, `pfupdate*.c`, pfaffpack/pfupdates関連 |
| **GreensFunctions.jl** | グリーン関数計算 | `greens.jl`, `calgrn*.c`, `locgrn*.c`, `lslocgrn*.c` |
| **VariationalMonteCarlo.jl** | モンテカルロサンプリングとVMCコア | `sampler.jl`, `updates.jl`, `vmcmake*.c`, RNG(`rng.jl`, `sfmt_rng.jl`, `sfmt/`) |
| **StochasticOptimization.jl** | 確率的再構成法と最適化 | `optimization.jl`, `stochastic_reconfiguration_precise.jl`, `stcopt*.c`, `sr_matrix_calculation.jl` |
| **VMCObservables.jl** | 物理量観測 | `observables.jl`, `average.c`, `avevar.c` |
| **LanczosExtensions.jl** | Lanczos法連携 | `lanczos*.jl`, `physcal_lanczos.c` |
| **StdFaceInterface.jl** | StdFace入力解析とexpert mode変換 | `stdface_parser.jl`, `stdface_expert_mode.jl`, `StdFace/` (137KB+69KB) |
| **MVMCFileIO.jl** | 入出力とファイルフォーマット | `io.jl`, `io_advanced.jl`, `mvmc_output_format.jl`, `initfile.c`, `readdef.c` |
| **mVMCRuntime.jl** | 実行管理とワークフロー | `vmcmain.jl`, `vmccal*.c`, `config.jl`, `parameters.jl` |
| **MVMCParallel.jl** | 並列化とMPI管理 | `parallel.jl`, `mpi_wrapper.jl`, `safempi*.c` |
| **MVMCMemory.jl** | メモリ管理 | `memory.jl`, `setmemory.c`, `workspace.c` |

## パッケージの依存関係構造

```
mVMCRuntime.jl (最上位統合パッケージ)
├── StdFaceInterface.jl
│   └── QuantumLatticeHamiltonians.jl
├── MVMCFileIO.jl
├── MVMCMemory.jl
├── MVMCParallel.jl
├── QuantumLatticeHamiltonians.jl
├── VariationalWavefunctions.jl
│   └── PfaffianLinearAlgebra.jl
├── QuantumProjections.jl
│   └── VariationalWavefunctions.jl
├── GreensFunctions.jl
│   └── VariationalWavefunctions.jl
├── VariationalMonteCarlo.jl
│   ├── VariationalWavefunctions.jl
│   ├── QuantumProjections.jl
│   └── MVMCMemory.jl
├── StochasticOptimization.jl
│   ├── PfaffianLinearAlgebra.jl
│   └── MVMCParallel.jl
├── VMCObservables.jl
│   ├── GreensFunctions.jl
│   └── MVMCFileIO.jl
└── LanczosExtensions.jl
    └── QuantumLatticeHamiltonians.jl
```

## 各パッケージの詳細

### 1. QuantumLatticeHamiltonians.jl
量子格子模型のハミルトニアン定義と構築

**機能:**
- Hubbard模型、Heisenberg模型、Kondo格子模型などの標準模型定義
- StdFaceインターフェース（格子構造の自動生成）
- 格子幾何学（Chain, Square, Triangular, Honeycomb, Kagome, Ladder等）
- ハミルトニアン項の管理（Transfer, Coulomb, Hund, Exchange, InterAll等）
- 入力ファイルパーサー（StdFace.def形式）

**対応するC実装:**
- `readdef.c`: 入力ファイル読み込み（94KB）
- `calham*.c`: ハミルトニアン計算
- `StdFace/`: 標準格子模型の定義

### 2. VariationalWavefunctions.jl
変分波動関数の全コンポーネント実装

**機能:**
- Slater行列式（スピン自由度あり/なし、実数/複素数）
- RBMネットワーク（制限ボルツマンマシン）
- Jastrow因子（Gutzwiller, density-density, spin-spin, 3体相関）
- バックフロー補正
- 固定スピン領域(FSZ)対応

**対応するC実装:**
- `slater*.c`: Slater行列式（46KB + FSZ版11KB）
- `rbm.c`: RBM実装（19KB）
- バックフロー関連機能

### 3. QuantumProjections.jl
対称性を考慮した量子射影演算子

**機能:**
- スピン射影
- 運動量射影
- 粒子数射影
- パリティ射影
- 点群対称性射影
- 時間反転射影
- 粒子-正孔射影
- Gauss-Legendre求積法による連続射影

**対応するC実装:**
- `projection.c`: 射影演算（13KB）
- `qp*.c`: 量子射影重み計算
- `gauleg.c`, `legendrepoly.c`: 数値積分

### 4. PfaffianLinearAlgebra.jl
Pfaffian計算と特殊線形代数演算

**機能:**
- Pfaffian計算とその逆行列
- Sherman-Morrison更新（rank-1更新）
- Woodbury更新（rank-k更新）
- 行列式比の効率的計算
- スレッドセーフな行列演算
- BLAS/LAPACK最適化ラッパー

**対応するC実装:**
- `matrix.c`: 行列操作（19KB）
- `pfupdate*.c`: Pfaffian更新（全変種で約100KB）
- `pfapack/`: Pfaffian計算ライブラリ
- `pfupdates/`: 高速更新アルゴリズム

### 5. GreensFunctions.jl
グリーン関数の計算と管理

**機能:**
- 1体グリーン関数
- 2体グリーン関数
- 大規模系向けグリーン関数計算
- グリーン関数キャッシュ管理
- FSZ対応グリーン関数

**対応するC実装:**
- `calgrn*.c`: グリーン関数計算（7KB + FSZ版4KB）
- `locgrn*.c`: 局所グリーン関数（全変種で約70KB）
- `lslocgrn*.c`: 大規模系グリーン関数（約47KB）

### 6. VariationalMonteCarlo.jl
モンテカルロサンプリングのコア機能

**機能:**
- メトロポリスサンプリング
- 提案アルゴリズム（単一電子、2電子、交換ホッピング等）
- 受理・棄却判定
- RNG管理（SFMT, PCG, Xoshiro等）
- 並列RNG管理
- バーンイン処理
- スプリットサンプリング
- 適応的ステップサイズ調整
- 自己相関計算

**対応するC実装:**
- `vmcmake*.c`: サンプル生成（全変種で約100KB）
- `splitloop.c`: スプリットサンプリング
- `sfmt/`: SFMT乱数生成器
- `workspace.c`: ワークスペース管理

### 7. StochasticOptimization.jl
変分パラメータの最適化アルゴリズム

**機能:**
- 確率的再構成法（SR法）
- 共役勾配法（CG法）
- ADAM最適化
- RMSprop最適化
- モーメンタム法
- 重なり行列（S行列）計算
- 力ベクトル計算
- SR方程式の求解（直接法/反復法）
- エネルギー分散計算
- 並列化対応（ScaLAPACK）

**対応するC実装:**
- `stcopt*.c`: 確率的最適化（全変種で約30KB）
- `stcopt_cg*.c`: 共役勾配実装（16KB）
- `stcopt_pdposv.c`: 並列直接法（18KB）
- `stcopt_dposv.c`: 直接法

### 8. VMCObservables.jl
物理量の測定と統計処理

**機能:**
- エネルギー測定（運動・ポテンシャル・全エネルギー）
- スピン相関関数
- 密度相関関数
- 運動量分布
- 超伝導相関
- 二重占有
- カスタム観測量
- 統計処理（平均、分散、誤差評価）
- ビニング解析

**対応するC実装:**
- `average.c`: 平均計算（8KB）
- `avevar.c`: 分散計算（8KB）

### 9. LanczosExtensions.jl
Lanczos法とVMCの連携

**機能:**
- VMCとLanczos法の統合
- 励起状態計算
- グリーン関数のLanczos表現
- エネルギー固有値計算

**対応するC実装:**
- `physcal_lanczos.c`: Lanczos物理量計算（11KB）

### 10. StdFaceInterface.jl
StdFace入力ファイル解析とexpert mode変換

**機能:**
- StdFace.def形式ファイルのパーサー
- Standard mode → Expert mode変換
- 格子幾何学の自動生成（Chain, Square, Triangular, Honeycomb, Kagome, Ladder等）
- 各種.defファイルの自動生成
  - namelist.def (マスター設定)
  - modpara.def (モデルパラメータ)
  - locspn.def (局所スピン)
  - trans.def (移動項)
  - coulombintra.def, coulombinter.def (クーロン相互作用)
  - hund.def, exchange.def (磁気相互作用)
  - gutzwiller.def, jastrow.def (変分パラメータ)
  - orbital.def (軌道パラメータ)
  - greenone.def, greentwo.def (グリーン関数定義)
  - qptrans.def (量子射影)
- Wannier90インターフェース
- 物理パラメータの妥当性検証

**対応するC実装:**
- `StdFace/src/StdFace_main.c`: メイン処理（137KB）
- `StdFace/src/StdFace_ModelUtil.c`: モデルユーティリティ（69KB）
- `StdFace/src/ChainLattice.c`: 鎖格子（15KB）
- `StdFace/src/SquareLattice.c`: 正方格子（10KB）
- `StdFace/src/TriangularLattice.c`: 三角格子（13KB）
- `StdFace/src/HoneycombLattice.c`: ハニカム格子（23KB）
- `StdFace/src/Kagome.c`: カゴメ格子（24KB）
- `StdFace/src/Ladder.c`: はしご格子（18KB）
- `StdFace/src/Orthorhombic.c`: 斜方晶格子（14KB）
- `StdFace/src/Pyrochlore.c`: パイロクロア格子（14KB）
- `StdFace/src/FCOrtho.c`: 面心斜方晶（12KB）
- `StdFace/src/export_wannier90.c`: Wannier90出力（23KB）

### 11. MVMCFileIO.jl
入出力とファイルフォーマット管理

**機能:**
- Expert mode入力ファイル読み込み
  - namelist.def解析
  - 各種.defファイルパーサー
- mVMC互換出力ファイル生成
  - zvo_out*.dat (メイン結果)
  - zvo_var*.dat (パラメータ変動)
  - zvo_SRinfo.dat (SR情報)
  - zvo_cisajs*.dat (1体グリーン関数)
  - zvo_cisajscktalt*.dat (2体グリーン関数)
  - zvo_Lanczos*.dat (Lanczos結果)
- 高度なフォーマット対応（HDF5, JSON, バイナリ）
- 初期値ファイル読み込み（Green関数、変分パラメータ）
- チェックポイント・リスタート機能
- ビニング出力（統計解析用）

**対応するC実装:**
- `initfile.c`: ファイル初期化・読み込み（6KB）
- `readdef.c`: .defファイル読み込み（94KB）
- mVMC出力ルーチン群

### 12. mVMCRuntime.jl
全体の実行管理とワークフロー制御

**機能:**
- メインワークフロー管理
  - パラメータ最適化モード
  - 物理量計算モード
  - Lanczos計算モード
- 設定管理とパラメータ検証
- 計算ステップの進行制御
- ドライラン機能
- タイミング・プロファイリング
- エラーハンドリング
- 全パッケージの統合インターフェース
- ベンチマーク機能
- 可視化インターフェース

**対応するC実装:**
- `vmcmain.c`: メインループ（25KB）
- `vmccal*.c`: VMC計算制御（全変種で約50KB）
- `vmcdry.c`: ドライラン（1.3KB）
- `vmcclock.c`: タイミング測定（9KB）
- `parameter.c`: パラメータ管理（8KB）

### 13. MVMCParallel.jl
並列化とMPI管理

**機能:**
- MPI初期化と終了処理
- プロセス間通信の安全なラッパー
- データの分散と集約
- 並列乱数生成管理
- 負荷分散
- スレッド並列とMPI並列のハイブリッド対応
- 通信タイミング最適化
- デッドロック検出とエラーハンドリング

**対応するC実装:**
- `safempi*.c`: MPI安全ラッパー（約5KB）
- mVMCのMPI並列化戦略

### 14. MVMCMemory.jl
メモリ管理と最適化

**機能:**
- 動的メモリ確保と解放
- メモリプール管理
- ワークスペース管理
- メモリレイアウト最適化
- キャッシュ効率化
- メモリ使用量のモニタリング
- ガベージコレクション制御
- 大規模計算向けメモリ戦略

**対応するC実装:**
- `setmemory.c`: メモリ確保（15KB）
- `workspace.c`: ワークスペース管理（7KB）
- `StdFace/src/setmemory.c`: StdFace用メモリ管理（8KB）

## 実装方針

### フェーズ1: 基盤パッケージ（低レイヤー）
1. **PfaffianLinearAlgebra.jl** - 数値計算の基礎
2. **MVMCMemory.jl** - メモリ管理の基礎
3. **MVMCParallel.jl** - 並列化の基礎

### フェーズ2: コア機能（物理・数理）
4. **QuantumLatticeHamiltonians.jl** - ハミルトニアン定義
5. **VariationalWavefunctions.jl** - 変分波動関数
6. **QuantumProjections.jl** - 量子射影
7. **GreensFunctions.jl** - グリーン関数

### フェーズ3: 計算エンジン
8. **VariationalMonteCarlo.jl** - モンテカルロサンプリング
9. **StochasticOptimization.jl** - パラメータ最適化
10. **VMCObservables.jl** - 物理量測定
11. **LanczosExtensions.jl** - Lanczos法連携

### フェーズ4: ユーザーインターフェース
12. **StdFaceInterface.jl** - Standard mode入力処理
13. **MVMCFileIO.jl** - ファイル入出力

### フェーズ5: 統合・実行環境
14. **mVMCRuntime.jl** - 実行管理と全体統合

## パッケージ間の主要な連携

### Standard Mode → Expert Mode ワークフロー
```
ユーザー入力 (StdFace.def)
    ↓
StdFaceInterface.jl
    ├─ パーサー: StdFace.defを解析
    ├─ 格子構造生成 (QuantumLatticeHamiltonians.jl使用)
    └─ Expert mode .defファイル群を出力
         ├─ namelist.def
         ├─ modpara.def
         ├─ locspn.def
         ├─ trans.def
         ├─ coulombintra.def / coulombinter.def
         ├─ gutzwiller.def / jastrow.def / orbital.def
         └─ greenone.def / greentwo.def
    ↓
MVMCFileIO.jl
    └─ Expert mode .defファイル群を読み込み
    ↓
mVMCRuntime.jl
    └─ VMC計算を実行
```

### VMC計算の実行フロー
```
mVMCRuntime.jl (メイン制御)
    ↓
1. 初期化フェーズ
   ├─ MVMCMemory.jl: メモリ確保
   ├─ MVMCParallel.jl: MPI初期化
   └─ VariationalWavefunctions.jl: 波動関数初期化
    ↓
2. サンプリングフェーズ
   └─ VariationalMonteCarlo.jl
       ├─ メトロポリスサンプリング
       ├─ QuantumProjections.jl: 射影演算適用
       └─ VMCObservables.jl: 物理量測定
    ↓
3. 最適化フェーズ (パラメータ最適化モード)
   └─ StochasticOptimization.jl
       ├─ S行列・力ベクトル計算
       ├─ MVMCParallel.jl: 並列線形代数
       └─ パラメータ更新
    ↓
4. 出力フェーズ
   └─ MVMCFileIO.jl
       └─ zvo_*.dat ファイル出力
```

## 注意事項

### 設計方針
- 各パッケージは独立してテスト・開発可能
- 既存の`ManyVariableVariationalMonteCarlo.jl`のコードを各パッケージに分割
- C互換性を維持しながらJuliaらしい設計に改善
- 型安定性とパフォーマンスを重視
- ドキュメント・テストを充実させる
- 他のプロジェクトでも再利用可能な設計

### 特に重要なパッケージ
1. **StdFaceInterface.jl**: ユーザーフレンドリーな入力インターフェース
   - Standard modeでの簡易入力を実現
   - Expert mode用ファイル自動生成で高度な設定も可能

2. **MVMCFileIO.jl**: C実装との完全互換性
   - 既存のmVMC入力ファイルをそのまま使用可能
   - 出力フォーマットの互換性で既存の解析スクリプトが利用可能

3. **PfaffianLinearAlgebra.jl**: 高性能計算の要
   - Pfaffian計算の効率がVMC全体の性能を決定
   - rank-1/rank-k更新アルゴリズムの最適化が重要

### パッケージ分割の利点
- **モジュール性**: 各機能を独立して開発・テスト可能
- **再利用性**: 他のプロジェクトでも個別パッケージを利用可能
- **保守性**: バグ修正や機能追加が局所化される
- **並行開発**: 複数の開発者が異なるパッケージを同時に開発可能
- **依存管理**: 必要な機能だけをインストール可能（軽量化）
