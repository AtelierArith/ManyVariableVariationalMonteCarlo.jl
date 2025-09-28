using ManyVariableVariationalMonteCarlo
using Random

# 2D 拡張 Hubbard モデル (簡素化デモ)
# - mVMC-tutorial HandsOn/2D_ExHubbard の流れを Julia API で模した最小例
# - 標準の Hubbard モデルに最近接サイト間クーロン相互作用 V を追加
# - 物理的に厳密ではありませんが、VMC ワークフローの雰囲気を示します

function main()
    # 格子サイズ
    nx, ny = 4, 4
    n_sites = nx * ny
    n_electrons = div(n_sites, 2)  # 半充填を想定

    # 2D を 1D に埋め込むインデックスで初期位置を作る（チェッカー配置風）
    coords = [(ix, iy) for iy = 1:ny for ix = 1:nx]
    linear = [(iy - 1) * nx + ix for (ix, iy) in coords]
    # チェッカー配置（最近接相互作用を考慮した配置）
    pos = [i for i in linear if ((div(i-1, nx) + (i-1) % nx) % 2 == 0)]
    initial_positions = pos[1:min(n_electrons, length(pos))]

    state = VMCState{ComplexF64}(n_electrons, n_sites)
    initialize_vmc_state!(state, initial_positions)

    # 拡張 Hubbard では相互作用が複雑なため、サンプリングを少し多めに
    config = VMCConfig(
        n_samples = 200,
        n_thermalization = 100,
        n_measurement = 100,
        n_update_per_sample = 2,
        use_two_electron_updates = true,
        two_electron_probability = 0.25,
    )

    rng = MersenneTwister(2024)

    println("[07_2D_ExHubbard] nx=$(nx), ny=$(ny), n_sites=$(n_sites), n_electrons=$(n_electrons)")
    println("  initial_positions=$(initial_positions)")

    results = run_vmc_sampling!(state, config, rng)

    println("--- Results ---")
    println("energy_mean = ", results.energy_mean)
    println("energy_std  = ", results.energy_std)
    println("acceptance  = ", results.acceptance_rate)
    println("samples     = ", results.n_samples)
    println()
    println("Note: 拡張 Hubbard の最近接クーロン相互作用 V は簡素化されており、")
    println("      実際の V 項のハミルトニアンは未実装です。")
end

main()