using ManyVariableVariationalMonteCarlo
using Random

# 2D J1-J2 Heisenberg モデル (簡素化デモ)
# - mVMC-tutorial HandsOn/2D_J1J2Heisenberg の流れを Julia API で模した最小例
# - 最近接交換相互作用 J1 と次近接交換相互作用 J2 を持つ量子スピン系
# - 競合する相互作用による複雑な磁気状態（スピン液体など）
# - 物理的に厳密ではありませんが、VMC ワークフローの雰囲気を示します

function main()
    # 格子サイズ
    nx, ny = 4, 4
    n_sites = nx * ny
    n_electrons = div(n_sites, 2)  # S=1/2 スピン系では電子密度0.5を想定

    # J1-J2 競合を反映した初期配置（スピラル風配置を模擬）
    coords = [(ix, iy) for iy = 1:ny for ix = 1:nx]
    linear = [(iy - 1) * nx + ix for (ix, iy) in coords]
    # スピラル配置風（位相を持った配置）
    spiral_positions = []
    for (idx, i) in enumerate(linear)
        if idx <= n_electrons
            # 簡単なスピラル風パターン
            if (idx % 4) in [1, 2]
                push!(spiral_positions, i)
            end
        end
    end
    initial_positions = spiral_positions[1:min(n_electrons, length(spiral_positions))]

    state = VMCState{ComplexF64}(n_electrons, n_sites)
    initialize_vmc_state!(state, initial_positions)

    # J1-J2 競合系では基底状態が複雑なため、充分なサンプリングが必要
    config = VMCConfig(
        n_samples = 250,
        n_thermalization = 125,
        n_measurement = 125,
        n_update_per_sample = 3,
        use_two_electron_updates = true,
        two_electron_probability = 0.3,
    )

    rng = MersenneTwister(2024)

    println("[08_2D_J1J2_Heisenberg] nx=$(nx), ny=$(ny), n_sites=$(n_sites), n_electrons=$(n_electrons)")
    println("  initial_positions=$(initial_positions)")

    results = run_vmc_sampling!(state, config, rng)

    println("--- Results ---")
    println("energy_mean = ", results.energy_mean)
    println("energy_std  = ", results.energy_std)
    println("acceptance  = ", results.acceptance_rate)
    println("samples     = ", results.n_samples)
    println()
    println("Note: J1-J2 Heisenberg の競合する交換相互作用は簡素化されており、")
    println("      実際の J1, J2 項のハミルトニアンは未実装です。")
    println("      実際の系ではスピン液体や VBS 状態などの競合が起こります。")
end

main()