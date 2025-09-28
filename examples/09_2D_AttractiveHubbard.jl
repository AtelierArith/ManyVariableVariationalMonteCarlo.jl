using ManyVariableVariationalMonteCarlo
using Random

# 2D Attractive Hubbard モデル (簡素化デモ)
# - mVMC-tutorial HandsOn/2D_AttractiveHubbard の流れを Julia API で模した最小例
# - 引力相互作用 U < 0 による Cooper ペア形成と超伝導状態
# - BCS-BEC クロスオーバー領域の物理
# - 物理的に厳密ではありませんが、VMC ワークフローの雰囲気を示します

function main()
    # 格子サイズ
    nx, ny = 4, 4
    n_sites = nx * ny
    n_electrons = div(n_sites, 2)  # 半充填（超伝導に適した密度）

    # Cooper ペア形成を促進する初期配置（ペア配置）
    coords = [(ix, iy) for iy = 1:ny for ix = 1:nx]
    linear = [(iy - 1) * nx + ix for (ix, iy) in coords]

    # ペア配置：隣接サイトにペアを作る
    pair_positions = Int[]
    for i = 1:2:(n_sites-1)
        if length(pair_positions) < n_electrons - 1
            push!(pair_positions, i, i+1)
        end
    end
    initial_positions = pair_positions[1:min(n_electrons, length(pair_positions))]

    state = VMCState{ComplexF64}(n_electrons, n_sites)
    initialize_vmc_state!(state, initial_positions)

    # 超伝導状態の検出には注意深いサンプリングが必要
    config = VMCConfig(
        n_samples = 300,
        n_thermalization = 150,
        n_measurement = 150,
        n_update_per_sample = 2,
        use_two_electron_updates = true,
        two_electron_probability = 0.4,  # ペア更新を重視
    )

    rng = MersenneTwister(2024)

    println(
        "[09_2D_AttractiveHubbard] nx=$(nx), ny=$(ny), n_sites=$(n_sites), n_electrons=$(n_electrons)",
    )
    println("  initial_positions=$(initial_positions)")

    results = run_vmc_sampling!(state, config, rng)

    println("--- Results ---")
    println("energy_mean = ", results.energy_mean)
    println("energy_std  = ", results.energy_std)
    println("acceptance  = ", results.acceptance_rate)
    println("samples     = ", results.n_samples)
    println()
    println("Note: Attractive Hubbard の引力相互作用 U < 0 は簡素化されており、")
    println("      実際の Cooper ペア形成やオーダーパラメータは未実装です。")
    println("      実際の系では超伝導秩序パラメータ Δ の測定が重要です。")
end

main()
