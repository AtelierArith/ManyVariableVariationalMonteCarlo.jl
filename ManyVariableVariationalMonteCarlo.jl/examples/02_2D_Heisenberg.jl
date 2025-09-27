using ManyVariableVariationalMonteCarlo
using Random

# 2D Heisenberg (簡素化デモ)
# - mVMC-tutorial の 2D Heisenberg をイメージ
# - 現段階の Julia 実装ではスピン厳密化は未実装のため、
#   サイト/電子数の設定とサンプリング手順の雰囲気を示します

function main()
    # 格子サイズ
    nx, ny = 4, 4
    n_sites = nx * ny
    n_electrons = div(n_sites, 2)  # 便宜上「半分」を配置

    # 2D を 1D に埋め込むインデックスで初期位置を作る（チェッカー配置風）
    coords = [(ix, iy) for iy = 1:ny for ix = 1:nx]
    linear = [(iy - 1) * nx + ix for (ix, iy) in coords]
    initial_positions =
        filter(!isnothing, map(i -> (isodd(i) ? i : nothing), linear))[1:n_electrons]

    state = VMCState{ComplexF64}(n_electrons, n_sites)
    initialize_vmc_state!(state, initial_positions)

    config = VMCConfig(
        n_samples = 120,
        n_thermalization = 60,
        n_measurement = 60,
        n_update_per_sample = 1,
        use_two_electron_updates = true,
        two_electron_probability = 0.2,
    )

    rng = MersenneTwister(2025)

    println(
        "[02_2D_Heisenberg] nx=$(nx), ny=$(ny), n_sites=$(n_sites), n_electrons=$(n_electrons)",
    )
    println("  initial_positions=$(initial_positions)")

    results = run_vmc_sampling!(state, config, rng)

    println("--- Results ---")
    println("energy_mean = ", results.energy_mean)
    println("energy_std  = ", results.energy_std)
    println("acceptance  = ", results.acceptance_rate)
    println("samples     = ", results.n_samples)
end

main()
