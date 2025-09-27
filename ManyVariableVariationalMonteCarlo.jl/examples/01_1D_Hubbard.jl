using ManyVariableVariationalMonteCarlo
using Random

# 1D Hubbard (簡素化デモ)
# - mVMC-tutorial HandsOn/1D_Hubbard の流れを Julia API で模した最小例
# - 物理的に厳密ではありませんが、VMC ワークフローの雰囲気を示します

function main()
    # 系のサイズ（1D 鎖）
    n_sites = 8
    n_electrons = 4  # 例: 半分以下の充填を想定

    # 初期配置（等間隔に配置）
    initial_positions = collect(1:2:(2*n_electrons))

    # VMC 状態の用意
    state = VMCState{ComplexF64}(n_electrons, n_sites)
    initialize_vmc_state!(state, initial_positions)

    # サンプリング設定（軽め）
    config = VMCConfig(
        n_samples = 100,
        n_thermalization = 50,
        n_measurement = 50,
        n_update_per_sample = 1,
        use_two_electron_updates = false,
    )

    rng = MersenneTwister(2024)

    println("[01_1D_Hubbard] n_sites=$(n_sites), n_electrons=$(n_electrons)")
    println("  initial_positions=$(initial_positions)")

    results = run_vmc_sampling!(state, config, rng)

    println("--- Results ---")
    println("energy_mean = ", results.energy_mean)
    println("energy_std  = ", results.energy_std)
    println("acceptance  = ", results.acceptance_rate)
    println("samples     = ", results.n_samples)
end

main()
