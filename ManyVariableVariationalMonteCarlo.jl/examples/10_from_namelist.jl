using ManyVariableVariationalMonteCarlo
using Random

# mVMC が出力した namelist.def を読み込み、Julia で軽量サンプリングを行う例
# 使い方:
#   julia --project ManyVariableVariationalMonteCarlo.jl/examples/10_from_namelist.jl path/to/namelist.def

function usage()
    println("Usage: julia --project examples/10_from_namelist.jl <path/to/namelist.def>")
    println("Hint: mVMC-tutorial の各サンプルで MakeInput.py を実行して namelist.def を生成してください。")
end

function main()
    if length(ARGS) < 1
        usage()
        return
    end

    namelist_path = ARGS[1]
    if !isfile(namelist_path)
        println("[error] namelist.def が見つかりません: ", namelist_path)
        usage()
        return
    end

    # mVMC 形式の定義ファイル群をロード
    def_files, params, face = load_vmc_configuration(namelist_path)

    # サイト数/電子数を決める（ModPara/namelist から埋まる）
    n_sites = params.nsite
    n_electrons = params.ne > 0 ? params.ne : max(1, div(n_sites, 2))

    # 初期配置を単純に作る
    initial_positions = collect(1:2:min(2 * n_electrons - 1, n_sites))
    resize!(initial_positions, min(length(initial_positions), n_electrons))
    if length(initial_positions) < n_electrons
        append!(initial_positions, collect(length(initial_positions) + 1:n_electrons))
    end

    println("[10_from_namelist]")
    println("  namelist.def      : ", namelist_path)
    println("  Nsite/Ne (loaded) : ", (n_sites, n_electrons))
    println("  Output head       : ", params.data_file_head)
    println("  Para head         : ", params.para_file_head)

    # VMC 状態と設定
    state = VMCState{ComplexF64}(n_electrons, n_sites)
    initialize_vmc_state!(state, initial_positions)

    config = VMCConfig(
        n_samples = max(100, params.nvmc_sample),
        n_thermalization = max(50, params.nvmc_warm_up),
        n_measurement = max(50, params.nvmc_sample),
        n_update_per_sample = 1,
        use_two_electron_updates = true,
        two_electron_probability = 0.15,
    )

    rng = MersenneTwister(123456)

    results = run_vmc_sampling!(state, config, rng)

    println("--- Results ---")
    println("energy_mean = ", results.energy_mean)
    println("energy_std  = ", results.energy_std)
    println("acceptance  = ", results.acceptance_rate)
    println("samples     = ", results.n_samples)
end

main()

