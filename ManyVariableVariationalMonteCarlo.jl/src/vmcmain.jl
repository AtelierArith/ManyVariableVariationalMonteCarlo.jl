"""
Main VMC Simulation Workflow

Implements the main VMC simulation workflow equivalent to vmcmain.c in the C reference implementation.
Provides parameter optimization and physics calculation modes.

Ported from vmcmain.c in the mVMC C reference implementation.
"""

using Printf
using LinearAlgebra
using Random

"""
    maybe_flush(io, sim)

Flush IO if `FlushFile` is enabled in SimulationConfig. Intended for large runs
where frequent flushing is preferred.
"""
@inline function maybe_flush(io::IO, sim)
    if sim.config.flush_file
        flush(io)
    end
end

"""
    maybe_flush_interval(io, sim, i::Int)

Flush IO every `NFileFlushInterval` lines when enabled.
"""
@inline function maybe_flush_interval(io::IO, sim, i::Int)
    N = sim.config.flush_interval
    if sim.config.flush_file && N > 0 && (i % N == 0)
        flush(io)
    end
end

"""
    VMCMode

Enumeration of VMC calculation modes.
"""
@enum VMCMode begin
    PARAMETER_OPTIMIZATION = 0  # NVMCCalMode = 0
    PHYSICS_CALCULATION = 1     # NVMCCalMode = 1
end

"""
    VMCSimulation

Main VMC simulation controller that manages the entire workflow.
"""
mutable struct VMCSimulation{T<:Union{Float64,ComplexF64}}
    # Configuration
    config::SimulationConfig
    parameters::ParameterSet{Vector{T},Vector{T},Vector{T},Vector{T}}

    # Calculation mode
    mode::VMCMode

    # Components
    slater_det::Union{Nothing,SlaterDeterminant{T}}
    rbm_network::Union{Nothing,RBMNetwork{T}}
    jastrow_factor::Union{Nothing,JastrowFactor{T}}

    # State management
    vmc_state::Union{Nothing,VMCState{T}}
    workspace::Union{Nothing,Workspace}

    # Optimization components
    sr_optimizer::Union{Nothing,StochasticReconfiguration{T}}

    # Results storage
    optimization_results::Vector{Dict{String,Any}}
    physics_results::Dict{String,Any}

    # Timing information
    timers::Dict{String,Float64}

    function VMCSimulation{T}(
        config::SimulationConfig,
        layout::ParameterLayout
    ) where {T}
        parameters = ParameterSet(layout; T=T)

        new{T}(
            config,
            parameters,
            VMCMode(config.nvmc_cal_mode),
            nothing,  # slater_det
            nothing,  # rbm_network
            nothing,  # jastrow_factor
            nothing,  # vmc_state
            nothing,  # workspace
            nothing,  # sr_optimizer
            Dict{String,Any}[],  # optimization_results
            Dict{String,Any}(),  # physics_results
            Dict{String,Float64}()  # timers
        )
    end
end

"""
    VMCSimulation(config::SimulationConfig, layout::ParameterLayout; T=ComplexF64)

Create a new VMC simulation with the specified configuration and parameter layout.
"""
VMCSimulation(config::SimulationConfig, layout::ParameterLayout; T=ComplexF64) =
    VMCSimulation{T}(config, layout)

"""
    initialize_simulation!(sim::VMCSimulation{T}) where {T}

Initialize all components of the VMC simulation.
"""
function initialize_simulation!(sim::VMCSimulation{T}) where {T}
    @info "Initializing VMC simulation..."

    # Initialize workspace and memory layout (basic placeholder for now)
    memory_layout = MemoryLayout(
        nsite = sim.config.nsites,
        ne = sim.config.nelec,
        ngutzwiller = length(sim.parameters.proj),
        nrbm_hidden = div(length(sim.parameters.rbm), 2),  # Rough estimate
        nrbm_visible = sim.config.nsites * 2  # spin up and down
    )
    # Provide a per-simulation scalar workspace so tests can verify presence
    sim.workspace = Workspace{T}()

    # Initialize VMC state
    sim.vmc_state = VMCState{T}(sim.config.nelec, sim.config.nsites)

    # Set up initial electron configuration
    initial_positions = collect(1:2:min(2*sim.config.nelec, sim.config.nsites))
    if length(initial_positions) < sim.config.nelec
        append!(initial_positions,
                collect((length(initial_positions)+1):sim.config.nelec))
    end
    initialize_vmc_state!(sim.vmc_state, initial_positions[1:sim.config.nelec])

    # Initialize a simple Hamiltonian for energy evaluation (Hubbard model)
    local lattice_sym = sim.config.lattice
    local hub_lattice::Symbol
    if lattice_sym in (:chain, :CHAIN_LATTICE)
        hub_lattice = :chain
    elseif lattice_sym in (:square, :SQUARE_LATTICE)
        hub_lattice = :square
    else
        hub_lattice = :chain
    end
    try
        sim.vmc_state.hamiltonian = create_hubbard_hamiltonian(
            sim.config.nsites,
            sim.config.nelec,
            T(sim.config.t),
            T(sim.config.u);
            lattice_type = hub_lattice,
            apbc = sim.config.apbc,
            twist_x = sim.config.twist_x,
            twist_y = sim.config.twist_y,
        )
    catch
        sim.vmc_state.hamiltonian = nothing
    end

    # Initialize wavefunction components
    initialize_wavefunction_components!(sim)

    # Initialize parameters
    layout = ParameterLayout(
        length(sim.parameters.proj),
        length(sim.parameters.rbm),
        length(sim.parameters.slater),
        length(sim.parameters.opttrans)
    )
    mask = ParameterMask(layout; default=true)  # All parameters active
    flags = ParameterFlags(T <: Complex, length(sim.parameters.rbm) > 0)
    initialize_parameters!(sim.parameters, layout, mask, flags)

    @info "VMC simulation initialized successfully"
end

"""
    initialize_wavefunction_components!(sim::VMCSimulation{T}) where {T}

Initialize the wavefunction components (Slater determinant, RBM, Jastrow).
"""
function initialize_wavefunction_components!(sim::VMCSimulation{T}) where {T}
    # Initialize Slater determinant
    if length(sim.parameters.slater) > 0
        sim.slater_det = SlaterDeterminant{T}(sim.config.nelec, sim.config.nsites)
        # Create a simple identity matrix for initialization
        init_matrix = Matrix{T}(I, sim.config.nelec, sim.config.nsites)
        initialize_slater!(sim.slater_det, init_matrix)
    end

    # Initialize RBM network if enabled
    if length(sim.parameters.rbm) > 0
        n_visible = 2 * sim.config.nsites  # spin up and down
        n_hidden = max(1, div(length(sim.parameters.rbm), n_visible + 1))
        sim.rbm_network = RBMNetwork{T}(n_visible, n_hidden)
        initialize_rbm!(sim.rbm_network)
    end

    # Initialize Jastrow factor
    if length(sim.parameters.proj) > 0
        sim.jastrow_factor = JastrowFactor{T}(sim.config.nsites, sim.config.nelec)
        # Add basic Gutzwiller parameters for each site
        for i in 1:sim.config.nsites
            add_gutzwiller_parameter!(sim.jastrow_factor, i, T(0.1))
        end
    end
end

"""
    run_simulation!(sim::VMCSimulation{T}) where {T}

Run the complete VMC simulation workflow.
"""
function run_simulation!(sim::VMCSimulation{T}) where {T}
    @info "Starting VMC simulation (mode: $(sim.mode))"

    # Initialize if not already done
    if sim.vmc_state === nothing
        initialize_simulation!(sim)
    end

    # Run appropriate calculation mode
    if sim.mode == PARAMETER_OPTIMIZATION
        run_parameter_optimization!(sim)
    elseif sim.mode == PHYSICS_CALCULATION
        run_physics_calculation!(sim)
    else
        error("Unknown VMC calculation mode: $(sim.mode)")
    end

    @info "VMC simulation completed"
end

"""
    run_parameter_optimization!(sim::VMCSimulation{T}) where {T}

Run parameter optimization using stochastic reconfiguration.
"""
function run_parameter_optimization!(sim::VMCSimulation{T}) where {T}
    @info "Starting parameter optimization..."

    # Setup optimization
    # Read optional SR-CG controls from face definition if present
    face = sim.config.face
    use_sr_cg = false
    sr_cg_max_iter = 0
    sr_cg_tol = 1e-6
    if haskey(face, :NSRCG)
        use_sr_cg = facevalue(face, :NSRCG, Int; default = 0) != 0
        sr_cg_max_iter = facevalue(face, :NSROptCGMaxIter, Int; default = 100)
        sr_cg_tol = facevalue(face, :DSROptCGTol, Float64; default = 1e-6)
    end

    opt_config = OptimizationConfig(
        method = STOCHASTIC_RECONFIGURATION,
        max_iterations = sim.config.nsr_opt_itr_step,
        convergence_tolerance = sim.config.dsr_opt_red_cut,
        learning_rate = sim.config.dsr_opt_step_dt,
        regularization_parameter = sim.config.dsr_opt_sta_del,
        use_sr_cg = use_sr_cg,
        sr_cg_max_iter = sr_cg_max_iter,
        sr_cg_tol = sr_cg_tol,
    )

    # optimizer will be created in compute_overlap_matrix! if needed

    # Optimization loop
    for iter in 1:sim.config.nsr_opt_itr_step
        sim.timers["optimization_step_$(iter)"] = @elapsed begin
            # Sample configurations
            sample_results = sample_configurations!(sim, sim.config.nsr_opt_itr_smp)

            # Compute gradients and overlaps
            gradients = compute_parameter_gradients!(sim, sample_results)
            overlap_matrix = compute_overlap_matrix!(sim, sample_results, gradients)
            force_vector = compute_force_vector!(sim, sample_results, gradients)

            # Update parameters
            if opt_config.use_sr_cg
                solve_sr_equations_cg!(sim.sr_optimizer, opt_config)
            else
                solve_sr_equations!(sim.sr_optimizer, opt_config)
            end
            parameter_update = sim.sr_optimizer.parameter_delta
            update_parameters!(sim.parameters, parameter_update)

            # Update wavefunction components (temporarily disabled for testing)
            # update_wavefunction_parameters!(sim)

            # Record results
            iter_results = Dict{String,Any}(
                "iteration" => iter,
                "energy" => sample_results.energy_mean,
                "energy_error" => sample_results.energy_std,
                "parameter_norm" => norm(parameter_update),
                "overlap_condition" => cond(overlap_matrix)
            )
            push!(sim.optimization_results, iter_results)

            # Progress reporting
            if iter % 10 == 0 || iter == sim.config.nsr_opt_itr_step
                @info @sprintf("Optimization iter %d: E = %.6f ± %.6f",
                              iter, real(sample_results.energy_mean), sample_results.energy_std)
            end
        end
    end

    @info "Parameter optimization completed"
end

"""
    run_physics_calculation!(sim::VMCSimulation{T}) where {T}

Run physics calculation to measure observables.
"""
function run_physics_calculation!(sim::VMCSimulation{T}) where {T}
    @info "Starting physics calculation..."

    # Update Slater elements with optimized parameters
    if sim.slater_det !== nothing
        update_slater_elements!(sim)
    end

    # Main sampling for physics quantities
    sim.timers["physics_sampling"] = @elapsed begin
        sample_results = sample_configurations!(sim, sim.config.nvmc_sample)

        # Measure additional observables
        observables = measure_physics_observables!(sim, sample_results)

        # Geometry-aware structure factors and momentum distribution
        geometry = get_lattice_geometry(sim)
        if sim.vmc_state !== nothing
            sfs = compute_structure_factors(sim.vmc_state; geometry = geometry)
            nk = compute_momentum_distribution(sim.vmc_state; geometry = geometry)
            observables["spin_structure_factor"] = sfs[:spin]
            observables["density_structure_factor"] = sfs[:density]
            observables["momentum_distribution"] = nk
            observables["k_grid"] = sfs[:k_grid]
        end

        # Store results
        sim.physics_results = Dict{String,Any}(
            "energy_mean" => sample_results.energy_mean,
            "energy_std" => sample_results.energy_std,
            "energy_samples" => sample_results.energy_samples,
            "double_occupation" => get(observables, "double_occupation", 0.0),
            "spin_correlation" => get(observables, "spin_correlation", ComplexF64[]),
            "density_correlation" => get(observables, "density_correlation", ComplexF64[]),
            "spin_structure_factor" => get(observables, "spin_structure_factor", ComplexF64[]),
            "density_structure_factor" => get(observables, "density_structure_factor", ComplexF64[]),
            "momentum_distribution" => get(observables, "momentum_distribution", ComplexF64[]),
            "k_grid" => get(observables, "k_grid", Any[]),
            "acceptance_rate" => sample_results.acceptance_rate,
            "acceptance_series" => sample_results.acceptance_series,
            "n_samples" => sample_results.n_samples
        )
    end

    @info @sprintf("Physics calculation: E = %.6f ± %.6f",
                   real(sim.physics_results["energy_mean"]),
                   sim.physics_results["energy_std"])
end

"""
    sample_configurations!(sim::VMCSimulation{T}, n_samples::Int) where {T}

Sample electron configurations and measure basic quantities.
"""
function sample_configurations!(sim::VMCSimulation{T}, n_samples::Int) where {T}
    config = VMCConfig(
        n_samples = n_samples,
        n_thermalization = sim.config.nvmc_warm_up,
        n_measurement = n_samples,
        n_update_per_sample = max(1, sim.config.nvmc_interval),
        use_two_electron_updates = true,
        two_electron_probability = 0.15,
    )

    # Initialize RNG
    rng = Random.MersenneTwister(12345)

    # Run sampling
    return run_vmc_sampling!(sim.vmc_state, config, rng)
end

"""
    compute_parameter_gradients!(sim::VMCSimulation{T}, sample_results) where {T}

Compute gradients of the wavefunction with respect to variational parameters.
"""
function compute_parameter_gradients!(sim::VMCSimulation{T}, sample_results) where {T}
    # Placeholder implementation - should compute actual gradients
    n_params = length(sim.parameters)
    gradients = zeros(T, n_params, sample_results.n_samples)

    # For now, return random gradients as placeholder
    for i in 1:n_params, j in 1:sample_results.n_samples
        gradients[i, j] = T(0.1 * randn())
    end

    return gradients
end

"""
    update_parameters!(params::ParameterSet, delta::Vector{T}) where {T}

Update parameter set with given deltas.
"""
function update_parameters!(params::ParameterSet, delta::Vector{T}) where {T}
    offset = 0

    # Update projection parameters
    n_proj = length(params.proj)
    if n_proj > 0
        params.proj .+= delta[offset+1:offset+n_proj]
        offset += n_proj
    end

    # Update RBM parameters
    n_rbm = length(params.rbm)
    if n_rbm > 0
        params.rbm .+= delta[offset+1:offset+n_rbm]
        offset += n_rbm
    end

    # Update Slater parameters
    n_slater = length(params.slater)
    if n_slater > 0
        params.slater .+= delta[offset+1:offset+n_slater]
        offset += n_slater
    end

    # Update OptTrans parameters
    n_opttrans = length(params.opttrans)
    if n_opttrans > 0
        params.opttrans .+= delta[offset+1:offset+n_opttrans]
        offset += n_opttrans
    end
end

"""
    compute_overlap_matrix!(sim::VMCSimulation{T}, sample_results, gradients) where {T}

Compute overlap matrix for stochastic reconfiguration.
"""
function compute_overlap_matrix!(sim::VMCSimulation{T}, sample_results, gradients) where {T}
    n_params = size(gradients, 1)
    n_samples = size(gradients, 2)

    # Create uniform weights for now
    weights = ones(Float64, n_samples)

    # Initialize optimizer if not done
    if sim.sr_optimizer === nothing
        sim.sr_optimizer = StochasticReconfiguration{T}(n_params, n_samples)
    end

    compute_overlap_matrix!(sim.sr_optimizer, Matrix(gradients'), weights)
    return sim.sr_optimizer.overlap_matrix
end

"""
    compute_force_vector!(sim::VMCSimulation{T}, sample_results, gradients) where {T}

Compute force vector for stochastic reconfiguration.
"""
function compute_force_vector!(sim::VMCSimulation{T}, sample_results, gradients) where {T}
    n_params = size(gradients, 1)
    n_samples = size(gradients, 2)

    # Create energy values (placeholder)
    energy_values = fill(sample_results.energy_mean, n_samples)
    weights = ones(Float64, n_samples)

    compute_force_vector!(sim.sr_optimizer, Matrix(gradients'), energy_values, weights)
    return sim.sr_optimizer.force_vector
end

"""
    update_wavefunction_parameters!(sim::VMCSimulation{T}) where {T}

Update wavefunction component parameters from the main parameter set.
"""
function update_wavefunction_parameters!(sim::VMCSimulation{T}) where {T}
    param_offset = 0

    # Update Jastrow parameters
    if sim.jastrow_factor !== nothing && length(sim.parameters.proj) > 0
        n_proj = min(length(sim.parameters.proj), jastrow_parameter_count(sim.jastrow_factor))
        if n_proj > 0
            jastrow_params = sim.parameters.proj[1:n_proj]
            set_jastrow_parameters!(sim.jastrow_factor, jastrow_params)
        end
        param_offset += n_proj
    end

    # Update RBM parameters
    if sim.rbm_network !== nothing && length(sim.parameters.rbm) > 0
        n_rbm = length(sim.parameters.rbm)
        rbm_params = sim.parameters.rbm[1:n_rbm]
        set_rbm_parameters!(sim.rbm_network, rbm_params)
        param_offset += n_rbm
    end

    # Update Slater parameters
    if sim.slater_det !== nothing && length(sim.parameters.slater) > 0
        # Slater parameters typically represent orbital coefficients
        # This would need proper implementation based on the specific model
    end
end

"""
    update_slater_elements!(sim::VMCSimulation{T}) where {T}

Update Slater determinant elements (equivalent to UpdateSlaterElm in C code).
"""
function update_slater_elements!(sim::VMCSimulation{T}) where {T}
    if sim.slater_det === nothing
        return
    end

    @info "Updating Slater elements..."

    # This should implement the logic from UpdateSlaterElm_fcmp() or UpdateSlaterElm_fsz()
    # For now, we'll do a basic update
    compute_determinant!(sim.slater_det)
    compute_inverse!(sim.slater_det)

    @info "Slater elements updated"
end

"""
    measure_physics_observables!(sim::VMCSimulation{T}, sample_results) where {T}

Measure additional physics observables beyond energy.
"""
function measure_physics_observables!(sim::VMCSimulation{T}, sample_results) where {T}
    observables = Dict{String,Any}()

    # Measure double occupation
    if sim.vmc_state !== nothing
        double_occ = measure_double_occupation(sim.vmc_state)
        observables["double_occupation"] = double_occ
    end

    # Equal-time spin/density correlation (geometry-aware if possible)
    if sim.vmc_state !== nothing
        max_d = max(0, min(5, sim.vmc_state.n_sites - 1))
        geometry = get_lattice_geometry(sim)
        corrs = compute_equal_time_correlations(sim.vmc_state; max_distance=max_d, geometry=geometry)
        observables["spin_correlation"] = corrs[:spin]
        observables["density_correlation"] = corrs[:density]
        # s-wave onsite pairing correlation (equal-time, snapshot-based)
        paircorr = compute_pair_correlation(sim.vmc_state; max_distance=max_d, geometry=geometry)
        observables["pair_correlation"] = paircorr
    else
        observables["spin_correlation"] = ComplexF64[]
        observables["density_correlation"] = ComplexF64[]
        observables["pair_correlation"] = ComplexF64[]
    end

    return observables
end

"""
    measure_double_occupation(state::VMCState{T}) where {T}

Measure double occupation (equivalent to CalculateDoubleOccupation in C code).
"""
function measure_double_occupation(state::VMCState{T}) where {T}
    n_sites = state.n_sites
    double_occ = 0.0

    # Count double occupancy
    for site in 1:n_sites
        n_up = count(pos -> pos == site, state.electron_positions[1:div(state.n_electrons, 2)])
        n_down = count(pos -> pos == site, state.electron_positions[div(state.n_electrons, 2)+1:end])
        double_occ += n_up * n_down
    end

    return double_occ / n_sites
end

"""
    compute_equal_time_correlations(state::VMCState{T}; max_distance::Int=5) where T

Compute equal-time spin and density correlations vs distance using the current
state's electron positions. Uses 1D |i-j| and splits electrons into up/down by
index halves (same convention as measure_double_occupation). Returns a NamedTuple.
"""
function compute_equal_time_correlations(
    state::VMCState{T};
    max_distance::Int = 5,
    geometry = nothing,
) where {T}
    n = state.n_sites
    maxd = max(0, min(max_distance, n - 1))

    n_up = zeros(Int, n)
    n_dn = zeros(Int, n)
    nup = div(state.n_electrons, 2)
    for (k, pos) in enumerate(state.electron_positions)
        if k <= nup
            n_up[pos] += 1
        else
            n_dn[pos] += 1
        end
    end
    n_tot = n_up .+ n_dn

    if geometry === nothing
        spin_corr = ComplexF64[0.0 + 0.0im for _ in 0:maxd]
        dens_corr = ComplexF64[0.0 + 0.0im for _ in 0:maxd]
        for d in 0:maxd
            cs = 0.0
            cd = 0.0
            cnt = 0
            for i in 1:n, j in 1:n
                if abs(i - j) == d
                    szi = (n_up[i] - n_dn[i]) / 2
                    szj = (n_up[j] - n_dn[j]) / 2
                    cs += szi * szj
                    cd += n_tot[i] * n_tot[j]
                    cnt += 1
                end
            end
            if cnt > 0
                spin_corr[d+1] = cs / cnt
                dens_corr[d+1] = cd / cnt
            end
        end
        return (spin = spin_corr, density = dens_corr)
    else
        coords = generate_site_coordinates(geometry)
        sums_spin = Dict{Float64,Float64}()
        sums_dens = Dict{Float64,Float64}()
        counts = Dict{Float64,Int}()
        for i in 1:n, j in 1:n
            d = round(norm(coords[i, :] .- coords[j, :]), digits=6)
            szi = (n_up[i] - n_dn[i]) / 2
            szj = (n_up[j] - n_dn[j]) / 2
            sums_spin[d] = get(sums_spin, d, 0.0) + szi * szj
            sums_dens[d] = get(sums_dens, d, 0.0) + n_tot[i] * n_tot[j]
            counts[d] = get(counts, d, 0) + 1
        end
        keys_sorted = sort(collect(keys(counts)))
        spin_corr = ComplexF64[]
        dens_corr = ComplexF64[]
        nmax = min(length(keys_sorted), maxd + 1)
        for k in 1:nmax
            key = keys_sorted[k]
            cnt = counts[key]
            push!(spin_corr, (sums_spin[key] / cnt) + 0.0im)
            push!(dens_corr, (sums_dens[key] / cnt) + 0.0im)
        end
        return (spin = spin_corr, density = dens_corr)
    end
end

"""
    get_lattice_geometry(sim::VMCSimulation)

Construct a lattice geometry using StdFace helpers based on SimulationConfig.
Returns `nothing` when geometry construction is not supported.
"""
function get_lattice_geometry(sim::VMCSimulation)
    face = sim.config.face
    L = haskey(face, :L) ? facevalue(face, :L, Int; default = sim.config.nsites) : sim.config.nsites
    W = haskey(face, :W) ? facevalue(face, :W, Int; default = 1) : 1
    lat = sim.config.lattice
    try
        if lat in (:chain, :CHAIN_LATTICE)
            return create_chain_lattice(L)
        elseif lat in (:square, :SQUARE_LATTICE)
            return create_square_lattice(L, W)
        elseif lat in (:triangular, :TRIANGULAR_LATTICE)
            return create_triangular_lattice(L, W)
        elseif lat in (:honeycomb, :HONEYCOMB_LATTICE)
            return create_honeycomb_lattice(L, W)
        elseif lat in (:kagome, :KAGOME_LATTICE)
            return create_kagome_lattice(L, W)
        elseif lat in (:ladder, :LADDER_LATTICE)
            return create_ladder_lattice(L, W)
        else
            return nothing
        end
    catch
        return nothing
    end
end

"""
    output_results(sim::VMCSimulation{T}, output_dir::String="output") where {T}

Output simulation results in mVMC-compatible format.
"""
function output_results(sim::VMCSimulation{T}, output_dir::String="output") where {T}
    mkpath(output_dir)

    if sim.mode == PARAMETER_OPTIMIZATION
        output_optimization_results(sim, output_dir)
    elseif sim.mode == PHYSICS_CALCULATION
        output_physics_results(sim, output_dir)
    end
end

"""
    output_optimization_results(sim::VMCSimulation{T}, output_dir::String) where {T}

Output parameter optimization results.
"""
function output_optimization_results(sim::VMCSimulation{T}, output_dir::String) where {T}
    # Output optimization history
    open(joinpath(output_dir, "zvo_result.dat"), "w") do f
        println(f, "# VMC Parameter Optimization Results")
        println(f, "# Iteration  Energy  Error  ParameterNorm  Condition")

        for result in sim.optimization_results
            @printf(f, "%8d  %12.6f  %12.6f  %12.6e  %12.6e\n",
                   result["iteration"],
                   real(result["energy"]),
                   result["energy_error"],
                   result["parameter_norm"],
                   result["overlap_condition"])
        end
    end

    # Output optimized parameters
    open(joinpath(output_dir, "zqp_opt.dat"), "w") do f
        println(f, "# Optimized Variational Parameters")
        println(f, "# Index  Real  Imaginary")

        all_params = [sim.parameters.proj; sim.parameters.rbm;
                     sim.parameters.slater; sim.parameters.opttrans]

        for (i, param) in enumerate(all_params)
            @printf(f, "%6d  %16.10f  %16.10f\n", i, real(param), imag(param))
        end
    end

    @info "Optimization results written to $output_dir"

    # SR info (zvo_SRinfo.dat): iterations and simple statistics
    if !isempty(sim.optimization_results)
        open(joinpath(output_dir, "zvo_SRinfo.dat"), "w") do f
            println(f, "# iter  Energy  Error  ParamNorm  Cond(S)")
            for result in sim.optimization_results
                @printf(f, "%6d  %16.10f  %12.6f  %12.6e  %12.6e\n",
                       Int(result["iteration"]), real(result["energy"]),
                       result["energy_error"], result["parameter_norm"], result["overlap_condition"])
            end
        end
    end
end

"""
    output_physics_results(sim::VMCSimulation{T}, output_dir::String) where {T}

Output physics calculation results.
"""
function output_physics_results(sim::VMCSimulation{T}, output_dir::String) where {T}
    # Main results
    open(joinpath(output_dir, "zvo_result.dat"), "w") do f
        println(f, "# VMC Physics Calculation Results")
        println(f, "# Energy: $(real(sim.physics_results["energy_mean"])) ± $(sim.physics_results["energy_std"])")
        println(f, "# Double Occupation: $(sim.physics_results["double_occupation"])")
        println(f, "# Acceptance Rate: $(sim.physics_results["acceptance_rate"])")
        println(f, "# Number of Samples: $(sim.physics_results["n_samples"])")
        maybe_flush(f, sim)
    end

    # mVMC-like zvo_out.dat summary (Etot and simple moments)
    open(joinpath(output_dir, "zvo_out.dat"), "w") do f
        println(f, "# Etot  Etot2  Sztot  Sztot2")
        Etot = real(get(sim.physics_results, "energy_mean", 0.0))
        Esamps = get(sim.physics_results, "energy_samples", ComplexF64[])
        Etot2 = isempty(Esamps) ? Etot^2 : mean(real.(Esamps) .^ 2)
        # Approximate Sztot from last state
        szt = 0.0
        if sim.vmc_state !== nothing
            n = sim.vmc_state.n_sites
            nup = div(sim.vmc_state.n_electrons, 2)
            n_up = zeros(Int, n); n_dn = zeros(Int, n)
            for (k, pos) in enumerate(sim.vmc_state.electron_positions)
                if k <= nup; n_up[pos] += 1; else; n_dn[pos] += 1; end
            end
            szt = 0.5 * (sum(n_up) - sum(n_dn))
        end
        Sztot2 = szt^2
        @printf(f, "%16.10f  %16.10f  %16.10f  %16.10f\n", Etot, Etot2, szt, Sztot2)
        maybe_flush(f, sim)
    end

    # Optional: custom observables (example: total magnetization as custom)
    open(joinpath(output_dir, "zvo_custom.dat"), "w") do f
        println(f, "# name  mean  stderr")
        if sim.vmc_state !== nothing
            n = sim.vmc_state.n_sites
            nup = div(sim.vmc_state.n_electrons, 2)
            n_up = zeros(Int, n); n_dn = zeros(Int, n)
            for (k, pos) in enumerate(sim.vmc_state.electron_positions)
                if k <= nup; n_up[pos] += 1; else; n_dn[pos] += 1; end
            end
            szt = 0.5 * (sum(n_up) - sum(n_dn))
            @printf(f, "%s  %16.10f  %16.10f
", "magnetization", szt, 0.0)
        end
        maybe_flush(f, sim)
    end

    # Write correlation data if present
    let spin = get(sim.physics_results, "spin_correlation", ComplexF64[]),
        dens = get(sim.physics_results, "density_correlation", ComplexF64[]),
        pair = get(sim.physics_results, "pair_correlation", ComplexF64[])
        if !isempty(spin) || !isempty(dens) || !isempty(pair)
            open(joinpath(output_dir, "zvo_corr.dat"), "w") do f
                println(f, "# distance  Re[Cspin(d)]  Im[Cspin(d)]  Re[Cdens(d)]  Im[Cdens(d)]  Re[Cpair_s(d)]  Im[Cpair_s(d)]")
                nd = maximum([length(spin), length(dens), length(pair), 0])
                for d in 0:nd-1
                    s = d < length(spin) ? spin[d+1] : 0.0 + 0.0im
                    n = d < length(dens) ? dens[d+1] : 0.0 + 0.0im
                    p = d < length(pair) ? pair[d+1] : 0.0 + 0.0im
                    @printf(f, "%6d  %16.10f %16.10f  %16.10f %16.10f  %16.10f %16.10f\n",
                           d, real(s), imag(s), real(n), imag(n), real(p), imag(p))
                    maybe_flush_interval(f, sim, d+1)
                end
                maybe_flush(f, sim)
            end
        end
    end

    # Write energy time series if available
    if haskey(sim.physics_results, "energy_samples")
        samples = sim.physics_results["energy_samples"]
        open(joinpath(output_dir, "zvo_energy.dat"), "w") do f
            println(f, "# index  Re[E]  Im[E]")
            for (i, e) in enumerate(samples)
                @printf(f, "%8d  %16.10f  %16.10f\n", i, real(e), imag(e))
                maybe_flush_interval(f, sim, i)
            end
            maybe_flush(f, sim)
        end
    end

    # Write acceptance time series if present
    if haskey(sim.physics_results, "acceptance_series")
        acc = sim.physics_results["acceptance_series"]
        open(joinpath(output_dir, "zvo_accept.dat"), "w") do f
            println(f, "# index  acceptance")
            for (i, a) in enumerate(acc)
                @printf(f, "%8d  %16.10f\n", i, a)
                maybe_flush_interval(f, sim, i)
            end
            maybe_flush(f, sim)
        end
    end

    # Write structure factors and momentum distribution if present
    let ks = get(sim.physics_results, "k_grid", Any[]),
        ssf = get(sim.physics_results, "spin_structure_factor", ComplexF64[]),
        dsf = get(sim.physics_results, "density_structure_factor", ComplexF64[]),
        nk = get(sim.physics_results, "momentum_distribution", ComplexF64[])
        if !isempty(ssf) || !isempty(dsf)
            open(joinpath(output_dir, "zvo_struct.dat"), "w") do f
                if !isempty(ks) && ks[1] isa Tuple
                    nd = length(ks[1])
                    if nd == 1
                        println(f, "# k  Re[Ss] Im[Ss]  Re[Sn] Im[Sn]")
                        for i in 1:length(ssf)
                            @printf(f, "%16.10f  %16.10f %16.10f  %16.10f %16.10f\n",
                                   ks[i][1], real(ssf[i]), imag(ssf[i]), real(dsf[i]), imag(dsf[i]))
                            maybe_flush_interval(f, sim, i)
                        end
                    elseif nd == 2
                        println(f, "# kx ky  Re[Ss] Im[Ss]  Re[Sn] Im[Sn]")
                        for i in 1:length(ssf)
                            @printf(f, "%16.10f %16.10f  %16.10f %16.10f  %16.10f %16.10f\n",
                                   ks[i][1], ks[i][2], real(ssf[i]), imag(ssf[i]), real(dsf[i]), imag(dsf[i]))
                            maybe_flush_interval(f, sim, i)
                        end
                    else
                        println(f, "# k-vector  Re[Ss] Im[Ss]  Re[Sn] Im[Sn]")
                        for i in 1:length(ssf)
                            @printf(f, "%s  %16.10f %16.10f  %16.10f %16.10f\n",
                                   string(ks[i]), real(ssf[i]), imag(ssf[i]), real(dsf[i]), imag(dsf[i]))
                            maybe_flush_interval(f, sim, i)
                        end
                    end
                else
                    println(f, "# idx  Re[Ss] Im[Ss]  Re[Sn] Im[Sn]")
                    for i in 1:length(ssf)
                        @printf(f, "%6d  %16.10f %16.10f  %16.10f %16.10f\n",
                               i, real(ssf[i]), imag(ssf[i]), real(dsf[i]), imag(dsf[i]))
                        maybe_flush_interval(f, sim, i)
                    end
                end
                maybe_flush(f, sim)
            end
        end
        if !isempty(nk)
            open(joinpath(output_dir, "zvo_momentum.dat"), "w") do f
                if !isempty(ks) && ks[1] isa Tuple
                    nd = length(ks[1])
                    if nd == 1
                        println(f, "# k  Re[nk] Im[nk]")
                        for i in 1:length(nk)
                            @printf(f, "%16.10f  %16.10f %16.10f\n", ks[i][1], real(nk[i]), imag(nk[i]))
                            maybe_flush_interval(f, sim, i)
                        end
                    elseif nd == 2
                        println(f, "# kx ky  Re[nk] Im[nk]")
                        for i in 1:length(nk)
                            @printf(f, "%16.10f %16.10f  %16.10f %16.10f\n", ks[i][1], ks[i][2], real(nk[i]), imag(nk[i]))
                            maybe_flush_interval(f, sim, i)
                        end
                    else
                        println(f, "# k-vector  Re[nk] Im[nk]")
                        for i in 1:length(nk)
                            @printf(f, "%s  %16.10f %16.10f\n", string(ks[i]), real(nk[i]), imag(nk[i]))
                            maybe_flush_interval(f, sim, i)
                        end
                    end
                else
                    println(f, "# idx  Re[nk] Im[nk]")
                    for i in 1:length(nk)
                        @printf(f, "%6d  %16.10f %16.10f\n", i, real(nk[i]), imag(nk[i]))
                        maybe_flush_interval(f, sim, i)
                    end
                end
                maybe_flush(f, sim)
            end
        end
    end

    # Write one-body Green function (advanced local or snapshot) (zvo_cisajs.dat)
    if sim.vmc_state !== nothing
        # Switch to LocalGreenFunction when requested via face definition
        Gup, Gdn = if haskey(sim.config.face, :OneBodyG) && sim.config.face[:OneBodyG] == true
            compute_onebody_green_local(sim)
        else
            compute_onebody_green(sim.vmc_state)
        end
        open(joinpath(output_dir, "zvo_cisajs.dat"), "w") do f
            println(f, "# i  s  j  t   Re[G]   Im[G]")
            n = size(Gup, 1)
            # spin index: 1 => up, 2 => down
            for i in 1:n, j in 1:n
                @printf(f, "%6d %2d %6d %2d  %16.10f %16.10f\n", i, 1, j, 1, real(Gup[i,j]), imag(Gup[i,j]))
                @printf(f, "%6d %2d %6d %2d  %16.10f %16.10f\n", i, 2, j, 2, real(Gdn[i,j]), imag(Gdn[i,j]))
                maybe_flush_interval(f, sim, (i-1)*n + j)
            end
            maybe_flush(f, sim)
        end
        # 4-body Green function: compute if enabled and small, else write placeholders
        if haskey(sim.config.face, :TwoBodyG) && sim.config.face[:TwoBodyG] == true
            open(joinpath(output_dir, "zvo_cisajscktaltex.dat"), "w") do f
                println(f, "# 4-body Green function (equal-time)")
                println(f, "# i s j t k u l v   Re[G4]   Im[G4]")
                n = sim.vmc_state.n_sites
                ne = sim.vmc_state.n_electrons
                if n <= 8 && ne <= 8
                    gf = LocalGreenFunction{ComplexF64}(n, ne, 2)
                    fill!(gf.ele_idx, 0); fill!(gf.ele_cfg, 0); fill!(gf.ele_num, 0); fill!(gf.proj_cnt, 0)
                    nup = div(ne, 2)
                    for k in 1:nup
                        pos = sim.vmc_state.electron_positions[k]
                        gf.ele_idx[k] = pos
                        rsi = pos + (1 - 1) * n
                        gf.ele_cfg[rsi] = 1; gf.ele_num[rsi] = 1
                    end
                    for k in 1:(ne - nup)
                        pos = sim.vmc_state.electron_positions[nup + k]
                        gf.ele_idx[nup + k] = pos
                        rsi = pos + (2 - 1) * n
                        gf.ele_cfg[rsi] = 1; gf.ele_num[rsi] = 1
                    end
                    ip = one(ComplexF64)
                    maxrows = 20000
                    count = 0
                    for i in 1:n, j in 1:n, k in 1:n, l in 1:n
                        z = green_function_2body!(gf, i, j, k, l, 1, 1, ip, gf.ele_idx, gf.ele_cfg, gf.ele_num, gf.proj_cnt)
                        @printf(f, "%6d %2d %6d %2d %6d %2d %6d %2d  %16.10f %16.10f\n",
                               i, 1, j, 1, k, 1, l, 1, real(z), imag(z))
                        count += 1
                        maybe_flush_interval(f, sim, count)
                        if count >= maxrows
                            break
                        end
                    end
                end
                maybe_flush(f, sim)
            end
            open(joinpath(output_dir, "zvo_cisajscktalt.dat"), "w") do f
                println(f, "# 4-body Green function DC (placeholder)")
                println(f, "# i s j t k u l v   Re[G4dc]   Im[G4dc]")
                maybe_flush(f, sim)
            end
        else
            open(joinpath(output_dir, "zvo_cisajscktaltex.dat"), "w") do f
                println(f, "# 4-body Green function (placeholder)")
                println(f, "# i s j t k u l v   Re[G4]   Im[G4]")
                maybe_flush(f, sim)
            end
            open(joinpath(output_dir, "zvo_cisajscktalt.dat"), "w") do f
                println(f, "# 4-body Green function DC (placeholder)")
                println(f, "# i s j t k u l v   Re[G4dc]   Im[G4dc]")
                maybe_flush(f, sim)
            end
        end
    end

    @info "Physics results written to $output_dir"
end

"""
    compute_onebody_green(state::VMCState{T}) where T

Compute a very simple equal-time one-body Green function snapshot G_{ij}^σ.
Currently uses diagonal occupancy from the electron positions as an approximation
for <c†_{iσ} c_{jσ}> (i==j -> n_{iσ}; i!=j -> 0). Returns (Gup, Gdn).
"""
function compute_onebody_green(state::VMCState{T}) where {T}
    n = state.n_sites
    nup = div(state.n_electrons, 2)
    n_up = zeros(Int, n)
    n_dn = zeros(Int, n)
    for (k, pos) in enumerate(state.electron_positions)
        if k <= nup
            n_up[pos] += 1
        else
            n_dn[pos] += 1
        end
    end
    Gup = zeros(ComplexF64, n, n)
    Gdn = zeros(ComplexF64, n, n)
    for i in 1:n
        Gup[i,i] = n_up[i]
        Gdn[i,i] = n_dn[i]
    end
    return Gup, Gdn
end

"""
    compute_onebody_green_local(sim::VMCSimulation{T}) where T

Compute equal-time one-body Green function using LocalGreenFunction machinery.
This uses the current snapshot electron positions to populate the required
arrays for the placeholder local Green implementation.
"""
function compute_onebody_green_local(sim::VMCSimulation{T}) where {T}
    n = sim.vmc_state.n_sites
    ne = sim.vmc_state.n_electrons
    nspin = 2
    # LocalGreenFunction works with T type for internal arrays; use ComplexF64
    gf = LocalGreenFunction{ComplexF64}(n, ne, nspin)

    # Build ele_idx (indices of electron positions per spin), ele_cfg (occupation per site,spin), ele_num (0/1 per site,spin)
    fill!(gf.ele_idx, 0)
    fill!(gf.ele_cfg, 0)
    fill!(gf.ele_num, 0)
    fill!(gf.proj_cnt, 0)

    nup = div(ne, 2)
    # Fill spin-up electrons
    for k in 1:nup
        pos = sim.vmc_state.electron_positions[k]
        gf.ele_idx[k] = pos
        rsi = pos + (1 - 1) * n
        gf.ele_cfg[rsi] = 1
        gf.ele_num[rsi] = 1
    end
    # Fill spin-down electrons
    for k in 1:(ne - nup)
        pos = sim.vmc_state.electron_positions[nup + k]
        gf.ele_idx[nup + k] = pos
        rsi = pos + (2 - 1) * n
        gf.ele_cfg[rsi] = 1
        gf.ele_num[rsi] = 1
    end

    Gup = zeros(ComplexF64, n, n)
    Gdn = zeros(ComplexF64, n, n)
    # ip is a placeholder parameter for the local green call; set to 1
    ip = one(ComplexF64)
    for i in 1:n, j in 1:n
        # spin up = 1
        Gup[i, j] = green_function_1body!(gf, i, j, 1, ip, gf.ele_idx, gf.ele_cfg, gf.ele_num, gf.proj_cnt)
        # spin down = 2
        Gdn[i, j] = green_function_1body!(gf, i, j, 2, ip, gf.ele_idx, gf.ele_cfg, gf.ele_num, gf.proj_cnt)
    end
    return Gup, Gdn
end

"""
    compute_pair_correlation(state::VMCState{T}; max_distance::Int=5, geometry=nothing) where T

Compute equal-time s-wave onsite pair correlation C_pair(d) ≈ ⟨D_i D_j⟩_d using
snapshot double-occupancy indicator D_i = n_{i↑} n_{i↓}. Geometry-aware distance
binning when geometry is provided; otherwise use 1D |i−j|.
"""
function compute_pair_correlation(
    state::VMCState{T};
    max_distance::Int = 5,
    geometry = nothing,
) where {T}
    n = state.n_sites
    maxd = max(0, min(max_distance, n - 1))

    # Build double-occupancy indicator D_i from positions
    nup = div(state.n_electrons, 2)
    n_up = zeros(Int, n)
    n_dn = zeros(Int, n)
    for (k, pos) in enumerate(state.electron_positions)
        if k <= nup
            n_up[pos] += 1
        else
            n_dn[pos] += 1
        end
    end
    D = n_up .* n_dn

    if geometry === nothing
        corr = ComplexF64[0.0 + 0.0im for _ in 0:maxd]
        for d in 0:maxd
            s = 0.0
            c = 0
            for i in 1:n, j in 1:n
                if abs(i - j) == d
                    s += D[i] * D[j]
                    c += 1
                end
            end
            if c > 0
                corr[d+1] = s / c
            end
        end
        return corr
    else
        coords = generate_site_coordinates(geometry)
        sums = Dict{Float64,Float64}()
        counts = Dict{Float64,Int}()
        for i in 1:n, j in 1:n
            d = round(norm(coords[i, :] .- coords[j, :]), digits=6)
            sums[d] = get(sums, d, 0.0) + D[i] * D[j]
            counts[d] = get(counts, d, 0) + 1
        end
        keys_sorted = sort(collect(keys(counts)))
        nm = min(length(keys_sorted), maxd + 1)
        corr = ComplexF64[]
        for k in 1:nm
            key = keys_sorted[k]
            push!(corr, (sums[key] / counts[key]) + 0.0im)
        end
        return corr
    end
end

"""
    print_simulation_summary(sim::VMCSimulation{T}) where {T}

Print a summary of the simulation configuration and results.
"""
function print_simulation_summary(sim::VMCSimulation{T}) where {T}
    println("="^60)
    println("VMC Simulation Summary")
    println("="^60)
    println("Mode: $(sim.mode)")
    println("System: $(sim.config.model) on $(sim.config.lattice)")
    println("Sites: $(sim.config.nsites), Electrons: $(sim.config.nelec)")
    println("Parameters: $(length(sim.parameters)) total")
    println("  - Projection: $(length(sim.parameters.proj))")
    println("  - RBM: $(length(sim.parameters.rbm))")
    println("  - Slater: $(length(sim.parameters.slater))")
    println("  - OptTrans: $(length(sim.parameters.opttrans))")

    if sim.mode == PARAMETER_OPTIMIZATION && !isempty(sim.optimization_results)
        final_result = sim.optimization_results[end]
        println("Final Energy: $(real(final_result["energy"])) ± $(final_result["energy_error"])")
    elseif sim.mode == PHYSICS_CALCULATION && !isempty(sim.physics_results)
        println("Final Energy: $(real(sim.physics_results["energy_mean"])) ± $(sim.physics_results["energy_std"])")
    end

    println("="^60)
end

"""
    get_reciprocal_grid(geometry)

Return a k-grid based on lattice geometry for chain and square lattices.
Chain: [(2π n/L,) for n=0..L-1]. Square: [(2π nx/Lx, 2π ny/Ly)].
"""
function get_reciprocal_grid(geometry)
    if geometry === nothing
        return Tuple{Float64}[]
    end
    if geometry.dimensions == 1 || (length(geometry.L) == 2 && geometry.L[2] == 1)
        L = geometry.n_sites_total
        ks = Tuple{Float64}[(2π * n / L,) for n in 0:(L-1)]
        return ks
    elseif geometry.dimensions == 2
        Lx, Ly = geometry.L[1], geometry.L[2]
        ks = Tuple{Float64,Float64}[]
        for nx in 0:(Lx-1), ny in 0:(Ly-1)
            push!(ks, (2π * nx / Lx, 2π * ny / Ly))
        end
        return ks
    else
        return Tuple{Float64}[]
    end
end

"""
    compute_structure_factors(state::VMCState{T}; geometry=nothing) where T

Compute equal-time spin and density structure factors S(q) from snapshot
occupations using geometry-aware k-grid when available.
"""
function compute_structure_factors(state::VMCState{T}; geometry=nothing) where {T}
    n = state.n_sites
    # Build occupations and spin-z
    nup = div(state.n_electrons, 2)
    n_up = zeros(Int, n)
    n_dn = zeros(Int, n)
    for (k, pos) in enumerate(state.electron_positions)
        if k <= nup
            n_up[pos] += 1
        else
            n_dn[pos] += 1
        end
    end
    n_tot = n_up .+ n_dn
    s_z = (n_up .- n_dn) ./ 2

    # Coordinates
    coords = geometry === nothing ? nothing : generate_site_coordinates(geometry)
    ks = get_reciprocal_grid(geometry)
    if isempty(ks)
        # Fallback: simple index-based grid
        ks = [(2π * i / n,) for i in 0:(n-1)]
    end

    spin_sf = ComplexF64[]
    dens_sf = ComplexF64[]
    for k in ks
        sum_s = 0.0 + 0.0im
        sum_n = 0.0 + 0.0im
        for i in 1:n, j in 1:n
            if coords === nothing
                # 1D fallback: r = i-1, j-1
                phase = exp(-1im * k[1] * ((i-1) - (j-1)))
            else
                if length(k) == 1
                    phase = exp(-1im * k[1] * (coords[i,1] - coords[j,1]))
                else
                    phase = exp(-1im * (k[1]*(coords[i,1]-coords[j,1]) + k[2]*(coords[i,2]-coords[j,2])))
                end
            end
            sum_s += s_z[i] * s_z[j] * phase
            sum_n += n_tot[i] * n_tot[j] * phase
        end
        push!(spin_sf, sum_s / n)
        push!(dens_sf, sum_n / n)
    end
    return (spin = spin_sf, density = dens_sf, k_grid = ks)
end

"""
    compute_momentum_distribution(state::VMCState{T}; geometry=nothing) where T

Compute a simple momentum distribution n(k) from real-space densities as
the Fourier transform of n_i (snapshot-based, approximate).
"""
function compute_momentum_distribution(state::VMCState{T}; geometry=nothing) where {T}
    n = state.n_sites
    n_tot = zeros(Int, n)
    for pos in state.electron_positions
        n_tot[pos] += 1
    end
    coords = geometry === nothing ? nothing : generate_site_coordinates(geometry)
    ks = get_reciprocal_grid(geometry)
    if isempty(ks)
        ks = [(2π * i / n,) for i in 0:(n-1)]
    end
    nk = ComplexF64[]
    for k in ks
        sum_n = 0.0 + 0.0im
        for i in 1:n
            if coords === nothing
                phase = exp(-1im * k[1] * (i-1))
            else
                if length(k) == 1
                    phase = exp(-1im * k[1] * coords[i,1])
                else
                    phase = exp(-1im * (k[1]*coords[i,1] + k[2]*coords[i,2]))
                end
            end
            sum_n += n_tot[i] * phase
        end
        push!(nk, sum_n / n)
    end
    return nk
end
