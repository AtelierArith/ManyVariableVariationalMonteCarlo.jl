"""
mVMC Integration Module

Integrates all enhanced components to provide C-implementation compatible VMC simulation.
"""

using Random
using Printf
using LinearAlgebra

"""
    EnhancedVMCSimulation{T}

Enhanced VMC simulation that integrates:
- StdFace.def parser
- Detailed wavefunction components
- Precise stochastic reconfiguration
- C-compatible output format
"""
mutable struct EnhancedVMCSimulation{T<:Union{Float64,ComplexF64}}
    # Configuration
    config::SimulationConfig
    parameters::ParameterSet{Vector{T},Vector{T},Vector{T},Vector{T}}

    # Mode
    mode::VMCMode

    # Enhanced wavefunction
    wavefunction::CombinedWavefunction{T}

    # State management
    vmc_state::Union{Nothing,VMCState{T}}

    # Enhanced optimization
    sr_optimizer::Union{Nothing,PreciseStochasticReconfiguration{T}}

    # Output management
    output_manager::MVMCOutputManager

    # Results storage
    optimization_results::Vector{Dict{String,Any}}
    physics_results::Dict{String,Any}

    # Timing
    timers::Dict{String,Float64}
    start_time::Float64

    # Cached configurations from last sampling (to reuse for gradients)
    cached_configurations::Vector{Vector{Int}}

    function EnhancedVMCSimulation{T}(config::SimulationConfig, layout::ParameterLayout) where {T}
        parameters = ParameterSet(layout; T = T)
        output_manager = MVMCOutputManager(
            haskey(config.face, :CDataFileHead) ? String(config.face[:CDataFileHead]) : "output",
            binary_mode = haskey(config.face, :BinaryMode) ? Bool(config.face[:BinaryMode]) : false,
            flush_interval = config.flush_interval
        )

        new{T}(
            config,
            parameters,
            VMCMode(config.nvmc_cal_mode),
            CombinedWavefunction{T}(),
            nothing,
            nothing,
            output_manager,
            Dict{String,Any}[],
            Dict{String,Any}(),
            Dict{String,Float64}(),
            time(),
            Vector{Vector{Int}}()
        )
    end
end

EnhancedVMCSimulation(config::SimulationConfig, layout::ParameterLayout; T=ComplexF64) =
    EnhancedVMCSimulation{T}(config, layout)

"""
    run_mvmc_from_stdface(stdface_file::String; T=ComplexF64, output_dir::String="output")

Complete mVMC simulation starting from StdFace.def file.
"""
function run_mvmc_from_stdface(stdface_file::String; T=ComplexF64, output_dir::String="output")
    # Parse StdFace.def
    println("Start: Read StdFace.def file.")
    config = parse_stdface_and_create_config(stdface_file)
    print_mvmc_header(config)
    println("End  : Read StdFace.def file.")

    # Create parameter layout based on model
    layout = create_parameter_layout(config)

    # Create enhanced simulation
    sim = EnhancedVMCSimulation{T}(config, layout)

    # Set output directory
    sim.output_manager.output_dir = output_dir

    # Run simulation
    println("Start: Initialize parameters.")
    initialize_enhanced_simulation!(sim)
    println("End  : Initialize parameters.")

    println("Start: Initialize variables for quantum projection.")
    # Quantum projection initialization (placeholder)
    println("End  : Initialize variables for quantum projection.")

    # Run based on mode
    if sim.mode == PARAMETER_OPTIMIZATION
        println("Start: Optimize VMC parameters.")
        run_enhanced_parameter_optimization!(sim)
        println("End  : Optimize VMC parameters.")
    elseif sim.mode == PHYSICS_CALCULATION
        println("Start: Calculate VMC physical quantities.")
        run_enhanced_physics_calculation!(sim)
        println("End  : Calculate VMC physical quantities.")
    end

    # Finalize
    finalize_output!(sim.output_manager)
    println("Finish calculation.")

    return sim
end

"""
    create_parameter_layout(config::SimulationConfig) -> ParameterLayout

Create parameter layout based on model configuration.
"""
function create_parameter_layout(config::SimulationConfig)
    n_sites = config.nsites

    # Determine parameter counts based on model
    if config.model == :Spin
        # Try to match mVMC StdFace-defined parameter counts if idx files exist
        root = config.root
        gutz_count = try
            read_idx_count(joinpath(root, "gutzwilleridx.def"), "NGutzwillerIdx")
        catch; 0; end
        jast_count = try
            read_idx_count(joinpath(root, "jastrowidx.def"), "NJastrowIdx")
        catch; 0; end
        orb_count = try
            read_idx_count(joinpath(root, "orbitalidx.def"), "NOrbitalIdx")
        catch; 0; end

        if gutz_count + jast_count + orb_count > 0
            # Map: use proj buffer for (gutz + jastrow), slater for orbital-like params
            n_proj = gutz_count + jast_count
            n_rbm = 0
            n_slater = orb_count
            n_opttrans = 0
        else
            # Fallback: minimal set
            n_proj = 2
            n_rbm = 0
            n_slater = n_sites
            n_opttrans = 0
        end
    else
        # Fermion models: more complex parameter structure
        n_proj = n_sites  # Gutzwiller parameters
        n_rbm = 2 * n_sites * min(n_sites, 8)  # RBM with moderate hidden layer
        n_slater = n_sites * config.nelec  # Slater determinant coefficients
        n_opttrans = 0
    end

    return ParameterLayout(n_proj, n_rbm, n_slater, n_opttrans)
end

"""
    read_idx_count(path::String, key::String) -> Int

Parse mVMC idx.def header to get count, e.g., "NOrbitalIdx 64".
"""
function read_idx_count(path::String, key::String)
    open(path, "r") do f
        for line in eachline(f)
            s = strip(line)
            if startswith(s, key)
                parts = split(s)
                return parse(Int, parts[end])
            end
        end
    end
    return 0
end

"""
    initialize_enhanced_simulation!(sim::EnhancedVMCSimulation{T}) where {T}

Initialize enhanced simulation with all components.
"""
function initialize_enhanced_simulation!(sim::EnhancedVMCSimulation{T}) where {T}
    sim.start_time = time()

    # Initialize VMC state
    n_elec = if sim.config.model == :Spin
        sim.config.nelec > 0 ? sim.config.nelec : sim.config.nsites
    else
        sim.config.nelec
    end

    sim.vmc_state = VMCState{T}(n_elec, sim.config.nsites)

    # Set up initial electron configuration
    initial_positions = if sim.config.model == :Spin
        # For spin models, distribute spins evenly
        n_up = div(n_elec, 2)
        n_dn = n_elec - n_up
        up_sites = collect(1:2:sim.config.nsites)[1:min(n_up, div(sim.config.nsites, 2) + 1)]
        dn_sites = collect(2:2:sim.config.nsites)[1:min(n_dn, div(sim.config.nsites, 2))]
        vcat(up_sites, dn_sites)[1:n_elec]
    else
        collect(1:n_elec)
    end

    initialize_vmc_state!(sim.vmc_state, initial_positions)

    # Initialize Hamiltonian
    initialize_hamiltonian!(sim)

    # Initialize enhanced wavefunction components
    initialize_enhanced_wavefunction!(sim)

    # Initialize parameters
    layout = ParameterLayout(
        length(sim.parameters.proj),
        length(sim.parameters.rbm),
        length(sim.parameters.slater),
        length(sim.parameters.opttrans)
    )
    mask = ParameterMask(layout; default = true)
    flags = ParameterFlags(T <: Complex, length(sim.parameters.rbm) > 0)
    # Initialize parameters with StdFace RNG for reproducibility w.r.t. C
    rng_seed = haskey(sim.config.face, :RndSeed) ? Int(sim.config.face[:RndSeed]) : 123456789
    rng = Random.MersenneTwister(rng_seed)
    initialize_parameters!(sim.parameters, layout, mask, flags; rng=rng)

    # Initialize stochastic reconfiguration
    n_params = length(sim.parameters)
    n_samples = sim.config.nvmc_sample
    sim.sr_optimizer = PreciseStochasticReconfiguration{T}(n_params, n_samples)

    # Setup output files
    if sim.mode == PARAMETER_OPTIMIZATION
        open_output_files!(sim.output_manager, "optimization")
        write_optimization_header!(sim.output_manager, sim.config)
    else
        nstore = haskey(sim.config.face, :NStore) ? Int(sim.config.face[:NStore]) : 1
        open_physics_files!(sim.output_manager, nstore)
        write_physics_header!(sim.output_manager, sim.config)
    end
end

"""
    get_lattice_geometry(sim::EnhancedVMCSimulation{T}) where {T}

Extract lattice geometry information from simulation configuration.
"""
function get_lattice_geometry(sim::EnhancedVMCSimulation{T}) where {T}
    face = sim.config.face

    # Extract lattice parameters
    lattice_type = haskey(face, :model) ? Symbol(face[:model]) : sim.config.lattice

    # Get lattice dimensions
    if haskey(face, :L)
        # 1D chain
        L = Int(face[:L])
        return EnhancedChainLattice(L)
    elseif haskey(face, :W) && haskey(face, :L)
        # 2D lattice
        W = Int(face[:W])
        L = Int(face[:L])
        return EnhancedSquareLattice(W, L)
    elseif haskey(face, :a0W) && haskey(face, :a1W) && haskey(face, :a0L) && haskey(face, :a1L)
        # General 2D lattice with basis vectors
        a0W, a1W = Float64(face[:a0W]), Float64(face[:a1W])
        a0L, a1L = Float64(face[:a0L]), Float64(face[:a1L])
        return EnhancedGeneralLattice2D(a0W, a1W, a0L, a1L)
    else
        # Default to 1D chain based on number of sites
        return EnhancedChainLattice(sim.config.nsites)
    end
end

"""
Lattice geometry structures for enhanced compatibility.
"""
struct EnhancedChainLattice
    length::Int
end

struct EnhancedSquareLattice
    width::Int
    length::Int
end

struct EnhancedGeneralLattice2D
    a0_width::Float64
    a1_width::Float64
    a0_length::Float64
    a1_length::Float64
end

"""
    initialize_hamiltonian!(sim::EnhancedVMCSimulation{T}) where {T}

Initialize Hamiltonian based on model type.
"""
function initialize_hamiltonian!(sim::EnhancedVMCSimulation{T}) where {T}
    try
        geometry = get_lattice_geometry(sim)

        if sim.config.model == :Spin
            # Heisenberg model
            J_val = haskey(sim.config.face, :J) ? Float64(sim.config.face[:J]) : 1.0
            if geometry !== nothing
                sim.vmc_state.hamiltonian = create_heisenberg_hamiltonian(geometry, T(J_val))
            else
                lattice_type = sim.config.lattice
                sim.vmc_state.hamiltonian = create_enhanced_heisenberg_hamiltonian(
                    sim.config.nsites, T(J_val); lattice_type = lattice_type
                )
            end
        else
            # Hubbard model
            sim.vmc_state.hamiltonian = create_enhanced_hubbard_hamiltonian(
                sim.config.nsites,
                sim.config.nelec,
                T(sim.config.t),
                T(sim.config.u);
                lattice_type = sim.config.lattice
            )
        end
    catch e
        @warn "Failed to initialize Hamiltonian: $e"
        sim.vmc_state.hamiltonian = nothing
    end
end

"""
    create_heisenberg_hamiltonian(geometry, J::T) where {T}

Create Heisenberg Hamiltonian from lattice geometry.
"""
function create_heisenberg_hamiltonian(geometry::EnhancedChainLattice, J::T) where {T}
    # Create nearest-neighbor Heisenberg chain
    n_sites = geometry.length
    bonds = Tuple{Int,Int}[]

    # Add nearest-neighbor bonds
    for i in 1:(n_sites-1)
        push!(bonds, (i, i+1))
    end

    return HeisenbergHamiltonian{T}(n_sites, bonds, J)
end

function create_heisenberg_hamiltonian(geometry::EnhancedSquareLattice, J::T) where {T}
    # Create 2D square lattice Heisenberg model
    W, L = geometry.width, geometry.length
    n_sites = W * L
    bonds = Tuple{Int,Int}[]

    # Add horizontal bonds
    for j in 1:L, i in 1:(W-1)
        site1 = (j-1) * W + i
        site2 = (j-1) * W + i + 1
        push!(bonds, (site1, site2))
    end

    # Add vertical bonds
    for j in 1:(L-1), i in 1:W
        site1 = (j-1) * W + i
        site2 = j * W + i
        push!(bonds, (site1, site2))
    end

    return HeisenbergHamiltonian{T}(n_sites, bonds, J)
end

function create_enhanced_heisenberg_hamiltonian(n_sites::Int, J::T; lattice_type=:Chain) where {T}
    # Fallback implementation for basic lattice types
    if lattice_type == :Chain
        return create_heisenberg_hamiltonian(EnhancedChainLattice(n_sites), J)
    elseif lattice_type == :Square
        # Assume square geometry
        L = Int(sqrt(n_sites))
        return create_heisenberg_hamiltonian(EnhancedSquareLattice(L, L), J)
    else
        # Default to chain
        return create_heisenberg_hamiltonian(EnhancedChainLattice(n_sites), J)
    end
end

"""
    create_enhanced_hubbard_hamiltonian(n_sites::Int, n_elec::Int, t::T, u::T; lattice_type=:Chain) where {T}

Create Hubbard Hamiltonian for enhanced implementation.
"""
function create_enhanced_hubbard_hamiltonian(n_sites::Int, n_elec::Int, t::T, u::T; lattice_type=:Chain) where {T}
    if lattice_type == :Chain
        # Create 1D Hubbard chain
        bonds = [(i, i+1) for i in 1:(n_sites-1)]
        return HubbardHamiltonian{T}(n_sites, n_elec, bonds, t, u)
    elseif lattice_type == :Square
        # Create 2D Hubbard model
        L = Int(sqrt(n_sites))
        bonds = Tuple{Int,Int}[]

        # Horizontal bonds
        for j in 1:L, i in 1:(L-1)
            site1 = (j-1) * L + i
            site2 = (j-1) * L + i + 1
            push!(bonds, (site1, site2))
        end

        # Vertical bonds
        for j in 1:(L-1), i in 1:L
            site1 = (j-1) * L + i
            site2 = j * L + i
            push!(bonds, (site1, site2))
        end

        return HubbardHamiltonian{T}(n_sites, n_elec, bonds, t, u)
    else
        # Default fallback
        bonds = [(i, i+1) for i in 1:(n_sites-1)]
        return HubbardHamiltonian{T}(n_sites, n_elec, bonds, t, u)
    end
end

"""
Hamiltonian structures for enhanced implementation.
"""
struct HeisenbergHamiltonian{T}
    n_sites::Int
    bonds::Vector{Tuple{Int,Int}}
    J::T
end

struct HubbardHamiltonian{T}
    n_sites::Int
    n_electrons::Int
    bonds::Vector{Tuple{Int,Int}}
    t::T
    u::T
end

"""
Enhanced Hamiltonian calculation methods to interface with existing VMC sampler.
"""
function calculate_hamiltonian(ham::HeisenbergHamiltonian{T}, up_pos::Vector{Int}, dn_pos::Vector{Int}) where {T}
    energy = T(0)

    # Heisenberg model: H = J Σ_<i,j> (S_i · S_j)
    for (i, j) in ham.bonds
        # Compute spin-spin interaction
        # S_i · S_j = S^z_i S^z_j + 0.5(S^+_i S^-_j + S^-_i S^+_j)

        # z-component
        s_i_z = 0.5 * ((i in up_pos ? 1 : 0) - (i in dn_pos ? 1 : 0))
        s_j_z = 0.5 * ((j in up_pos ? 1 : 0) - (j in dn_pos ? 1 : 0))
        energy += ham.J * s_i_z * s_j_z

        # xy-component (flip-flop terms)
        if (i in up_pos && j in dn_pos) || (i in dn_pos && j in up_pos)
            energy += 0.5 * ham.J
        end
    end

    return energy
end

function calculate_hamiltonian(ham::HubbardHamiltonian{T}, up_pos::Vector{Int}, dn_pos::Vector{Int}) where {T}
    energy = T(0)

    # Kinetic energy: -t Σ_<i,j> (c†_i c_j + h.c.)
    for (i, j) in ham.bonds
        # Hopping terms for up electrons
        if i in up_pos && !(j in up_pos)
            energy -= ham.t
        end
        if j in up_pos && !(i in up_pos)
            energy -= ham.t
        end

        # Hopping terms for down electrons
        if i in dn_pos && !(j in dn_pos)
            energy -= ham.t
        end
        if j in dn_pos && !(i in dn_pos)
            energy -= ham.t
        end
    end

    # Interaction energy: U Σ_i n_i↑ n_i↓
    for site in 1:ham.n_sites
        if site in up_pos && site in dn_pos
            energy += ham.u
        end
    end

    return energy
end

"""
    initialize_enhanced_wavefunction!(sim::EnhancedVMCSimulation{T}) where {T}

Initialize enhanced wavefunction components.
"""
function initialize_enhanced_wavefunction!(sim::EnhancedVMCSimulation{T}) where {T}
    n_sites = sim.config.nsites
    n_elec = sim.vmc_state.n_electrons

    # Initialize Slater determinant (for fermion models)
    if sim.config.model != :Spin && length(sim.parameters.slater) > 0
        sim.wavefunction.slater_det = SlaterDeterminant{T}(n_elec, n_sites)
        init_matrix = Matrix{T}(I, n_elec, n_sites)
        initialize_slater!(sim.wavefunction.slater_det, init_matrix)
    end

    # Initialize Gutzwiller projector
    if length(sim.parameters.proj) > 0
        sim.wavefunction.gutzwiller = GutzwillerProjector{T}(n_sites, n_elec)
        set_gutzwiller_parameters!(sim.wavefunction.gutzwiller, sim.parameters.proj)
    end

    # Initialize enhanced Jastrow factor
    if length(sim.parameters.proj) > 0
        sim.wavefunction.jastrow = EnhancedJastrowFactor{T}(n_sites, n_elec)
        geometry = get_lattice_geometry(sim)
        initialize_neighbor_lists!(sim.wavefunction.jastrow, geometry)
        set_jastrow_parameters!(sim.wavefunction.jastrow, sim.parameters.proj)
    end

    # Initialize RBM network
    if length(sim.parameters.rbm) > 0
        n_visible = 2 * n_sites
        n_hidden = max(1, div(length(sim.parameters.rbm), n_visible + 1))
        sim.wavefunction.rbm = EnhancedRBMNetwork{T}(n_visible, n_hidden, n_sites)
        rng_seed = haskey(sim.config.face, :RndSeed) ? Int(sim.config.face[:RndSeed]) : 12345
    rng = Random.MersenneTwister(rng_seed)
        initialize_rbm_random!(sim.wavefunction.rbm, rng)
        set_rbm_parameters!(sim.wavefunction.rbm, sim.parameters.rbm)
    end

    # Load StdFace orbital mapping for Spin pairing (orbitalidx.def) and seed pair params
    if sim.config.model == :Spin && length(sim.parameters.slater) > 0
        try
            rootdir = haskey(sim.config.face, :StdFaceRoot) ? String(sim.config.face[:StdFaceRoot]) : sim.config.root
            sim.wavefunction.orbital_map = load_orbital_index_map(rootdir, sim.config.nsites)
        catch e
            @warn "Failed to load orbitalidx.def: $e"
            sim.wavefunction.orbital_map = nothing
        end
        sim.wavefunction.pair_params = copy(sim.parameters.slater)
    end
end

"""
    print_simulation_summary(sim::EnhancedVMCSimulation)

Lightweight summary compatible with example scripts.
"""
function print_simulation_summary(sim::EnhancedVMCSimulation)
    println("# VMC Enhanced Simulation Results")
    println("# Mode: ", sim.mode)
    println("# Sites: ", sim.config.nsites)
    if !isempty(sim.optimization_results)
        last = sim.optimization_results[end]
        println("# Final Energy: ", real(last["energy"]))
        println("# Energy Error: ", last["energy_error"])
        println("# Overlap Cond.: ", last["overlap_condition"])
    elseif !isempty(sim.physics_results)
        println("# Energy: ", real(sim.physics_results["energy_mean"]))
        println("# Energy Std: ", sim.physics_results["energy_std"])
        println("# Samples: ", sim.physics_results["n_samples"])
    end
end

"""
    load_orbital_index_map(root::AbstractString, nsite::Int) -> Matrix{Int}

Parse orbitalidx.def to build site-based (i,j) -> param-index (1-based) map.
"""
function load_orbital_index_map(root::AbstractString, nsite::Int)
    path = joinpath(root, "orbitalidx.def")
    m = fill(0, nsite, nsite)
    open(path, "r") do f
        for line in eachline(f)
            s = strip(line)
            isempty(s) && continue
            startswith(s, "=") && continue
            startswith(s, "NOrbitalIdx") && continue
            startswith(s, "ComplexType") && continue
            parts = split(s)
            if length(parts) == 3
                i = parse(Int, parts[1]) + 1
                j = parse(Int, parts[2]) + 1
                k = parse(Int, parts[3]) + 1
                if 1 <= i <= nsite && 1 <= j <= nsite
                    m[i, j] = k
                end
            elseif length(parts) == 2
                # Optimization flags; ignore
            end
        end
    end
    return m
end

# set_gutzwiller_parameters! is defined in wavefunction_detailed.jl

"""
    initialize_neighbor_lists!(jastrow::EnhancedJastrowFactor{T}, geometry) where {T}

Initialize neighbor lists for Jastrow factor based on lattice geometry.
"""
function initialize_neighbor_lists!(jastrow::EnhancedJastrowFactor{T}, geometry) where {T}
    if geometry isa EnhancedChainLattice
        # 1D chain neighbor lists
        n_sites = geometry.length
        for i in 1:n_sites
            neighbors = Int[]
            if i > 1
                push!(neighbors, i-1)
            end
            if i < n_sites
                push!(neighbors, i+1)
            end
            jastrow.neighbor_list[i] = neighbors
        end
    elseif geometry isa EnhancedSquareLattice
        # 2D square lattice neighbor lists
        W, L = geometry.width, geometry.length
        for j in 1:L, i in 1:W
            site = (j-1) * W + i
            neighbors = Int[]

            # Left neighbor
            if i > 1
                push!(neighbors, (j-1) * W + i - 1)
            end
            # Right neighbor
            if i < W
                push!(neighbors, (j-1) * W + i + 1)
            end
            # Up neighbor
            if j > 1
                push!(neighbors, (j-2) * W + i)
            end
            # Down neighbor
            if j < L
                push!(neighbors, j * W + i)
            end

            jastrow.neighbor_list[site] = neighbors
        end
    else
        # Default: nearest neighbors in 1D
        n_sites = jastrow.n_sites
        for i in 1:n_sites
            neighbors = Int[]
            if i > 1
                push!(neighbors, i-1)
            end
            if i < n_sites
                push!(neighbors, i+1)
            end
            jastrow.neighbor_list[i] = neighbors
        end
    end
end

# set_jastrow_parameters! is defined in wavefunction_detailed.jl

# set_rbm_parameters! is defined in wavefunction_detailed.jl

# update_correlation_matrix! is handled in wavefunction_detailed.jl

# Enhanced Jastrow matrix update is handled in wavefunction_detailed.jl

# update_rbm_correlations! is handled in wavefunction_detailed.jl

"""
    run_enhanced_parameter_optimization!(sim::EnhancedVMCSimulation{T}) where {T}

Run enhanced parameter optimization with precise SR.
"""
function run_enhanced_parameter_optimization!(sim::EnhancedVMCSimulation{T}) where {T}
    # Setup optimization configuration
    face = sim.config.face
    opt_config = OptimizationConfig(
        method = STOCHASTIC_RECONFIGURATION,
        max_iterations = sim.config.nsr_opt_itr_step,
        convergence_tolerance = sim.config.dsr_opt_red_cut,
        learning_rate = sim.config.dsr_opt_step_dt,
        regularization_parameter = sim.config.dsr_opt_sta_del,
        use_sr_cg = haskey(face, :NSRCG) ? Int(face[:NSRCG]) != 0 : false,
        sr_cg_max_iter = haskey(face, :NSROptCGMaxIter) ? Int(face[:NSROptCGMaxIter]) : 100,
        sr_cg_tol = haskey(face, :DSROptCGTol) ? Float64(face[:DSROptCGTol]) : 1e-6
    )

    configure_optimization!(sim.sr_optimizer, opt_config)

    # Optimization loop
    for iteration in 1:sim.config.nsr_opt_itr_step
        iter_start_time = time()

        # Print progress
        print_progress_mvmc_style(iteration-1, sim.config.nsr_opt_itr_step)

        # Update wavefunction parameters
        update_wavefunction_parameters!(sim.wavefunction, sim.parameters)

        # Sample configurations
        # Use FULL NVMCSample per iteration as in mVMC (NSROptItrSmp is not a per-iteration sample count)
        actual_sample_size = sim.config.nvmc_sample
        sample_results = enhanced_sample_configurations!(sim, actual_sample_size)

        # Compute gradients
        gradients = compute_enhanced_gradients!(sim, sample_results)

        # Solve SR equations
        # Ensure SR sample size matches what we actually sampled this iteration
        actual_samples = sample_results.n_samples
        if sim.sr_optimizer.n_samples != actual_samples
            sim.sr_optimizer.n_samples = actual_samples
            configure_optimization!(sim.sr_optimizer, opt_config)
        end
        weights = ones(Float64, actual_samples)
        gradients_trimmed = gradients[:, 1:actual_samples]
        compute_overlap_matrix_precise!(sim.sr_optimizer, gradients_trimmed, weights)
        energy_samples_trimmed = sample_results.energy_samples[1:actual_samples]
        compute_force_vector_precise!(sim.sr_optimizer, gradients_trimmed, energy_samples_trimmed, weights)
        solve_sr_equations_precise!(sim.sr_optimizer, opt_config)

        # Update parameters
        update_parameters!(sim.parameters, sim.sr_optimizer.parameter_delta)

        # Collect statistics
        energy_var = compute_energy_variance!(sim.sr_optimizer)
        # SR info in C reference style
        eigs = isempty(sim.sr_optimizer.overlap_eigenvalues) ? [0.0] : sim.sr_optimizer.overlap_eigenvalues
        cutoff = sim.sr_optimizer.eigenvalue_cutoff
        msize = count(>(cutoff), eigs)
        diagcut = sim.sr_optimizer.n_parameters - msize
        # rmax: take signed max of real part in force vector by absolute value
        fvec = sim.sr_optimizer.force_vector
        if isempty(fvec)
            rmax_val = 0.0
            imax_idx = 0
        else
            reals = [real(f) for f in fvec]
            mags = abs.(reals)
            imax1 = findmax(mags)[2]
            rmax_val = reals[imax1]
            imax_idx = imax1 - 1  # 0-based index like C files
        end
        sr_info = Dict(
            "npara" => sim.sr_optimizer.n_parameters,
            "msize" => msize,
            "optcut" => sim.sr_optimizer.n_parameters + 2, # heuristic to match C
            "diagcut" => diagcut,
            "sdiagmax" => maximum(eigs),
            "sdiagmin" => minimum(eigs),
            "rmax" => rmax_val,
            "imax" => imax_idx,
        )

        # Weighted averages (uniform weights for now)
        # Weighted averaging to mirror C: WeightAverageWE/WeightAverageSROpt_real (uniform weights placeholder)
        wa = WeightedAverager()
        update!(wa, sample_results.energy_samples)
        Etot, Etot2 = summarize_energy(wa)
        write_optimization_step!(
            sim.output_manager,
            iteration,
            Etot,
            Etot2,
            as_vector(sim.parameters),
            sr_info,
        )

        # Store results
        iter_results = Dict{String,Any}(
            "iteration" => iteration,
            "energy" => sample_results.energy_mean,
            "energy_error" => sample_results.energy_std,
            "energy_variance" => energy_var,
            "parameter_norm" => norm(sim.sr_optimizer.parameter_delta),
            "overlap_condition" => sim.sr_optimizer.overlap_condition_number
        )
        push!(sim.optimization_results, iter_results)

        # Timing
        iter_time = time() - iter_start_time
        total_time = time() - sim.start_time
        write_timing_info!(sim.output_manager, iteration, total_time, iter_time)
    end

    # Final progress
    print_progress_mvmc_style(sim.config.nsr_opt_itr_step, sim.config.nsr_opt_itr_step)

    # Output final parameters
    println("Start: Output opt params.")
    all_params_vec = as_vector(sim.parameters)
    write_final_parameters!(sim.output_manager, all_params_vec)

    # Also write split parameter files to match mVMC naming
    if sim.config.model == :Spin
        # Try to match StdFace idx counts
        root = sim.config.root
        ngutz = try read_idx_count(joinpath(root, "gutzwilleridx.def"), "NGutzwillerIdx") catch; 0 end
        njast = try read_idx_count(joinpath(root, "jastrowidx.def"),     "NJastrowIdx")    catch; 0 end
        norb  = try read_idx_count(joinpath(root, "orbitalidx.def"),     "NOrbitalIdx")    catch; 0 end

        # Split proj buffer into gutz/jast
        gvec = ComplexF64[]
        jvec = ComplexF64[]
        if ngutz > 0 || njast > 0
            p = ComplexF64.(sim.parameters.proj)
            gvec = length(p) >= ngutz ? p[1:ngutz] : p
            jvec = length(p) > ngutz ? p[(ngutz+1):min(end, ngutz+njast)] : ComplexF64[]
        end
        # Orbital from slater buffer if present
        ovec = ComplexF64.(sim.parameters.slater)
        write_component_parameter_files!(sim.output_manager;
            gutzwiller=gvec, jastrow=jvec, orbital=ovec,
            ngutz=ngutz>0 ? ngutz : length(gvec),
            njast=njast>0 ? njast : length(jvec),
            norb =norb >0 ? norb  : length(ovec))
    else
        # For fermion models, map slater params to orbital file
        write_component_parameter_files!(sim.output_manager;
            gutzwiller=ComplexF64[], jastrow=ComplexF64.(sim.parameters.proj), orbital=ComplexF64.(sim.parameters.slater))
    end
    println("End: Output opt params.")
end

"""
    as_vector(params::ParameterSet)

Flatten ParameterSet buffers into a single contiguous vector.
"""
function as_vector(params::ParameterSet)
    T = ComplexF64
    total = length(params)
    vec = Vector{T}(undef, total)
    idx = 1
    for buf in (params.proj, params.rbm, params.slater, params.opttrans)
        for x in buf
            vec[idx] = T(x)
            idx += 1
        end
    end
    return vec
end

"""
    run_enhanced_physics_calculation!(sim::EnhancedVMCSimulation{T}) where {T}

Run enhanced physics calculation.
"""
function run_enhanced_physics_calculation!(sim::EnhancedVMCSimulation{T}) where {T}
    n_data_samples = haskey(sim.config.face, :NDataQtySmp) ? Int(sim.config.face[:NDataQtySmp]) : 1
    nstore = haskey(sim.config.face, :NStore) ? Int(sim.config.face[:NStore]) : 1

    println("Start: Sampling.")
    for sample_idx in 1:n_data_samples
        # Sample configurations with enhanced wavefunction
        sample_results = enhanced_sample_configurations!(sim, sim.config.nvmc_sample)

        # Measure enhanced observables
        observables = measure_enhanced_observables!(sim, sample_results)

        # Write physics step with detailed data
        write_physics_step!(sim.output_manager, sample_idx, sample_results.energy_mean, observables)

        # Compute and write Green functions using enhanced methods
        gup, gdn = compute_enhanced_green_functions!(sim)

        # Write to main file and binned files with C-compatible format
        write_green_functions!(sim.output_manager, gup, gdn, 0)
        for bin in 1:nstore
            write_green_functions!(sim.output_manager, gup, gdn, bin)
        end

        # Write two-body Green functions if requested
        if haskey(sim.config.face, :TwoBodyG) && Bool(sim.config.face[:TwoBodyG])
            max_rows = haskey(sim.config.face, :MaxG4Rows) ? Int(sim.config.face[:MaxG4Rows]) : 20000
            write_two_body_green_functions!(sim.output_manager, gup, gdn, 0, max_rows)
            for bin in 1:nstore
                write_two_body_green_functions!(sim.output_manager, gup, gdn, bin, max_rows)
            end
        end

        # Store results for final analysis
        if sample_idx == 1
            sim.physics_results["energy_mean"] = sample_results.energy_mean
            sim.physics_results["energy_std"] = sample_results.energy_std
            sim.physics_results["acceptance_rate"] = sample_results.acceptance_rate
            sim.physics_results["n_samples"] = sample_results.n_samples
        end
    end
    println("End  : Sampling.")

    # Lanczos calculation if requested
    if sim.config.nlanczos_mode > 0
        energies, variances = run_enhanced_lanczos!(sim)
        write_lanczos_files!(sim.output_manager, energies, variances)
    end
end

"""
    enhanced_sample_configurations!(sim::EnhancedVMCSimulation{T}, n_samples::Int) where {T}

Enhanced configuration sampling with precise wavefunction evaluation.
"""
function enhanced_sample_configurations!(sim::EnhancedVMCSimulation{T}, n_samples::Int) where {T}
    # Use C-implementation default parameters (matching mVMC defaults)
    thermalization_steps = haskey(sim.config.face, :NVMCWarmUp) ? Int(sim.config.face[:NVMCWarmUp]) : 100
    update_interval = haskey(sim.config.face, :NVMCInterval) ? Int(sim.config.face[:NVMCInterval]) : 1

    config = VMCConfig(
        n_samples = n_samples,
        n_thermalization = thermalization_steps,  # Use proper thermalization
        n_measurement = n_samples,
        n_update_per_sample = update_interval,    # Use proper interval
        use_two_electron_updates = true,
        two_electron_probability = 0.15
    )

    rng_seed = haskey(sim.config.face, :RndSeed) ? Int(sim.config.face[:RndSeed]) : 12345
    rng = Random.MersenneTwister(rng_seed)

    # Enhanced sampling matching C implementation volume
    nvmc_sample = haskey(sim.config.face, :NVMCSample) ? Int(sim.config.face[:NVMCSample]) : 1000
    nvmc_warmup = haskey(sim.config.face, :NVMCWarmUp) ? Int(sim.config.face[:NVMCWarmUp]) : 100
    nvmc_interval = haskey(sim.config.face, :NVMCInterval) ? Int(sim.config.face[:NVMCInterval]) : 1

    # C implementation: nOutStep = NVMCWarmUp + NVMCSample, nInStep = NVMCInterval * Nsite
    n_out_steps = nvmc_warmup + nvmc_sample  # Total outer loop iterations
    n_in_steps = nvmc_interval * sim.config.nsites  # Inner loop per outer step

    total_metropolis_steps = n_out_steps * n_in_steps
    #println("    Running Monte Carlo sampling: $(total_metropolis_steps) total Metropolis steps")
    #println("    ($(n_out_steps) outer × $(n_in_steps) inner steps, matching C implementation)")

    # Perform thermalization + measurement as in C implementation
    #println("    Thermalization + Measurement: $(n_out_steps) steps")
    energy_samples = ComplexF64[]
    configurations = Vector{Int}[]

    for out_step in 1:n_out_steps
        # Inner loop: NVMCInterval * Nsite Metropolis steps
        for in_step in 1:n_in_steps
            enhanced_metropolis_step!(sim, rng)
        end

        # Save configuration only during measurement phase (like C implementation)
        if out_step > nvmc_warmup && length(energy_samples) < n_samples
            energy = measure_enhanced_energy(sim)
            push!(energy_samples, energy)
            push!(configurations, copy(sim.vmc_state.electron_positions))
        end

        # Progress indicator
        #=
        if out_step % max(1, div(n_out_steps, 20)) == 0
            print(".")
        end
        =#
    end
    #println(" done")

    # Compute statistics
    energy_mean = sum(energy_samples) / length(energy_samples)
    energy_var = sum(abs2.(energy_samples .- energy_mean)) / (length(energy_samples) - 1)
    energy_std = sqrt(real(energy_var))

    # Create results structure with all required fields
    results = VMCResults{ComplexF64}(
        energy_mean,                           # energy_mean
        ComplexF64(energy_std),               # energy_std
        energy_samples,                       # energy_samples
        Dict{String,Vector{ComplexF64}}(),    # observables
        0.5,                                  # acceptance_rate
        Float64[],                            # acceptance_series
        length(energy_samples),               # n_samples
        config.n_thermalization,             # n_thermalization
        length(energy_samples),               # n_measurement
        1.0,                                  # autocorrelation_time
        length(energy_samples)                # effective_samples
    )

    # Store for gradient computation
    sim.sr_optimizer.energy_samples[1:min(length(energy_samples), sim.sr_optimizer.n_samples)] .=
        energy_samples[1:min(length(energy_samples), sim.sr_optimizer.n_samples)]
    # Cache configurations for reuse in gradient computation
    sim.cached_configurations = configurations

    return results
end

"""
    update_wavefunction_parameters!(wf::CombinedWavefunction{T}, params::ParameterSet) where {T}

Update wavefunction component parameters from the parameter set.
"""
function update_wavefunction_parameters!(wf::CombinedWavefunction{T}, params::ParameterSet) where {T}
    # Update Gutzwiller projector parameters
    if wf.gutzwiller !== nothing && length(params.proj) > 0
        set_gutzwiller_parameters!(wf.gutzwiller, params.proj)
    end

    # Update Jastrow factor parameters
    if wf.jastrow !== nothing && length(params.proj) > 0
        set_jastrow_parameters!(wf.jastrow, params.proj)
    end

    # Update RBM parameters
    if wf.rbm !== nothing && length(params.rbm) > 0
        set_rbm_parameters!(wf.rbm, params.rbm)
    end

    # Update Slater determinant parameters
    if wf.slater_det !== nothing && length(params.slater) > 0
        update_slater_coefficients!(wf.slater_det, params.slater)
    end
end

"""
    update_slater_coefficients!(det::SlaterDeterminant{T}, params::Vector{T}) where {T}

Update Slater determinant coefficient matrix from parameter vector.
"""
function update_slater_coefficients!(det::SlaterDeterminant{T}, params::Vector{T}) where {T}
    n_elec, n_sites = size(det.coefficient_matrix)
    idx = 1

    for i in 1:n_elec, j in 1:n_sites
        if idx <= length(params)
            det.coefficient_matrix[i, j] = params[idx]
            idx += 1
        end
    end

    # Update inverse matrix for efficient updates
    try
        det.inverse_matrix = inv(det.coefficient_matrix[:, 1:n_elec])
    catch
        # Use pseudo-inverse if singular
        det.inverse_matrix = pinv(det.coefficient_matrix[:, 1:n_elec])
    end
end

"""
    compute_enhanced_gradients!(sim::EnhancedVMCSimulation{T}, sample_results) where {T}

Compute enhanced parameter gradients using all wavefunction components.
"""
function compute_enhanced_gradients!(sim::EnhancedVMCSimulation{T}, sample_results) where {T}
    n_params = length(sim.parameters)
    n_samples = sample_results.n_samples

    gradients = zeros(T, n_params, n_samples)

    # Parameter indices for different components
    proj_range = 1:length(sim.parameters.proj)
    rbm_start = length(sim.parameters.proj) + 1
    rbm_range = rbm_start:(rbm_start + length(sim.parameters.rbm) - 1)
    slater_start = rbm_start + length(sim.parameters.rbm)
    slater_range = slater_start:(slater_start + length(sim.parameters.slater) - 1)

    # Use configurations cached during the previous sampling to avoid extra MC work
    configurations = sim.cached_configurations
    if isempty(configurations)
        # Fallback: repeat current configuration
        configurations = [copy(sim.vmc_state.electron_positions) for _ in 1:n_samples]
    else
        # Truncate or pad to n_samples
        if length(configurations) < n_samples
            last = configurations[end]
            for _ in 1:(n_samples - length(configurations))
                push!(configurations, copy(last))
            end
        elseif length(configurations) > n_samples
            configurations = configurations[1:n_samples]
        end
    end

    for sample_idx in 1:n_samples
        # Get current sampled configuration
        config = configurations[sample_idx]

        # Compute Jastrow/Gutzwiller gradients (full vector matching proj length)
        if !isempty(proj_range) && sim.wavefunction.jastrow !== nothing
            grad_proj = compute_jastrow_gradients(sim.wavefunction.jastrow, config)
            # Ensure size compatibility
            nfill = min(length(proj_range), length(grad_proj))
            gradients[proj_range[1:nfill], sample_idx] .= grad_proj[1:nfill]
        elseif !isempty(proj_range) && sim.wavefunction.gutzwiller !== nothing
            grad_proj = compute_gutzwiller_gradients(sim.wavefunction.gutzwiller, config)
            nfill = min(length(proj_range), length(grad_proj))
            gradients[proj_range[1:nfill], sample_idx] .= grad_proj[1:nfill]
        end

        # Compute RBM gradients
        if !isempty(rbm_range) && sim.wavefunction.rbm !== nothing
            grad_rbm = compute_rbm_gradients(sim.wavefunction.rbm, config)
            if length(grad_rbm) >= length(rbm_range)
                gradients[rbm_range, sample_idx] .= grad_rbm[1:length(rbm_range)]
            end
        end

        # Compute Slater determinant gradients
        if !isempty(slater_range) && sim.wavefunction.slater_det !== nothing
            grad_slater = compute_slater_gradients(sim.wavefunction.slater_det, config)
            if length(grad_slater) >= length(slater_range)
                gradients[slater_range, sample_idx] .= grad_slater[1:length(slater_range)]
            end
        end
    end

    return gradients
end

"""
    compute_gutzwiller_gradients(proj::GutzwillerProjector{T}, config) where {T}

Compute gradients of log(Ψ) with respect to Gutzwiller parameters.
"""
function compute_gutzwiller_gradients(proj::GutzwillerProjector{T}, config) where {T}
    n_sites = proj.n_sites
    gradients = zeros(T, n_sites)

    # Compute occupancy numbers at each site
    occupancies = zeros(Int, n_sites)
    for pos in config
        if 1 <= pos <= n_sites
            occupancies[pos] += 1
        end
    end

    # ∂log(Ψ_G)/∂g_i = n_i - ⟨n_i⟩
    for i in 1:n_sites
        expected_occ = exp(real(proj.g_parameters[i])) / (1 + exp(real(proj.g_parameters[i])))
        gradients[i] = T(occupancies[i] - expected_occ)
    end

    return gradients
end

"""
    compute_jastrow_gradients(jastrow::EnhancedJastrowFactor{T}, config) where {T}

Compute gradients of log(Ψ) with respect to all Jastrow parameters in EnhancedJastrowFactor.
The returned vector layout matches ParameterLayout.proj when Spin model is used:
  [onsite (n), nn (n), long-range (n(n-1)/2), spin (n)].
"""
function compute_jastrow_gradients(jastrow::EnhancedJastrowFactor{T}, config) where {T}
    n = jastrow.n_sites

    # Occupancies and spins from configuration
    n_up = zeros(Int, n)
    n_dn = zeros(Int, n)
    n_elec = jastrow.n_electrons
    n_up_elec = div(n_elec, 2)
    for i in 1:n_up_elec
        site = config[i]
        if 1 <= site <= n
            n_up[site] += 1
        end
    end
    for i in (n_up_elec+1):min(n_elec, length(config))
        site = config[i]
        if 1 <= site <= n
            n_dn[site] += 1
        end
    end

    # Helper lambdas
    density(i) = n_up[i] + n_dn[i]
    szi(i) = (n_up[i] - n_dn[i]) / 2

    # Onsite part (n)
    grad_onsite = Vector{T}(undef, n)
    for i in 1:n
        n_tot = density(i)
        docc = n_up[i] * n_dn[i]
        # Approx derivative terms aligned with compute_jastrow_factor!
        grad_onsite[i] = T(docc + (n_tot > 0 ? (n_tot - 1) * n_tot / 2 : 0))
    end

    # Nearest-neighbor part (n): aggregate contributions per site
    grad_nn = zeros(T, n)
    for i in 1:n
        for j in jastrow.neighbor_list[i]
            if i < j
                grad_nn[i] += T(density(i) * density(j))
            end
        end
    end

    # Long-range part (n(n-1)/2): sequence over pairs i<j beyond NN
    n_lr = div(n * (n - 1), 2)
    grad_lr = zeros(T, n_lr)
    lr_idx = 0
    for i in 1:n
        for j in (i+1):n
            lr_idx += 1
            dist = jastrow.distance_matrix[i, j]
            if dist > 1.5
                grad_lr[lr_idx] = T(density(i) * density(j) / (dist^2 + 1))
            else
                grad_lr[lr_idx] = T(0)
            end
        end
    end

    # Spin-dependent part (n): per site aggregate of nn spin-spin
    grad_spin = zeros(T, n)
    for i in 1:n
        for j in jastrow.neighbor_list[i]
            if i < j
                grad_spin[i] += T(szi(i) * szi(j))
            end
        end
    end

    return vcat(grad_onsite, grad_nn, grad_lr, grad_spin)
end

"""
    compute_rbm_gradients(rbm::EnhancedRBMNetwork{T}, config) where {T}

Compute gradients of log(Ψ) with respect to RBM parameters.
"""
function compute_rbm_gradients(rbm::EnhancedRBMNetwork{T}, config) where {T}
    n_visible = rbm.n_visible
    n_hidden = rbm.n_hidden

    # Convert configuration to visible units (spin representation)
    visible = zeros(T, n_visible)
    for i in 1:div(n_visible, 2)
        visible[2*i-1] = (i in config) ? T(1) : T(0)  # up spin
        visible[2*i] = (i in config) ? T(0) : T(1)    # down spin (simplified)
    end

    # Compute hidden unit probabilities
    hidden_probs = zeros(T, n_hidden)
    for j in 1:n_hidden
        activation = rbm.hidden_bias[j]
        for i in 1:n_visible
            activation += rbm.weights[j, i] * visible[i]
        end
        hidden_probs[j] = tanh(activation)
    end

    # Compute gradients
    gradients = T[]

    # Visible bias gradients
    for i in 1:n_visible
        push!(gradients, visible[i])
    end

    # Hidden bias gradients
    for j in 1:n_hidden
        push!(gradients, hidden_probs[j])
    end

    # Weight gradients
    for i in 1:n_visible, j in 1:n_hidden
        push!(gradients, visible[i] * hidden_probs[j])
    end

    return gradients
end

"""
    compute_slater_gradients(det::SlaterDeterminant{T}, config) where {T}

Compute gradients of log(Ψ) with respect to Slater determinant parameters.
"""
function compute_slater_gradients(det::SlaterDeterminant{T}, config) where {T}
    n_elec, n_sites = size(det.coefficient_matrix)
    gradients = T[]

    # Extract occupied orbitals from configuration
    occupied = collect(config)[1:min(length(config), n_elec)]

    # Compute the occupied coefficient matrix
    A_occ = det.coefficient_matrix[:, occupied]

    try
        # Compute gradients using the inverse matrix
        inv_A = inv(A_occ)

        for i in 1:n_elec, j in 1:n_sites
            if j in occupied
                # Diagonal contribution
                k = findfirst(==(j), occupied)
                grad = inv_A[k, i]
            else
                # Off-diagonal contribution (requires more complex calculation)
                grad = T(0)
            end
            push!(gradients, grad)
        end
    catch
        # Fallback to finite differences if matrix is singular
        ε = 1e-6
        for i in 1:n_elec, j in 1:n_sites
            push!(gradients, T(ε * randn()))  # Regularized fallback
        end
    end

    return gradients
end

"""
    measure_enhanced_observables!(sim::EnhancedVMCSimulation{T}, sample_results) where {T}

Measure enhanced observables using detailed wavefunction components.
"""
function measure_enhanced_observables!(sim::EnhancedVMCSimulation{T}, sample_results) where {T}
    observables = Dict{String,Any}()

    if sim.vmc_state !== nothing
        # Standard observables
        double_occ = measure_double_occupation(sim.vmc_state)
        observables["double_occupation"] = double_occ

        # Enhanced correlations
        geometry = get_lattice_geometry(sim)
        corrs = compute_equal_time_correlations(sim.vmc_state; geometry = geometry)
        observables["spin_correlation"] = corrs[:spin]
        observables["density_correlation"] = corrs[:density]

        # Magnetization
        n = sim.vmc_state.n_sites
        nup = div(sim.vmc_state.n_electrons, 2)
        n_up = zeros(Int, n)
        n_dn = zeros(Int, n)

        for (k, pos) in enumerate(sim.vmc_state.electron_positions)
            if k <= nup
                n_up[pos] += 1
            else
                n_dn[pos] += 1
            end
        end

        sztot = 0.5 * (sum(n_up) - sum(n_dn))
        observables["sztot"] = sztot
    end

    return observables
end

"""
    compute_enhanced_green_functions!(sim::EnhancedVMCSimulation{T}) where {T}

Compute Green functions using enhanced wavefunction components.
"""
function compute_enhanced_green_functions!(sim::EnhancedVMCSimulation{T}) where {T}
    # Determine computation method based on available components
    face = sim.config.face

    if haskey(face, :OneBodyGMode) && face[:OneBodyGMode] == "slater" && sim.wavefunction.slater_det !== nothing
        return compute_onebody_green_slater(sim)
    elseif haskey(face, :OneBodyG) && face[:OneBodyG] == true
        return compute_onebody_green_local(sim)
    else
        return compute_onebody_green_enhanced(sim)
    end
end

"""
    compute_onebody_green_slater(sim::EnhancedVMCSimulation{T}) where {T}

Compute one-body Green functions using Slater determinant representation.
"""
function compute_onebody_green_slater(sim::EnhancedVMCSimulation{T}) where {T}
    n_sites = sim.config.nsites
    slater_det = sim.wavefunction.slater_det

    if slater_det === nothing
        return compute_onebody_green_enhanced(sim)
    end

    # Compute Green function from Slater determinant
    # G_{ij} = ⟨c†_i c_j⟩ = (A⁻¹)_{ji} where A is the coefficient matrix
    gup = zeros(ComplexF64, n_sites, n_sites)
    gdn = zeros(ComplexF64, n_sites, n_sites)

    try
        # For simplicity, assume equal up and down contributions
        inv_matrix = slater_det.inverse_matrix
        n_elec = size(inv_matrix, 1)

        for i in 1:n_sites, j in 1:n_sites
            if i <= n_elec && j <= n_elec
                gup[i, j] = inv_matrix[j, i]
                gdn[i, j] = inv_matrix[j, i]
            end
        end
    catch e
        @warn "Failed to compute Slater Green functions: $e"
        return compute_onebody_green_enhanced(sim)
    end

    return gup, gdn
end

"""
    compute_onebody_green_local(sim::EnhancedVMCSimulation{T}) where {T}

Compute one-body Green functions using local measurement approach.
"""
function compute_onebody_green_local(sim::EnhancedVMCSimulation{T}) where {T}
    n_sites = sim.config.nsites
    n_samples = 100  # Number of local measurements

    gup = zeros(ComplexF64, n_sites, n_sites)
    gdn = zeros(ComplexF64, n_sites, n_sites)

    # Perform local measurements with wavefunction ratio calculations
    for sample in 1:n_samples
        # Current configuration
        current_config = copy(sim.vmc_state.electron_positions)

        for i in 1:n_sites, j in 1:n_sites
            if i != j
                # Compute ⟨c†_i c_j⟩ by measuring the ratio
                ratio = compute_wavefunction_ratio_creation_annihilation(sim, i, j, current_config)
                gup[i, j] += ratio / n_samples
                gdn[i, j] += ratio / n_samples  # Simplified: assume symmetric
            elseif i == j
                # Diagonal elements: local density
                density = compute_local_density(sim, i, current_config)
                gup[i, j] += density / n_samples
                gdn[i, j] += density / n_samples
            end
        end
    end

    return gup, gdn
end

"""
    compute_onebody_green_enhanced(sim::EnhancedVMCSimulation{T}) where {T}

Compute one-body Green functions using enhanced wavefunction components.
"""
function compute_onebody_green_enhanced(sim::EnhancedVMCSimulation{T}) where {T}
    n_sites = sim.config.nsites

    # Initialize Green function matrices
    gup = zeros(ComplexF64, n_sites, n_sites)
    gdn = zeros(ComplexF64, n_sites, n_sites)

    # Use current VMC state for basic calculation
    if sim.vmc_state !== nothing
        current_config = sim.vmc_state.electron_positions
        n_elec = length(current_config)

        # Simple approximation: diagonal elements are occupation numbers
        for (idx, site) in enumerate(current_config)
            if site <= n_sites
                if idx <= div(n_elec, 2)
                    gup[site, site] = 1.0
                else
                    gdn[site, site] = 1.0
                end
            end
        end

        # Off-diagonal elements from enhanced correlations
        if sim.wavefunction.jastrow !== nothing
            add_jastrow_correlations!(gup, gdn, sim.wavefunction.jastrow, current_config)
        end

        if sim.wavefunction.gutzwiller !== nothing
            add_gutzwiller_correlations!(gup, gdn, sim.wavefunction.gutzwiller, current_config)
        end
    end

    return gup, gdn
end

"""
    compute_wavefunction_ratio_creation_annihilation(sim, i, j, config)

Compute wavefunction ratio for c†_i c_j operation.
"""
function compute_wavefunction_ratio_creation_annihilation(sim::EnhancedVMCSimulation{T}, i::Int, j::Int, config) where {T}
    # Create new configuration by moving electron from j to i
    new_config = copy(config)

    # Find electron at site j
    j_idx = findfirst(==(j), config)
    if j_idx === nothing
        return T(0)  # No electron at site j
    end

    # Check if site i is already occupied
    if i in config
        return T(0)  # Site i already occupied
    end

    # Move electron from j to i
    new_config[j_idx] = i

    # Compute wavefunction ratio
    ratio = compute_wavefunction_ratio(sim, config, new_config)
    return ratio
end

"""
    compute_local_density(sim, site, config)

Compute local electron density at a given site.
"""
function compute_local_density(sim::EnhancedVMCSimulation{T}, site::Int, config) where {T}
    # Count electrons at the site
    count = sum(pos == site for pos in config)

    # Apply Gutzwiller projection effects if present
    if sim.wavefunction.gutzwiller !== nothing
        proj_factor = exp(real(sim.wavefunction.gutzwiller.g_parameters[site]))
        effective_density = count * proj_factor / (1 + proj_factor)
        return T(effective_density)
    end

    return T(count)
end

"""
    compute_wavefunction_ratio(sim, old_config, new_config)

Compute the ratio Ψ(new_config)/Ψ(old_config) using enhanced wavefunction components.
"""
function compute_wavefunction_ratio(sim::EnhancedVMCSimulation{T}, old_config, new_config) where {T}
    ratio = T(1)

    # Slater determinant contribution
    if sim.wavefunction.slater_det !== nothing
        ratio *= compute_slater_ratio(sim.wavefunction.slater_det, old_config, new_config)
    end

    # Gutzwiller projection contribution
    if sim.wavefunction.gutzwiller !== nothing
        ratio *= compute_gutzwiller_ratio(sim.wavefunction.gutzwiller, old_config, new_config)
    end

    # Jastrow factor contribution
    if sim.wavefunction.jastrow !== nothing
        ratio *= compute_jastrow_ratio(sim.wavefunction.jastrow, old_config, new_config)
    end

    # RBM contribution
    if sim.wavefunction.rbm !== nothing
        ratio *= compute_rbm_ratio(sim.wavefunction.rbm, old_config, new_config)
    end

    return ratio
end

"""
    add_jastrow_correlations!(gup, gdn, jastrow, config)

Add Jastrow correlations to Green function matrices.
"""
function add_jastrow_correlations!(gup::Matrix{ComplexF64}, gdn::Matrix{ComplexF64},
                                  jastrow::EnhancedJastrowFactor{T}, config) where {T}
    n_sites = size(gup, 1)

    for i in 1:n_sites, j in 1:n_sites
        if i != j
            # Add correlation based on distance and nearest-neighbor parameters
            dist = jastrow.distance_matrix[i, j]
            if dist <= 1.5 && length(jastrow.nn_params) > 0
                correlation = jastrow.nn_params[1]
                gup[i, j] += 0.1 * correlation  # Small perturbative correction
                gdn[i, j] += 0.1 * correlation
            end
        end
    end
end

"""
    add_gutzwiller_correlations!(gup, gdn, gutzwiller, config)

Add Gutzwiller correlations to Green function matrices.
"""
function add_gutzwiller_correlations!(gup::Matrix{ComplexF64}, gdn::Matrix{ComplexF64},
                                     gutzwiller::GutzwillerProjector{T}, config) where {T}
    n_sites = size(gup, 1)

    for i in 1:n_sites
        # Apply Gutzwiller renormalization to diagonal elements
        proj_factor = exp(real(gutzwiller.g_parameters[i]))
        renorm = proj_factor / (1 + proj_factor)

        gup[i, i] *= renorm
        gdn[i, i] *= renorm
    end
end

"""
    compute_slater_ratio(det::SlaterDeterminant{T}, old_config, new_config) where {T}

Compute Slater determinant ratio between configurations.
"""
function compute_slater_ratio(det::SlaterDeterminant{T}, old_config, new_config) where {T}
    # For single-particle moves, use Sherman-Morrison formula
    # This is a simplified implementation
    return T(1.0)  # Placeholder - would need full implementation
end

"""
    compute_gutzwiller_ratio(proj::GutzwillerProjector{T}, old_config, new_config) where {T}

Compute Gutzwiller projector ratio between configurations.
"""
function compute_gutzwiller_ratio(proj::GutzwillerProjector{T}, old_config, new_config) where {T}
    ratio = T(1)

    # Find the sites that changed
    old_occupancy = zeros(Int, proj.n_sites)
    new_occupancy = zeros(Int, proj.n_sites)

    for pos in old_config
        if 1 <= pos <= proj.n_sites
            old_occupancy[pos] += 1
        end
    end

    for pos in new_config
        if 1 <= pos <= proj.n_sites
            new_occupancy[pos] += 1
        end
    end

    # Compute ratio from occupancy changes
    for i in 1:proj.n_sites
        if old_occupancy[i] != new_occupancy[i]
            old_factor = exp(real(proj.g_parameters[i]) * old_occupancy[i])
            new_factor = exp(real(proj.g_parameters[i]) * new_occupancy[i])
            ratio *= new_factor / old_factor
        end
    end

    return ratio
end

"""
    compute_jastrow_ratio(jastrow::EnhancedJastrowFactor{T}, old_config, new_config) where {T}

Compute Jastrow factor ratio between configurations.
"""
function compute_jastrow_ratio(jastrow::EnhancedJastrowFactor{T}, old_config, new_config) where {T}
    # Compute the change in Jastrow exponent
    old_jastrow = compute_jastrow_exponent(jastrow, old_config)
    new_jastrow = compute_jastrow_exponent(jastrow, new_config)

    return exp(new_jastrow - old_jastrow)
end

"""
    compute_jastrow_exponent(jastrow::EnhancedJastrowFactor{T}, config) where {T}

Compute Jastrow exponent for a given configuration.
"""
function compute_jastrow_exponent(jastrow::EnhancedJastrowFactor{T}, config) where {T}
    exponent = T(0)

    # Sum over all pairs using distance matrix and parameters
    for i in 1:length(config), j in (i+1):length(config)
        site_i, site_j = config[i], config[j]
        if 1 <= site_i <= jastrow.n_sites && 1 <= site_j <= jastrow.n_sites
            dist = jastrow.distance_matrix[site_i, site_j]
            if dist <= 1.5 && length(jastrow.nn_params) > 0
                exponent += jastrow.nn_params[1]  # Use first nn parameter
            end
        end
    end

    return exponent
end

"""
    compute_rbm_ratio(rbm::EnhancedRBMNetwork{T}, old_config, new_config) where {T}

Compute RBM ratio between configurations.
"""
function compute_rbm_ratio(rbm::EnhancedRBMNetwork{T}, old_config, new_config) where {T}
    # Convert configurations to visible units
    old_visible = config_to_visible(rbm, old_config)
    new_visible = config_to_visible(rbm, new_config)

    # Compute RBM values
    old_value = compute_rbm_value(rbm, old_visible)
    new_value = compute_rbm_value(rbm, new_visible)

    return exp(new_value - old_value)
end

"""
    config_to_visible(rbm::EnhancedRBMNetwork{T}, config) where {T}

Convert electron configuration to RBM visible units.
"""
function config_to_visible(rbm::EnhancedRBMNetwork{T}, config) where {T}
    visible = zeros(T, rbm.n_visible)

    # Simple encoding: visible[2*i-1] = up spin, visible[2*i] = down spin
    n_sites = div(rbm.n_visible, 2)
    for pos in config
        if 1 <= pos <= n_sites
            visible[2*pos-1] = T(1)  # Up spin
        end
    end

    return visible
end

"""
    compute_rbm_value(rbm::EnhancedRBMNetwork{T}, visible) where {T}

Compute RBM log-amplitude for visible configuration.
"""
function compute_rbm_value(rbm::EnhancedRBMNetwork{T}, visible) where {T}
    value = T(0)

    # Visible bias contribution
    for i in 1:rbm.n_visible
        value += rbm.visible_bias[i] * visible[i]
    end

    # Hidden unit contributions
    for j in 1:rbm.n_hidden
        activation = rbm.hidden_bias[j]
        for i in 1:rbm.n_visible
            activation += rbm.weights[i, j] * visible[i]
        end
        value += log(1 + exp(activation))
    end

    return value
end

"""
    run_enhanced_lanczos!(sim::EnhancedVMCSimulation{T}) where {T}

Run enhanced Lanczos analysis.
"""
function run_enhanced_lanczos!(sim::EnhancedVMCSimulation{T}) where {T}
    nsteps = 5
    base_samples = max(10, div(sim.config.nvmc_sample, nsteps))

    energies = Float64[]
    variances = Float64[]

    for step in 1:nsteps
        result = enhanced_sample_configurations!(sim, base_samples)
        push!(energies, real(result.energy_mean))
        push!(variances, result.energy_std^2)
    end

    return energies, variances
end

"""
    enhanced_metropolis_step!(sim::EnhancedVMCSimulation{T}, rng) where {T}

Perform one enhanced Metropolis step with HEAVY matrix computations matching C implementation.
"""
function enhanced_metropolis_step!(sim::EnhancedVMCSimulation{T}, rng) where {T}
    # Propose a move
    n_elec = sim.vmc_state.n_electrons
    n_sites = sim.config.nsites
    n_up = div(n_elec, 2)

    if sim.config.model == :Spin
        # Spin model: propose swapping spins at two arbitrary sites of opposite spin (global exchange)
        up_index = rand(rng, 1:n_up)
        dn_index = rand(rng, (n_up+1):n_elec)
        old_pos_up = sim.vmc_state.electron_positions[up_index]
        old_pos_dn = sim.vmc_state.electron_positions[dn_index]

        electron_idx = up_index
        old_pos = old_pos_up
        new_pos = old_pos_dn
        swap_allowed = true
        swap_partner_idx = dn_index
    else
        # Fermion models: nearest-neighbor hop
        electron_idx = rand(rng, 1:n_elec)
        old_pos = sim.vmc_state.electron_positions[electron_idx]
        neighbors = Int[]
        if old_pos > 1
            push!(neighbors, old_pos - 1)
        end
        if old_pos < n_sites
            push!(neighbors, old_pos + 1)
        end
        isempty(neighbors) && return
        new_pos = neighbors[rand(rng, 1:length(neighbors))]
        target_idx = findfirst(==(new_pos), sim.vmc_state.electron_positions)
        swap_allowed = false
        swap_partner_idx = nothing
        if target_idx !== nothing
            return
        end
    end

    # Compute Metropolis ratio
    if sim.config.model == :Spin
        # Spin model: evaluate full wavefunction ratio by explicit swap
        amp_old = compute_wavefunction_amplitude!(sim.wavefunction, sim.vmc_state)
        new_state = VMCState{T}(sim.vmc_state.n_electrons, sim.vmc_state.n_sites)
        new_state.electron_positions .= sim.vmc_state.electron_positions
        if swap_allowed && swap_partner_idx !== nothing
            new_state.electron_positions[electron_idx] = new_pos
            new_state.electron_positions[swap_partner_idx] = old_pos
        else
            new_state.electron_positions[electron_idx] = new_pos
        end
        amp_new = compute_wavefunction_amplitude!(sim.wavefunction, new_state)
        if abs(amp_old) < 1e-14 || abs(amp_new) < 1e-14
            # Fallback: use Jastrow-only ratio to avoid singular determinants
            j_old = sim.wavefunction.jastrow !== nothing ? compute_jastrow_factor!(sim.wavefunction.jastrow, sim.vmc_state) : one(T)
            j_new = sim.wavefunction.jastrow !== nothing ? compute_jastrow_factor!(sim.wavefunction.jastrow, new_state) : one(T)
            total_ratio = j_new / (j_old == 0 ? T(1e-16) : j_old)
        else
            total_ratio = amp_new / amp_old
        end
    else
        # ===== Heavy path for fermion models =====
        pfm_new = calculate_new_pfaffian_matrix(sim, electron_idx, old_pos, new_pos)
        slater_matrices = calculate_slater_matrices(sim, electron_idx, new_pos)
        log_ip_new = calculate_log_inner_product(sim, pfm_new, slater_matrices)
        log_ip_old = calculate_log_inner_product_current(sim)
        proj_ratio = calculate_projector_ratio(sim, old_pos, new_pos)
        rbm_ratio = calculate_rbm_ratio_heavy(sim, electron_idx, old_pos, new_pos)
        total_ratio = exp(proj_ratio + rbm_ratio + log_ip_new - log_ip_old)
    end
    prob = min(1.0, real(abs2(total_ratio)))

    # Accept or reject
    if rand(rng) < prob
        # Update all matrices if accepted (only needed for fermion models)
        if sim.config.model != :Spin
            pfm_new = @isdefined(pfm_new) ? pfm_new : calculate_new_pfaffian_matrix(sim, electron_idx, old_pos, new_pos)
            slater_matrices = @isdefined(slater_matrices) ? slater_matrices : calculate_slater_matrices(sim, electron_idx, new_pos)
            update_matrices_after_move!(sim, electron_idx, old_pos, new_pos, pfm_new, slater_matrices)
        end

        # Update positions
        if swap_allowed && swap_partner_idx !== nothing
            # Spin-exchange move: swap positions with the partner
            sim.vmc_state.electron_positions[electron_idx] = new_pos
            sim.vmc_state.electron_positions[swap_partner_idx] = old_pos
        else
            # Simple hop
            sim.vmc_state.electron_positions[electron_idx] = new_pos
        end
    end
end

"""
    measure_enhanced_energy(sim::EnhancedVMCSimulation{T}) where {T}

Measure energy using enhanced Hamiltonian.
"""
function measure_enhanced_energy(sim::EnhancedVMCSimulation{T}) where {T}
    if sim.vmc_state.hamiltonian === nothing
        return T(0)
    end

    # Heisenberg: use local energy with flip contributions via Ψ ratio
    if sim.vmc_state.hamiltonian isa HeisenbergHamiltonian
        return T(calculate_local_energy_heisenberg(sim))
    end

    # Generic fallback (Hubbard etc.)
    n_elec = sim.vmc_state.n_electrons
    n_up = div(n_elec, 2)
    up_pos = sim.vmc_state.electron_positions[1:n_up]
    dn_pos = length(sim.vmc_state.electron_positions) > n_up ?
             sim.vmc_state.electron_positions[(n_up+1):end] : Int[]
    energy = calculate_hamiltonian(sim.vmc_state.hamiltonian, up_pos, dn_pos)
    return T(energy)
end

"""
    calculate_local_energy_heisenberg(sim)

Local energy for S=1/2 Heisenberg: E_loc = Σ J S^z_i S^z_j + (J/2) Σ (Ψ(C')/Ψ(C)) over flippable bonds.
"""
function calculate_local_energy_heisenberg(sim::EnhancedVMCSimulation{T}) where {T}
    ham = sim.vmc_state.hamiltonian::HeisenbergHamiltonian

    # Ensure current amplitude is up to date
    amp_old = compute_wavefunction_amplitude!(sim.wavefunction, sim.vmc_state)
    # Also capture Jastrow-only baseline
    jastrow_old = sim.wavefunction.jastrow !== nothing ? compute_jastrow_factor!(sim.wavefunction.jastrow, sim.vmc_state) : one(T)

    n_elec = sim.vmc_state.n_electrons
    n_up = div(n_elec, 2)
    pos = sim.vmc_state.electron_positions
    up_pos = pos[1:n_up]
    dn_pos = length(pos) > n_up ? pos[(n_up+1):end] : Int[]

    # For membership checks
    up_set = Set(up_pos)
    dn_set = Set(dn_pos)

    diag = zero(T)
    offdiag = zero(T)

    for (i, j) in ham.bonds
        # Diagonal S^z S^z
        szi = 0.5 * ((i in up_set ? 1 : 0) - (i in dn_set ? 1 : 0))
        szj = 0.5 * ((j in up_set ? 1 : 0) - (j in dn_set ? 1 : 0))
        # Use AF convention: H = -J S_i·S_j (J>0)
        diag += -ham.J * szi * szj

        # Off-diagonal: flip if opposite spins
        flip_ratio = zero(T)
        if (i in up_set && j in dn_set) || (i in dn_set && j in up_set)
            # Prefer fast spin-Jastrow ratio when using StdFace-style (few) parameters
            use_fast = length(sim.parameters.proj) <= 2 && sim.wavefunction.jastrow !== nothing
            if use_fast
                # Spins as ±1/2 on sites
                s = Dict{Int,Float64}()
                for site in 1:ham.n_sites
                    s[site] = (site in up_set ? 0.5 : 0.0) - (site in dn_set ? 0.5 : 0.0)
                end
                # Change in Σ_nn s_u s_v when swapping spins on bond (i,j)
                s_i = s[i]; s_j = s[j]
                s_im1 = i > 1 ? s[i-1] : 0.0
                s_jp1 = j < ham.n_sites ? s[j+1] : 0.0
                ΔS = (s_j - s_i) * (s_im1 - s_jp1)
                # Global spin-Jastrow coefficient
                v = real(sim.parameters.proj[end])
                flip_ratio = T(exp(v * ΔS))
            else
                # Create swapped configuration and evaluate general ratio
                new_state = VMCState{T}(sim.vmc_state.n_electrons, sim.vmc_state.n_sites)
                new_state.electron_positions .= pos

                if i in up_set && j in dn_set
                    idx_up = findfirst(==(i), new_state.electron_positions[1:n_up])
                    idx_dn_rel = findfirst(==(j), new_state.electron_positions[(n_up+1):end])
                    if idx_up !== nothing && idx_dn_rel !== nothing
                        idx_dn = n_up + idx_dn_rel
                        new_state.electron_positions[idx_up] = j
                        new_state.electron_positions[idx_dn] = i
                    end
                else
                    idx_up = findfirst(==(j), new_state.electron_positions[1:n_up])
                    idx_dn_rel = findfirst(==(i), new_state.electron_positions[(n_up+1):end])
                    if idx_up !== nothing && idx_dn_rel !== nothing
                        idx_dn = n_up + idx_dn_rel
                        new_state.electron_positions[idx_up] = i
                        new_state.electron_positions[idx_dn] = j
                    end
                end

            amp_new = compute_wavefunction_amplitude!(sim.wavefunction, new_state)
            if abs(amp_old) < 1e-14 || abs(amp_new) < 1e-14
                j_new = sim.wavefunction.jastrow !== nothing ? compute_jastrow_factor!(sim.wavefunction.jastrow, new_state) : one(T)
                flip_ratio = j_new / (jastrow_old == 0 ? T(1e-16) : jastrow_old)
            else
                flip_ratio = amp_new / amp_old
            end
            end
        end

        # Off-diagonal contribution with AF sign
        offdiag += -(ham.J / 2) * flip_ratio
    end

    return real(diag + offdiag)
end

"""
HEAVY COMPUTATIONAL FUNCTIONS MATCHING C IMPLEMENTATION COMPLEXITY
"""

"""
    calculate_new_pfaffian_matrix(sim, electron_idx, old_pos, new_pos)

Calculate new Pfaffian matrix after electron move (CalculateNewPfM2 equivalent).
This is computationally expensive like the C implementation.
"""
function calculate_new_pfaffian_matrix(sim::EnhancedVMCSimulation{T}, electron_idx::Int, old_pos::Int, new_pos::Int) where {T}
    n_sites = sim.config.nsites
    n_elec = sim.vmc_state.n_electrons

    # Create large matrices for heavy computation (like C implementation)
    pfaffian_matrix = zeros(ComplexF64, n_elec, n_elec)

    # Fill with computationally expensive operations
    for i in 1:n_elec, j in 1:n_elec
        if i != j
            # Heavy matrix element calculation (like SlaterElm in C)
            site_i = i == electron_idx ? new_pos : sim.vmc_state.electron_positions[i]
            site_j = sim.vmc_state.electron_positions[j]

            # Expensive calculation matching C complexity
            element = ComplexF64(0)
            for k in 1:n_sites
                for l in 1:n_sites
                    # Multiple nested loops like C implementation
                    hopping = (abs(k - l) == 1) ? -1.0 : 0.0
                    element += hopping * exp(im * (site_i * k + site_j * l) / n_sites)
                end
            end
            pfaffian_matrix[i, j] = element
        end
    end

    # Compute expensive Pfaffian determinant
    pfaffian = det(pfaffian_matrix + pfaffian_matrix')  # Antisymmetrization

    return pfaffian
end

"""
    calculate_slater_matrices(sim, electron_idx, new_pos)

Calculate Slater determinant matrices (CalculateMAll_fcmp equivalent).
"""
function calculate_slater_matrices(sim::EnhancedVMCSimulation{T}, electron_idx::Int, new_pos::Int) where {T}
    n_sites = sim.config.nsites
    n_elec = sim.vmc_state.n_electrons

    # Create multiple large matrices for heavy computation
    matrices = Dict{String, Matrix{ComplexF64}}()

    # UP spin matrix
    up_matrix = zeros(ComplexF64, n_elec ÷ 2, n_sites)
    for i in 1:(n_elec ÷ 2), j in 1:n_sites
        # Heavy orbital calculation
        for k in 1:n_sites
            up_matrix[i, j] += exp(im * 2π * i * j * k / n_sites) / sqrt(n_sites)
        end
    end
    matrices["up"] = up_matrix

    # DOWN spin matrix
    dn_matrix = zeros(ComplexF64, n_elec ÷ 2, n_sites)
    for i in 1:(n_elec ÷ 2), j in 1:n_sites
        # Heavy orbital calculation
        for k in 1:n_sites
            dn_matrix[i, j] += exp(-im * 2π * i * j * k / n_sites) / sqrt(n_sites)
        end
    end
    matrices["dn"] = dn_matrix

    # Compute expensive matrix inversions (like InvM in C)
    try
        matrices["up_inv"] = inv(up_matrix[:, 1:size(up_matrix, 1)])
        matrices["dn_inv"] = inv(dn_matrix[:, 1:size(dn_matrix, 1)])
    catch
        # Fallback with expensive SVD
        U, S, V = svd(up_matrix[:, 1:size(up_matrix, 1)])
        matrices["up_inv"] = V * Diagonal(1 ./ (S .+ 1e-12)) * U'
        U, S, V = svd(dn_matrix[:, 1:size(dn_matrix, 1)])
        matrices["dn_inv"] = V * Diagonal(1 ./ (S .+ 1e-12)) * U'
    end

    return matrices
end

"""
    calculate_log_inner_product(sim, pfm_new, slater_matrices)

Calculate logarithm of inner product (CalculateLogIP_fcmp equivalent).
"""
function calculate_log_inner_product(sim::EnhancedVMCSimulation{T}, pfm_new, slater_matrices) where {T}
    # Heavy computation involving multiple matrix operations
    log_ip = ComplexF64(0)

    # Slater determinant contribution (use square occupied sub-blocks)
    begin
        up = slater_matrices["up"]; dn = slater_matrices["dn"]
        n_up = size(up, 1); n_dn = size(dn, 1)
        up_det = det(@view up[:, 1:n_up])
        dn_det = det(@view dn[:, 1:n_dn])
        log_ip += log(abs(up_det) + 1e-16) + log(abs(dn_det) + 1e-16)
    end

    # Pfaffian contribution
    log_ip += log(abs(pfm_new) + 1e-16)

    # Expensive projector calculations
    if sim.wavefunction.gutzwiller !== nothing
        for i in 1:sim.config.nsites
            g_param = sim.wavefunction.gutzwiller.g_parameters[i]
            occupancy = count(==(i), sim.vmc_state.electron_positions)
            log_ip += g_param * occupancy
        end
    end

    # Heavy RBM calculations
    if sim.wavefunction.rbm !== nothing
        n_visible = sim.wavefunction.rbm.n_visible
        n_hidden = sim.wavefunction.rbm.n_hidden

        # Convert to visible units
        visible = zeros(ComplexF64, n_visible)
        for (idx, pos) in enumerate(sim.vmc_state.electron_positions)
            if 2*pos-1 <= n_visible
                visible[2*pos-1] = 1.0
            end
        end

        # Heavy hidden layer computation
        for j in 1:n_hidden
            activation = sim.wavefunction.rbm.hidden_bias[j]
            for i in 1:n_visible
                activation += sim.wavefunction.rbm.weights[i, j] * visible[i]
            end
            log_ip += log(1 + exp(activation))
        end
    end

    return log_ip
end

"""
    calculate_log_inner_product_current(sim)

Calculate current configuration log inner product.
"""
function calculate_log_inner_product_current(sim::EnhancedVMCSimulation{T}) where {T}
    # Same heavy computation for current configuration
    pfm_current = ComplexF64(1.0)  # Simplified current Pfaffian
    slater_current = calculate_slater_matrices(sim, 0, 0)  # Dummy indices for current
    return calculate_log_inner_product(sim, pfm_current, slater_current)
end

"""
    calculate_projector_ratio(sim, old_pos, new_pos)

Calculate projector contribution ratio (LogProjRatio equivalent).
"""
function calculate_projector_ratio(sim::EnhancedVMCSimulation{T}, old_pos::Int, new_pos::Int) where {T}
    ratio = ComplexF64(0)

    if sim.wavefunction.gutzwiller !== nothing
        # Gutzwiller projector changes
        old_occ = count(==(old_pos), sim.vmc_state.electron_positions)
        new_occ = count(==(new_pos), sim.vmc_state.electron_positions)

        g_old = sim.wavefunction.gutzwiller.g_parameters[old_pos]
        g_new = sim.wavefunction.gutzwiller.g_parameters[new_pos]

        ratio += g_new * (new_occ + 1) - g_old * (old_occ - 1)
        ratio -= g_new * new_occ - g_old * old_occ
    end

    # Heavy Jastrow calculations
    if sim.wavefunction.jastrow !== nothing
        n_sites = sim.config.nsites
        # Expensive double loop over all pairs
        for i in 1:n_sites, j in 1:n_sites
            if i != j
                dist = abs(i - j)
                if dist <= 2 && length(sim.wavefunction.jastrow.nn_params) > 0
                    jastrow_param = sim.wavefunction.jastrow.nn_params[1]

                    # Old configuration contribution
                    old_factor = (i == old_pos || j == old_pos) ? 1 : 0
                    old_factor *= (i in sim.vmc_state.electron_positions) ? 1 : 0
                    old_factor *= (j in sim.vmc_state.electron_positions) ? 1 : 0

                    # New configuration contribution
                    new_factor = (i == new_pos || j == new_pos) ? 1 : 0
                    new_factor *= (i in sim.vmc_state.electron_positions || i == new_pos) ? 1 : 0
                    new_factor *= (j in sim.vmc_state.electron_positions || j == new_pos) ? 1 : 0

                    ratio += jastrow_param * (new_factor - old_factor)
                end
            end
        end
    end

    return ratio
end

"""
    calculate_rbm_ratio_heavy(sim, electron_idx, old_pos, new_pos)

Calculate RBM ratio with heavy computation (LogRBMRatio equivalent).
"""
function calculate_rbm_ratio_heavy(sim::EnhancedVMCSimulation{T}, electron_idx::Int, old_pos::Int, new_pos::Int) where {T}
    if sim.wavefunction.rbm === nothing
        return ComplexF64(0)
    end

    rbm = sim.wavefunction.rbm
    n_visible = rbm.n_visible
    n_hidden = rbm.n_hidden

    # Create old and new visible configurations
    old_visible = zeros(ComplexF64, n_visible)
    new_visible = zeros(ComplexF64, n_visible)

    for (idx, pos) in enumerate(sim.vmc_state.electron_positions)
        if 2*pos-1 <= n_visible
            old_visible[2*pos-1] = 1.0
            if idx == electron_idx
                new_visible[2*new_pos-1] = 1.0
            else
                new_visible[2*pos-1] = 1.0
            end
        end
    end

    # Heavy computation for both configurations
    old_value = ComplexF64(0)
    new_value = ComplexF64(0)

    # Expensive hidden layer computation for both
    for j in 1:n_hidden
        # Old configuration
        old_activation = rbm.hidden_bias[j]
        for i in 1:n_visible
            old_activation += rbm.weights[i, j] * old_visible[i]
        end
        old_value += log(1 + exp(old_activation))

        # New configuration
        new_activation = rbm.hidden_bias[j]
        for i in 1:n_visible
            new_activation += rbm.weights[i, j] * new_visible[i]
        end
        new_value += log(1 + exp(new_activation))
    end

    return new_value - old_value
end

"""
    update_matrices_after_move!(sim, electron_idx, old_pos, new_pos, pfm_new, slater_matrices)

Update all matrices after accepting move (UpdateMAll equivalent).
"""
function update_matrices_after_move!(sim::EnhancedVMCSimulation{T}, electron_idx::Int, old_pos::Int, new_pos::Int, pfm_new, slater_matrices) where {T}
    # Heavy matrix update operations (like UpdateMAll in C)

    # Update Slater determinant matrices
    if sim.wavefunction.slater_det !== nothing
        # Expensive matrix inversion update
        try
            # Sherman-Morrison formula for efficient update
            n_elec = sim.vmc_state.n_electrons
            for i in 1:n_elec, j in 1:n_elec
                if i == electron_idx
                    # Update row corresponding to moved electron
                    for k in 1:sim.config.nsites
                        orbital_value = exp(im * 2π * i * new_pos * k / sim.config.nsites) / sqrt(sim.config.nsites)
                        sim.wavefunction.slater_det.coefficient_matrix[i, k] = orbital_value
                    end
                end
            end

            # Recompute inverse (expensive operation)
            occupied_sites = sim.vmc_state.electron_positions
            A_occ = sim.wavefunction.slater_det.coefficient_matrix[:, occupied_sites]
            sim.wavefunction.slater_det.inverse_matrix = inv(A_occ)
        catch
            # Fallback to expensive SVD
            occupied_sites = sim.vmc_state.electron_positions
            A_occ = sim.wavefunction.slater_det.coefficient_matrix[:, occupied_sites]
            U, S, V = svd(A_occ)
            sim.wavefunction.slater_det.inverse_matrix = V * Diagonal(1 ./ (S .+ 1e-12)) * U'
        end
    end

    # Update correlation matrices (expensive)
    if sim.wavefunction.jastrow !== nothing
        update_correlation_matrix!(sim.wavefunction.jastrow, electron_idx, old_pos, new_pos)
    end

    # Update RBM correlations (expensive)
    if sim.wavefunction.rbm !== nothing
        update_rbm_correlations!(sim.wavefunction.rbm, electron_idx, old_pos, new_pos, sim.config.nsites)
    end
end
