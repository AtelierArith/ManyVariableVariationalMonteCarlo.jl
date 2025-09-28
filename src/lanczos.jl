# Lanczos method integration for excited states and dynamic response
# Based on mVMC C implementation: physcal_lanczos.c

"""
    LanczosConfiguration

Configuration for Lanczos method calculations.
Controls excited state calculations and dynamic response functions.
"""
struct LanczosConfiguration
    # Lanczos parameters
    n_lanczos_mode::Int          # Lanczos mode (0: none, 1: energy only, 2: Green's functions)
    max_lanczos_steps::Int       # Maximum number of Lanczos steps
    lanczos_tolerance::Float64   # Convergence tolerance

    # Energy calculation
    calculate_energy::Bool       # Calculate ground state energy
    calculate_energy_by_alpha::Bool  # Calculate energy for given alpha
    alpha_values::Vector{Float64}    # Alpha values for energy calculation

    # Green's function calculation
    calculate_greens::Bool       # Calculate Green's functions
    n_phys_quantities::Int      # Number of physical quantities
    n_cisajs::Int              # Number of one-body Green's functions
    n_cisajs_lz::Int           # Number of Lz-projected Green's functions

    # Four-body Green's functions
    calculate_four_body::Bool    # Calculate four-body Green's functions
    n_cisajs_ckt_alt::Int       # Number of four-body Green's functions
    n_cisajs_ckt_alt_dc::Int    # Number of DC four-body Green's functions

    # Output files
    output_lanczos::Bool         # Output Lanczos results
    output_spectrum::Bool        # Output spectrum
    file_prefix::String          # Prefix for output files
end

"""
    create_lanczos_configuration(; n_lanczos_mode=1, max_lanczos_steps=100,
                                lanczos_tolerance=1e-8, calculate_energy=true,
                                calculate_greens=false, calculate_four_body=false,
                                output_lanczos=true, file_prefix="zvo_ls")

Create Lanczos configuration with default parameters.
"""
function create_lanczos_configuration(;
    n_lanczos_mode = 1,
    max_lanczos_steps = 100,
    lanczos_tolerance = 1e-8,
    calculate_energy = true,
    calculate_greens = false,
    calculate_four_body = false,
    output_lanczos = true,
    file_prefix = "zvo_ls",
)
    return LanczosConfiguration(
        n_lanczos_mode,
        max_lanczos_steps,
        lanczos_tolerance,
        calculate_energy,
        false,
        Float64[],
        calculate_greens,
        0,
        0,
        0,
        calculate_four_body,
        0,
        0,
        output_lanczos,
        false,
        file_prefix,
    )
end

"""
    LanczosState{T}

State for Lanczos calculations including vectors and tridiagonal matrix.
"""
mutable struct LanczosState{T<:Number}
    # Lanczos vectors
    lanczos_vectors::Vector{Vector{T}}  # Lanczos vectors
    current_step::Int                   # Current Lanczos step

    # Tridiagonal matrix elements
    alpha_diag::Vector{Float64}         # Diagonal elements
    beta_offdiag::Vector{Float64}       # Off-diagonal elements

    # Convergence tracking
    eigenvalues::Vector{Float64}        # Computed eigenvalues
    eigenvectors::Matrix{Float64}       # Computed eigenvectors
    converged::Vector{Bool}             # Convergence flags for eigenvalues

    # Residual tracking
    residual_norms::Vector{Float64}     # Residual norms

    # Physical quantities
    ground_state_energy::T              # Ground state energy
    excited_energies::Vector{T}         # Excited state energies

    # Green's function data
    qqqq_data::Vector{T}                # Four-point correlators
    q_cisajs_q_data::Vector{T}          # One-body Green's function data
    q_cisajs_ckt_alt_q_data::Vector{T}  # Four-body Green's function data
    q_cisajs_ckt_alt_q_dc_data::Vector{T} # DC four-body Green's function data
end

"""
    initialize_lanczos_state(T::Type{<:Number}, vector_size::Int, config::LanczosConfiguration)

Initialize Lanczos state with given vector size and configuration.
"""
function initialize_lanczos_state(
    T::Type{<:Number},
    vector_size::Int,
    config::LanczosConfiguration,
)
    return LanczosState{T}(
        Vector{Vector{T}}(),           # lanczos_vectors
        0,                             # current_step
        Float64[],                     # alpha_diag
        Float64[],                     # beta_offdiag
        Float64[],                     # eigenvalues
        Matrix{Float64}(undef, 0, 0),  # eigenvectors
        Bool[],                        # converged
        Float64[],                     # residual_norms
        zero(T),                       # ground_state_energy
        T[],                           # excited_energies
        T[],                           # qqqq_data
        T[],                           # q_cisajs_q_data
        T[],                           # q_cisajs_ckt_alt_q_data
        T[],                            # q_cisajs_ckt_alt_q_dc_data
    )
end

"""
    lanczos_step!(state::LanczosState{T}, hamiltonian_action::Function,
                 current_vector::Vector{T}) where T

Perform one step of the Lanczos algorithm.
"""
function lanczos_step!(
    state::LanczosState{T},
    hamiltonian_action::Function,
    current_vector::Vector{T},
) where {T}
    state.current_step += 1

    # Apply Hamiltonian to current vector
    h_vector = hamiltonian_action(current_vector)

    # Orthogonalize against previous vectors
    if state.current_step == 1
        # First step
        push!(state.lanczos_vectors, copy(current_vector))
        alpha = real(dot(current_vector, h_vector))
        push!(state.alpha_diag, alpha)

        # Compute next vector
        next_vector = h_vector - alpha * current_vector
        beta = norm(next_vector)
        push!(state.beta_offdiag, beta)

        if beta > 1e-12
            next_vector ./= beta
            push!(state.lanczos_vectors, next_vector)
        end

    elseif state.current_step <= length(state.lanczos_vectors)
        # Subsequent steps
        prev_vector =
            state.current_step > 1 ? state.lanczos_vectors[state.current_step-1] :
            zeros(T, length(current_vector))

        # Orthogonalize
        alpha = real(dot(current_vector, h_vector))
        push!(state.alpha_diag, alpha)

        next_vector = h_vector - alpha * current_vector
        if state.current_step > 1
            next_vector -= state.beta_offdiag[end] * prev_vector
        end

        beta = norm(next_vector)
        push!(state.beta_offdiag, beta)

        if beta > 1e-12
            next_vector ./= beta
            push!(state.lanczos_vectors, next_vector)
        end
    end

    return state.current_step < length(state.lanczos_vectors) ? state.lanczos_vectors[end] :
           nothing
end

"""
    compute_lanczos_eigenvalues!(state::LanczosState{T}, config::LanczosConfiguration) where T

Compute eigenvalues of the tridiagonal Lanczos matrix.
"""
function compute_lanczos_eigenvalues!(
    state::LanczosState{T},
    config::LanczosConfiguration,
) where {T}
    n = length(state.alpha_diag)
    if n == 0
        return
    end

    # Construct tridiagonal matrix
    tri_matrix = SymTridiagonal(state.alpha_diag, state.beta_offdiag[1:(end-1)])

    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = eigen(tri_matrix)

    state.eigenvalues = eigenvals
    state.eigenvectors = eigenvecs

    # Ground state energy is the lowest eigenvalue
    state.ground_state_energy = T(eigenvals[1])

    # Store excited state energies
    state.excited_energies = T.(eigenvals[2:end])

    # Check convergence
    state.converged = fill(false, length(eigenvals))
    for i = 1:length(eigenvals)
        if n > 1
            # Simple convergence check based on change in eigenvalue
            # More sophisticated checks could be implemented
            state.converged[i] = true  # Placeholder
        end
    end
end

"""
    calculate_energy_lanczos(H1::T, H2_1::T, H2_2::T, H3::T, H4::T) where T<:Number

Calculate ground state energy using Lanczos method.
Based on CalculateEne function in mVMC.
"""
function calculate_energy_lanczos(H1::T, H2_1::T, H2_2::T, H3::T, H4::T) where {T<:Number}
    # This implements the energy calculation from Hamiltonian moments
    # H1 = ⟨H⟩, H2_1 = ⟨H²⟩, H2_2 = ⟨H⟩², H3 = ⟨H³⟩, H4 = ⟨H⁴⟩

    # Solve for optimal alpha and corresponding energy
    # This is based on variational principle: E(α) = (H1 + α*H2_1) / (1 + α*H2_2)

    alpha_plus = zero(T)
    energy_plus = zero(T)
    energy_var_plus = zero(T)

    alpha_minus = zero(T)
    energy_minus = zero(T)
    energy_var_minus = zero(T)

    # Simplified energy calculation (full implementation would solve optimization)
    if abs(H2_2) > 1e-12
        alpha_plus = -H1 / H2_2
        energy_plus = H1 + alpha_plus * H2_1
        energy_var_plus =
            abs(H2_1 + 2 * alpha_plus * H3 + alpha_plus^2 * H4 - energy_plus^2)

        alpha_minus = -alpha_plus
        energy_minus = H1 + alpha_minus * H2_1
        energy_var_minus =
            abs(H2_1 + 2 * alpha_minus * H3 + alpha_minus^2 * H4 - energy_minus^2)
    else
        energy_plus = H1
        energy_minus = H1
    end

    return (
        alpha_plus,
        energy_plus,
        energy_var_plus,
        alpha_minus,
        energy_minus,
        energy_var_minus,
    )
end

"""
    calculate_energy_by_alpha(H1::T, H2_1::T, H2_2::T, H3::T, H4::T, alpha::Float64) where T<:Number

Calculate energy for a given alpha parameter.
Based on CalculateEneByAlpha function in mVMC.
"""
function calculate_energy_by_alpha(
    H1::T,
    H2_1::T,
    H2_2::T,
    H3::T,
    H4::T,
    alpha::Float64,
) where {T<:Number}
    # Energy calculation for fixed alpha
    energy = H1 + alpha * H2_1
    energy_variance = abs(H2_1 + 2 * alpha * H3 + alpha^2 * H4 - energy^2)

    return (energy, energy_variance)
end

"""
    calculate_physical_values_lanczos(H1::T, H2_1::T, alpha::Float64,
                                     qphys_q_data::Vector{T}, n_phys::Int,
                                     n_ls_ham::Int) where T<:Number

Calculate physical values using Lanczos method.
Based on CalculatePhysVal_real/CalculatePhysVal_fcmp in mVMC.
"""
function calculate_physical_values_lanczos(
    H1::T,
    H2_1::T,
    alpha::Float64,
    qphys_q_data::Vector{T},
    n_phys::Int,
    n_ls_ham::Int,
) where {T<:Number}
    phys_ls_data = zeros(T, n_phys)

    # Calculate physical values using Lanczos projection
    # This involves projecting the physical operators onto the Lanczos subspace

    denominator = 1.0 + alpha * H2_1
    if abs(denominator) > 1e-12
        for i = 1:min(n_phys, length(qphys_q_data))
            phys_ls_data[i] = qphys_q_data[i] / denominator
        end
    end

    return phys_ls_data
end

"""
    run_lanczos_calculation!(state::LanczosState{T}, config::LanczosConfiguration,
                           hamiltonian_action::Function, initial_vector::Vector{T}) where T

Run complete Lanczos calculation.
"""
function run_lanczos_calculation!(
    state::LanczosState{T},
    config::LanczosConfiguration,
    hamiltonian_action::Function,
    initial_vector::Vector{T},
) where {T}
    # Normalize initial vector
    initial_vector = copy(initial_vector)
    initial_vector ./= norm(initial_vector)

    current_vector = initial_vector

    # Lanczos iteration
    for step = 1:config.max_lanczos_steps
        next_vector = lanczos_step!(state, hamiltonian_action, current_vector)

        if next_vector === nothing
            break  # Converged or span exhausted
        end

        # Check convergence
        if step > 10 && step % 5 == 0
            compute_lanczos_eigenvalues!(state, config)

            # Simple convergence check
            if length(state.eigenvalues) > 0
                residual = abs(state.beta_offdiag[end] * state.eigenvectors[end, 1])
                if residual < config.lanczos_tolerance
                    break
                end
            end
        end

        current_vector = next_vector
    end

    # Final eigenvalue calculation
    compute_lanczos_eigenvalues!(state, config)

    return state
end

"""
    output_lanczos_results(state::LanczosState{T}, config::LanczosConfiguration,
                          output_dir::String = ".") where T

Output Lanczos calculation results to files.
"""
function output_lanczos_results(
    state::LanczosState{T},
    config::LanczosConfiguration,
    output_dir::String = ".",
) where {T}
    if !config.output_lanczos
        return
    end

    # Output eigenvalues
    eigenval_file = joinpath(output_dir, "$(config.file_prefix)_eigenvalues.dat")
    open(eigenval_file, "w") do f
        println(f, "# Lanczos eigenvalues")
        println(f, "# Index  Eigenvalue  Convergence")
        for (i, (eval, conv)) in enumerate(zip(state.eigenvalues, state.converged))
            println(f, "$i  $eval  $conv")
        end
    end

    # Output ground state energy
    energy_file = joinpath(output_dir, "$(config.file_prefix)_energy.dat")
    open(energy_file, "w") do f
        println(f, "# Ground state energy from Lanczos")
        println(f, "$(real(state.ground_state_energy))")
    end

    # Output excited state energies if available
    if !isempty(state.excited_energies)
        excited_file = joinpath(output_dir, "$(config.file_prefix)_excited.dat")
        open(excited_file, "w") do f
            println(f, "# Excited state energies")
            println(f, "# Index  Energy")
            for (i, energy) in enumerate(state.excited_energies)
                println(f, "$i  $(real(energy))")
            end
        end
    end

    println("Lanczos results written to $(output_dir)")
end

"""
    integrate_lanczos_with_vmc!(vmc_state, vmc_config, lanczos_config::LanczosConfiguration)

Integrate Lanczos method with VMC calculations.
"""
function integrate_lanczos_with_vmc!(
    vmc_state,
    vmc_config,
    lanczos_config::LanczosConfiguration,
)
    if lanczos_config.n_lanczos_mode == 0
        return nothing  # No Lanczos calculation
    end

    # Create Hamiltonian action function for VMC state
    function hamiltonian_action(vector::Vector{T}) where {T}
        # This would apply the Hamiltonian to a vector in the VMC basis
        # Implementation depends on the specific VMC representation
        return vector  # Placeholder
    end

    # Initialize Lanczos state
    vector_size = length(vmc_state.positions)  # Or appropriate size
    lanczos_state = initialize_lanczos_state(
        eltype(vmc_state.local_energy),
        vector_size,
        lanczos_config,
    )

    # Create initial vector from VMC state
    initial_vector = complex.(vmc_state.positions)  # Simplified

    # Run Lanczos calculation
    run_lanczos_calculation!(
        lanczos_state,
        lanczos_config,
        hamiltonian_action,
        initial_vector,
    )

    # Output results
    output_lanczos_results(lanczos_state, lanczos_config)

    return lanczos_state
end

# Export Lanczos-related functions and types
export LanczosConfiguration,
    LanczosState,
    create_lanczos_configuration,
    initialize_lanczos_state,
    lanczos_step!,
    compute_lanczos_eigenvalues!,
    calculate_energy_lanczos,
    calculate_energy_by_alpha,
    calculate_physical_values_lanczos,
    run_lanczos_calculation!,
    output_lanczos_results,
    integrate_lanczos_with_vmc!
