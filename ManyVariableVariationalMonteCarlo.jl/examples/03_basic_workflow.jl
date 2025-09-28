using ManyVariableVariationalMonteCarlo
using Random

# Basic VMC workflow demonstration
function main()
    println("=== Basic VMC Workflow Demonstration ===")

    # Create a simple 1D Hubbard chain
    face = FaceDefinition()
    push_definition!(face, :model, "FermionHubbard")
    push_definition!(face, :lattice, "chain")
    push_definition!(face, :L, 4)
    push_definition!(face, :nelec, 2)
    push_definition!(face, :t, 1.0)
    push_definition!(face, :U, 2.0)
    push_definition!(face, :NVMCCalMode, 0)  # Optimization mode
    push_definition!(face, :NSROptItrStep, 3)  # Short optimization
    push_definition!(face, :NSROptItrSmp, 10)
    push_definition!(face, :NVMCSample, 20)

    config = SimulationConfig(face)
    layout = ParameterLayout(2, 0, 4, 0)  # Some projection and Slater parameters

    println("System: $(config.model) on $(config.lattice)")
    println("Sites: $(config.nsites), Electrons: $(config.nelec)")
    println("Parameters: $(length(layout)) total")

    # Create simulation
    sim = VMCSimulation(config, layout; T=ComplexF64)

    # Run simulation
    println("\n--- Running VMC Simulation ---")
    run_simulation!(sim)

    # Print results
    print_simulation_summary(sim)

    # --- Optional: use Slater projector for one-body Green's output ---
    # By default, zvo_cisajs.dat writes a snapshot-based estimator (diagonal occupancy).
    # To enable a projector-based one-body G from the Slater determinant, set:
    #   push_definition!(face, :OneBodyGMode, "slater")
    # or equivalently:
    #   push_definition!(face, :OneBodyG, "slater")
    # Then re-run the simulation and output to a different directory to compare.
    # Example (commented out):
    # face2 = FaceDefinition()
    # push_definition!(face2, :model, "FermionHubbard")
    # push_definition!(face2, :lattice, "chain")
    # push_definition!(face2, :L, 4)
    # push_definition!(face2, :nelec, 2)
    # push_definition!(face2, :OneBodyGMode, "slater")
    # config2 = SimulationConfig(face2)
    # sim2 = VMCSimulation(config2, layout; T=ComplexF64)
    # run_simulation!(sim2)
    # output_results(sim2, "output_slater")

    # Test Hamiltonian creation
    println("\n--- Testing Hamiltonian ---")
    ham = create_hubbard_hamiltonian(config.nsites, config.nelec, config.t, config.u; lattice_type=:chain)
    hamiltonian_summary(ham)

    # Test energy calculation
    electron_config = [1, 2]  # Simple configuration
    electron_numbers = [1, 0, 0, 0, 0, 1, 0, 0]  # [n1↑, n2↑, n3↑, n4↑, n1↓, n2↓, n3↓, n4↓]
    energy = calculate_hamiltonian(ham, electron_config, electron_numbers)
    println("Test energy calculation: E = ", real(energy))

    println("\n=== Demonstration Complete ===")
end

main()
