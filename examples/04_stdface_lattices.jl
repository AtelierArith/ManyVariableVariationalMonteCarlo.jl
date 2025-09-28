using ManyVariableVariationalMonteCarlo
using Printf

# StdFace Standard Lattice Models Demonstration
function main()
    println("="^60)
    println("StdFace Standard Lattice Models Demonstration")
    println("="^60)

    # 1. Chain lattice (1D Hubbard model)
    println("\n--- 1D Chain Lattice ---")
    L = 8
    ham_chain, geom_chain, config_chain = stdface_chain(L, "Hubbard"; t=1.0, U=4.0, V=0.5)

    println("Chain lattice: L=$L")
    lattice_summary(geom_chain)
    hamiltonian_summary(ham_chain)

    # Test energy calculation
    electron_config = collect(1:4)  # 4 electrons
    electron_numbers = zeros(Int, 2*L)
    for (i, pos) in enumerate(electron_config)
        spin = (i-1) % 2  # Alternate spins
        electron_numbers[pos + spin*L] = 1
    end
    energy_chain = calculate_hamiltonian(ham_chain, electron_config, electron_numbers)
    println("Sample energy: E = ", real(energy_chain))

    # 2. Square lattice (2D Hubbard model)
    println("\n--- 2D Square Lattice ---")
    Lx, Ly = 3, 3
    ham_square, geom_square, config_square = stdface_square(Lx, Ly, "Hubbard";
                                                           t=1.0, U=4.0, t_prime=0.2)

    println("Square lattice: $(Lx)×$(Ly)")
    lattice_summary(geom_square)
    hamiltonian_summary(ham_square)

    # 3. Triangular lattice (Spin model)
    println("\n--- 2D Triangular Lattice (Spin Model) ---")
    Lx, Ly = 4, 4
    ham_tri, geom_tri, config_tri = stdface_triangular(Lx, Ly, "Spin"; J=1.0)

    println("Triangular lattice: $(Lx)×$(Ly)")
    lattice_summary(geom_tri)
    hamiltonian_summary(ham_tri)

    # 4. Honeycomb lattice
    println("\n--- 2D Honeycomb Lattice ---")
    Lx, Ly = 3, 3
    ham_honey, geom_honey, config_honey = stdface_honeycomb(Lx, Ly, "Hubbard"; t=1.0, U=3.0)

    println("Honeycomb lattice: $(Lx)×$(Ly)")
    lattice_summary(geom_honey)
    hamiltonian_summary(ham_honey)

    # 5. Kagome lattice
    println("\n--- 2D Kagome Lattice ---")
    Lx, Ly = 2, 2
    ham_kagome, geom_kagome, config_kagome = stdface_kagome(Lx, Ly, "Hubbard"; t=1.0, U=2.0)

    println("Kagome lattice: $(Lx)×$(Ly)")
    lattice_summary(geom_kagome)
    hamiltonian_summary(ham_kagome)

    # 6. Ladder lattice
    println("\n--- Ladder Lattice ---")
    L = 6
    W = 2
    ham_ladder, geom_ladder, config_ladder = stdface_ladder(L, W, "Hubbard";
                                                           t=1.0, t_perp=0.8, U=2.0)

    println("Ladder lattice: L=$L, W=$W")
    lattice_summary(geom_ladder)
    hamiltonian_summary(ham_ladder)

    # 7. Coordinate and neighbor analysis
    println("\n--- Coordinate and Neighbor Analysis ---")

    # Generate coordinates for square lattice
    coords = generate_site_coordinates(geom_square)
    println("Square lattice coordinates (first 5 sites):")
    for i in 1:min(5, size(coords, 1))
        @printf("  Site %d: (%.2f, %.2f)\n", i, coords[i, 1], coords[i, 2])
    end

    # Generate neighbor lists
    neighbors_nn = generate_neighbor_list(geom_square, 1.1)  # Nearest neighbors
    neighbors_nnn = generate_neighbor_list(geom_square, 1.5)  # Include next-nearest

    println("Neighbor analysis for square lattice:")
    println("  Site 1 (corner): $(length(neighbors_nn[1])) NN, $(length(neighbors_nnn[1])) NN+NNN")
    println("  Site 5 (center): $(length(neighbors_nn[5])) NN, $(length(neighbors_nnn[5])) NN+NNN")

    # 8. Comparison of lattice properties
    println("\n--- Lattice Properties Comparison ---")
    lattices = [
        ("Chain", geom_chain),
        ("Square", geom_square),
        ("Triangular", geom_tri),
        ("Honeycomb", geom_honey),
        ("Kagome", geom_kagome),
        ("Ladder", geom_ladder)
    ]

    println(@sprintf("%-12s %-8s %-8s %-12s %-8s", "Lattice", "Sites", "UC Sites", "Dimensions", "Coord#"))
    println("-"^50)

    for (name, geom) in lattices
        coord_num = length(generate_neighbor_list(geom, 1.1)[1])  # First site coordination
        println(@sprintf("%-12s %-8d %-8d %-12d %-8d",
                name, geom.n_sites_total, geom.n_sites_unit_cell,
                geom.dimensions, coord_num))
    end

    # 9. Model parameter demonstration
    println("\n--- Model Parameter Effects ---")

    # Compare different U values for chain
    U_values = [0.0, 2.0, 4.0, 8.0]
    println("Chain Hubbard model with different U values:")
    println("U\tCoulomb terms\tTransfer terms")

    for U in U_values
        ham_test, _, _ = stdface_chain(4, "Hubbard"; t=1.0, U=U)
        n_coulomb = length(ham_test.coulomb_intra_terms)
        n_transfer = length(ham_test.transfer_terms)
        println("$U\t$n_coulomb\t\t$n_transfer")
    end

    println("\n" * "="^60)
    println("StdFace Demonstration Complete")
    println("="^60)
end

main()
