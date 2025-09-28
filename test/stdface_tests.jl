# ReTestItems does not allow using statements outside of @testitem blocks

@testitem "Chain lattice creation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test basic chain lattice creation
    L = 6
    geometry = create_chain_lattice(L)

    @test geometry.lattice_type == CHAIN_LATTICE
    @test geometry.dimensions == 1
    @test geometry.L == [L]
    @test geometry.n_sites_total == L
    @test geometry.n_sites_unit_cell == 1
    @test length(geometry.neighbor_vectors) == 2  # +1 and -1 directions

    # Test coordinate generation
    coords = generate_site_coordinates(geometry)
    @test size(coords) == (L, 1)
    @test coords[:, 1] ≈ collect(0:(L-1))

    # Test neighbor list generation
    neighbors = generate_neighbor_list(geometry, 1.1)
    @test length(neighbors) == L
    @test length(neighbors[1]) == 1  # First site has 1 neighbor (periodic)
    @test length(neighbors[3]) == 2  # Middle site has 2 neighbors
end

@testitem "Square lattice creation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test square lattice creation
    Lx, Ly = 3, 4
    geometry = create_square_lattice(Lx, Ly)

    @test geometry.lattice_type == SQUARE_LATTICE
    @test geometry.dimensions == 2
    @test geometry.L == [Lx, Ly]
    @test geometry.n_sites_total == Lx * Ly
    @test geometry.n_sites_unit_cell == 1
    @test length(geometry.neighbor_vectors) == 8  # 4 NN + 4 NNN

    # Test coordinate generation
    coords = generate_site_coordinates(geometry)
    @test size(coords) == (Lx * Ly, 2)

    # Check that coordinates are properly arranged
    @test coords[1, :] ≈ [0.0, 0.0]  # First site at origin
    @test coords[2, :] ≈ [1.0, 0.0]  # Second site

    # Test neighbor distances
    nn_distances = geometry.neighbor_distances[1:4]  # First 4 are NN
    nnn_distances = geometry.neighbor_distances[5:8]  # Next 4 are NNN
    @test all(nn_distances .≈ 1.0)
    @test all(nnn_distances .≈ sqrt(2.0))
end

@testitem "Triangular lattice creation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test triangular lattice creation
    Lx, Ly = 3, 3
    geometry = create_triangular_lattice(Lx, Ly)

    @test geometry.lattice_type == TRIANGULAR_LATTICE
    @test geometry.dimensions == 2
    @test geometry.L == [Lx, Ly]
    @test geometry.n_sites_total == Lx * Ly
    @test geometry.n_sites_unit_cell == 1
    @test length(geometry.neighbor_vectors) == 6  # 6 nearest neighbors

    # All nearest neighbors should have the same distance
    @test all(geometry.neighbor_distances .≈ 1.0)
end

@testitem "Honeycomb lattice creation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test honeycomb lattice creation
    Lx, Ly = 2, 2
    geometry = create_honeycomb_lattice(Lx, Ly)

    @test geometry.lattice_type == HONEYCOMB_LATTICE
    @test geometry.dimensions == 2
    @test geometry.L == [Lx, Ly]
    @test geometry.n_sites_total == 2 * Lx * Ly  # 2 sites per unit cell
    @test geometry.n_sites_unit_cell == 2
    @test length(geometry.neighbor_vectors) == 3  # 3 nearest neighbors per site
end

@testitem "Kagome lattice creation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test kagome lattice creation
    Lx, Ly = 2, 2
    geometry = create_kagome_lattice(Lx, Ly)

    @test geometry.lattice_type == KAGOME_LATTICE
    @test geometry.dimensions == 2
    @test geometry.L == [Lx, Ly]
    @test geometry.n_sites_total == 3 * Lx * Ly  # 3 sites per unit cell
    @test geometry.n_sites_unit_cell == 3
    @test length(geometry.neighbor_vectors) == 4
end

@testitem "Ladder lattice creation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test ladder lattice creation
    L = 5
    W = 2
    geometry = create_ladder_lattice(L, W)

    @test geometry.lattice_type == LADDER_LATTICE
    @test geometry.dimensions == 2
    @test geometry.L == [L, 1]
    @test geometry.n_sites_total == L * W
    @test geometry.n_sites_unit_cell == W
    @test length(geometry.neighbor_vectors) == 4  # Along and across
end

@testitem "StdFace chain Hamiltonian" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test chain Hubbard model
    L = 4
    t = 1.0
    U = 2.0
    ham, geometry, config = stdface_chain(L, "Hubbard"; t = t, U = U)

    @test ham.n_sites == L
    @test length(ham.coulomb_intra_terms) == L  # U terms on each site
    @test length(ham.transfer_terms) > 0  # Hopping terms

    # Check that all sites have U interaction
    for term in ham.coulomb_intra_terms
        @test term.coefficient == U
    end

    # Check hopping terms
    hopping_found = false
    for term in ham.transfer_terms
        if abs(term.coefficient + t) < 1e-10  # -t coefficient
            hopping_found = true
        end
    end
    @test hopping_found

    @test config.lattice_type == CHAIN_LATTICE
    @test config.model_type == HUBBARD_MODEL
    @test config.t == t
    @test config.U == U
end

@testitem "StdFace square Hamiltonian" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test square lattice Hubbard model
    Lx, Ly = 3, 3
    t = 1.0
    U = 4.0
    t_prime = 0.2
    ham, geometry, config =
        stdface_square(Lx, Ly, "Hubbard"; t = t, U = U, t_prime = t_prime)

    @test ham.n_sites == Lx * Ly
    @test length(ham.coulomb_intra_terms) == Lx * Ly
    @test length(ham.transfer_terms) > 0

    # Should have both nearest and next-nearest neighbor hopping
    t_coeffs = [abs(term.coefficient) for term in ham.transfer_terms]
    @test t in t_coeffs
    @test t_prime in t_coeffs

    @test config.lattice_type == SQUARE_LATTICE
    @test config.t_prime == t_prime
end

@testitem "StdFace triangular spin model" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test triangular lattice spin model
    Lx, Ly = 3, 3
    J = 1.0
    ham, geometry, config = stdface_triangular(Lx, Ly, "Spin"; J = J)

    @test ham.n_sites == Lx * Ly
    @test length(ham.hund_terms) > 0  # Exchange interactions
    @test length(ham.coulomb_intra_terms) == 0  # No U terms for spin model

    # Check exchange interactions
    for term in ham.hund_terms
        @test term.coefficient == J
    end

    @test config.lattice_type == TRIANGULAR_LATTICE
    @test config.model_type == SPIN_MODEL
    @test config.J == J
end

@testitem "StdFace honeycomb Hamiltonian" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test honeycomb lattice
    Lx, Ly = 2, 2
    t = 1.0
    U = 3.0
    ham, geometry, config = stdface_honeycomb(Lx, Ly, "Hubbard"; t = t, U = U)

    @test ham.n_sites == 2 * Lx * Ly  # 2 sites per unit cell
    @test length(ham.coulomb_intra_terms) == 2 * Lx * Ly
    @test length(ham.transfer_terms) > 0

    @test config.lattice_type == HONEYCOMB_LATTICE
end

@testitem "StdFace ladder Hamiltonian" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test ladder lattice
    L = 4
    W = 2
    t = 1.0
    t_perp = 0.8
    U = 2.0
    ham, geometry, config = stdface_ladder(L, W, "Hubbard"; t = t, t_perp = t_perp, U = U)

    @test ham.n_sites == L * W
    @test length(ham.coulomb_intra_terms) == L * W
    @test length(ham.transfer_terms) > 0

    @test config.lattice_type == LADDER_LATTICE
end

@testitem "Neighbor list generation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test neighbor list for small square lattice
    geometry = create_square_lattice(3, 3)
    neighbors = generate_neighbor_list(geometry, 1.1)  # Only nearest neighbors

    @test length(neighbors) == 9  # 3x3 = 9 sites

    # Corner sites should have 2 neighbors (with periodic BC)
    # Edge sites should have 3 neighbors
    # Center site should have 4 neighbors
    neighbor_counts = [length(n) for n in neighbors]
    @test minimum(neighbor_counts) >= 2
    @test maximum(neighbor_counts) <= 4

    # Test with larger distance to include next-nearest neighbors
    neighbors_nnn = generate_neighbor_list(geometry, 1.5)
    nnn_counts = [length(n) for n in neighbors_nnn]
    @test all(nnn_counts .>= neighbor_counts)  # Should have more neighbors
end

@testitem "Site coordinate generation" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test coordinate generation for different lattices

    # Chain
    geometry = create_chain_lattice(5)
    coords = generate_site_coordinates(geometry)
    @test size(coords) == (5, 1)
    @test coords[:, 1] ≈ [0, 1, 2, 3, 4]

    # Square
    geometry = create_square_lattice(2, 3)
    coords = generate_site_coordinates(geometry)
    @test size(coords) == (6, 2)

    # Check that all coordinates are unique
    for i = 1:size(coords, 1)
        for j = (i+1):size(coords, 1)
            @test norm(coords[i, :] - coords[j, :]) > 1e-10
        end
    end
end

@testitem "StdFace configuration parameters" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    # Test configuration with multiple parameters
    L = 6
    ham, geometry, config = stdface_chain(L, "Hubbard"; t = 1.5, U = 3.0, V = 0.5, mu = 0.2)

    @test config.t == 1.5
    @test config.U == 3.0
    @test config.V == 0.5
    @test config.mu == 0.2

    # Test that V parameter creates inter-site Coulomb terms
    @test length(ham.coulomb_inter_terms) > 0

    # Check V coefficient
    v_found = false
    for term in ham.coulomb_inter_terms
        if abs(term.coefficient - 0.5) < 1e-10
            v_found = true
        end
    end
    @test v_found
end

@testitem "Lattice summary output" begin
    using ManyVariableVariationalMonteCarlo

    # Test that lattice summary doesn't error
    geometry = create_square_lattice(3, 4)
    lattice_summary(geometry)  # Should print without error

    geometry = create_triangular_lattice(4, 4)
    lattice_summary(geometry)  # Should also work

    geometry = create_honeycomb_lattice(2, 3)
    lattice_summary(geometry)  # Multi-site unit cell
    @test true
end
