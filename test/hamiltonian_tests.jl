# ReTestItems does not allow using statements outside of @testitem blocks

@testitem "Hamiltonian basic construction" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    ham = Hamiltonian(4, 2)  # 4 sites, 2 electrons

    @test ham.n_sites == 4
    @test ham.n_electrons == 2
    @test isempty(ham.transfer_terms)
    @test isempty(ham.coulomb_intra_terms)
    @test ham.kinetic_matrix === nothing
    @test isempty(ham.potential_cache)
end

@testitem "Hamiltonian term addition" begin
    ham = Hamiltonian{ComplexF64}(4, 2)

    # Add transfer terms
    add_transfer!(ham, -1.0, 1, 0, 2, 0)  # Hopping between sites 1 and 2, spin up
    add_transfer!(ham, -1.0, 2, 1, 3, 1)  # Hopping between sites 2 and 3, spin down

    @test length(ham.transfer_terms) == 2
    @test ham.transfer_terms[1].coefficient == -1.0
    @test ham.transfer_terms[1].site_i == 1
    @test ham.transfer_terms[1].spin_i == 0
    @test ham.transfer_terms[1].site_j == 2
    @test ham.transfer_terms[1].spin_j == 0

    # Add Coulomb terms
    add_coulomb_intra!(ham, 4.0, 1)  # On-site interaction at site 1
    add_coulomb_inter!(ham, 1.0, 1, 2)  # Inter-site interaction between sites 1 and 2

    @test length(ham.coulomb_intra_terms) == 1
    @test length(ham.coulomb_inter_terms) == 1
    @test ham.coulomb_intra_terms[1].coefficient == 4.0
    @test ham.coulomb_intra_terms[1].site == 1

    # Add other interaction terms
    add_hund_coupling!(ham, 0.5, 1, 2)
    add_pair_hopping!(ham, 0.2, 2, 3)
    add_exchange!(ham, 0.1, 3, 4)
    add_interall!(ham, 0.05, 1, 0, 2, 1, 3, 0, 4, 1)

    @test length(ham.hund_terms) == 1
    @test length(ham.pair_hopping_terms) == 1
    @test length(ham.exchange_terms) == 1
    @test length(ham.interall_terms) == 1
end

@testitem "Hubbard Hamiltonian creation" begin
    # 1D Hubbard chain
    n_sites = 4
    n_electrons = 2
    t = 1.0
    U = 4.0

    ham = create_hubbard_hamiltonian(n_sites, n_electrons, t, U; lattice_type = :chain)

    @test ham.n_sites == 4
    @test ham.n_electrons == 2

    # Should have on-site Coulomb terms for each site
    @test length(ham.coulomb_intra_terms) == 4
    for term in ham.coulomb_intra_terms
        @test term.coefficient == U
    end

    # Should have hopping terms (including periodic boundary conditions)
    # 2 spins × 2 directions per bond × 4 bonds (including PBC) = 16 terms
    @test length(ham.transfer_terms) == 16

    # Check some specific hopping terms
    found_hopping = false
    for term in ham.transfer_terms
        if term.site_i == 1 && term.site_j == 2 && term.spin_i == 0 && term.spin_j == 0
            @test term.coefficient == -t
            found_hopping = true
        end
    end
    @test found_hopping
end

@testitem "Heisenberg Hamiltonian creation" begin
    n_sites = 6
    J = 1.0

    ham = create_heisenberg_hamiltonian(n_sites, J; lattice_type = :chain)

    @test ham.n_sites == 6
    @test ham.n_electrons == 6  # One electron per site

    # Should have Hund coupling terms for nearest neighbors
    @test length(ham.hund_terms) == 6  # 5 bonds + 1 PBC

    for term in ham.hund_terms
        @test term.coefficient == J
    end
end

@testitem "Hamiltonian energy calculation - simple cases" begin
    # Test with a simple 2-site system
    ham = Hamiltonian{Float64}(2, 2)
    add_coulomb_intra!(ham, 4.0, 1)
    add_coulomb_intra!(ham, 4.0, 2)

    # Configuration: one electron on each site, different spins
    electron_config = [1, 2]  # Positions
    electron_numbers = [1, 1, 0, 0]  # [n1↑, n2↑, n1↓, n2↓]

    energy = calculate_hamiltonian(ham, electron_config, electron_numbers)
    @test energy == 0.0  # No double occupation, so no Coulomb energy

    # Configuration with double occupation
    electron_numbers = [1, 0, 1, 0]  # Both electrons on site 1
    energy = calculate_hamiltonian(ham, electron_config, electron_numbers)
    @test energy == 4.0  # One doubly occupied site
end

@testitem "Double occupation calculation" begin
    # Test double occupation calculation
    n_sites = 4

    # Case 1: No double occupation
    electron_numbers = [1, 1, 0, 0, 0, 0, 1, 1]  # [n1↑, n2↑, n3↑, n4↑, n1↓, n2↓, n3↓, n4↓]
    double_occ = calculate_double_occupation(electron_numbers, n_sites)
    @test double_occ == 0.0

    # Case 2: One doubly occupied site
    electron_numbers = [1, 1, 0, 0, 1, 0, 0, 0]  # Site 1 has both spins
    double_occ = calculate_double_occupation(electron_numbers, n_sites)
    @test double_occ == 1.0

    # Case 3: Two doubly occupied sites
    electron_numbers = [1, 1, 0, 0, 1, 1, 0, 0]  # Sites 1 and 2 have both spins
    double_occ = calculate_double_occupation(electron_numbers, n_sites)
    @test double_occ == 2.0
end

@testitem "Kinetic matrix construction" begin
    using ManyVariableVariationalMonteCarlo
    using LinearAlgebra

    ham = Hamiltonian{ComplexF64}(3, 2)

    # Add some transfer terms - build_kinetic_matrix! automatically adds hermitian conjugates
    # So we only add one direction to avoid double counting
    add_transfer!(ham, -1.0, 1, 0, 2, 0)  # 1↑ -> 2↑ (will also create 2↑ -> 1↑)
    add_transfer!(ham, -0.5, 1, 1, 3, 1)  # 1↓ -> 3↓ (will also create 3↓ -> 1↓)

    matrix = build_kinetic_matrix!(ham)

    @test size(matrix) == (6, 6)  # 2 * n_sites
    @test matrix[1, 2] ≈ -1.0  # 1↑ -> 2↑
    @test matrix[2, 1] ≈ -1.0  # 2↑ -> 1↑ (hermitian conjugate)
    @test matrix[4, 6] ≈ -0.5  # 1↓ -> 3↓ (indices: 1+3, 3+3)
    @test matrix[6, 4] ≈ -0.5  # 3↓ -> 1↓ (hermitian conjugate)

    # Test caching
    matrix2 = build_kinetic_matrix!(ham)
    @test matrix === matrix2  # Should return cached version
end

@testitem "Coulomb interaction energies" begin
    ham = Hamiltonian{Float64}(3, 4)

    # Add Coulomb terms
    add_coulomb_intra!(ham, 4.0, 1)
    add_coulomb_intra!(ham, 4.0, 2)
    add_coulomb_inter!(ham, 1.0, 1, 2)
    add_coulomb_inter!(ham, 0.5, 2, 3)

    # Test configuration: [n1↑, n2↑, n3↑, n1↓, n2↓, n3↓]
    electron_numbers = [1, 1, 0, 1, 0, 1]

    # Calculate individual contributions
    intra_energy = calculate_coulomb_intra_energy(ham, electron_numbers)
    @test intra_energy == 4.0  # Only site 1 is doubly occupied

    inter_energy = calculate_coulomb_inter_energy(ham, electron_numbers)
    # Total occupations: n1=2, n2=1, n3=1
    # Inter-site terms: 1.0 * n1 * n2 + 0.5 * n2 * n3 = 1.0 * 2 * 1 + 0.5 * 1 * 1 = 2.5
    @test inter_energy == 2.5
end

@testitem "Hund coupling energy" begin
    ham = Hamiltonian{Float64}(3, 4)
    add_hund_coupling!(ham, 1.0, 1, 2)

    # Configuration with parallel spins: n1↑=1, n1↓=0, n2↑=1, n2↓=0
    electron_numbers = [1, 1, 0, 0, 0, 1]

    hund_energy = calculate_hund_energy(ham, electron_numbers)
    # Hund term: n1↑*n2↑ + n1↓*n2↓ - n1↑*n2↓ - n1↓*n2↑ - (n1↑+n1↓)*(n2↑+n2↓)/4
    # = 1*1 + 0*0 - 1*0 - 0*1 - (1+0)*(1+0)/4 = 1 - 0.25 = 0.75
    @test hund_energy ≈ 0.75
end

@testitem "Hamiltonian summary output" begin
    ham = create_hubbard_hamiltonian(4, 2, 1.0, 4.0; lattice_type = :chain)

    # Test that summary doesn't error
    hamiltonian_summary(ham)  # Should print without error

    # Test with complex Hamiltonian
    ham_complex = Hamiltonian{ComplexF64}(6, 3)
    add_transfer!(ham_complex, 1.0 + 0.5im, 1, 0, 2, 0)
    add_coulomb_intra!(ham_complex, 2.0, 1)

    hamiltonian_summary(ham_complex)  # Should also work
end

@testitem "Square lattice Hamiltonian" begin
    # Test 2x2 square lattice
    n_sites = 4
    ham = create_hubbard_hamiltonian(n_sites, 2, 1.0, 4.0; lattice_type = :square)

    @test ham.n_sites == 4
    @test length(ham.coulomb_intra_terms) == 4

    # Should have hopping terms for square lattice connectivity
    @test length(ham.transfer_terms) > 0
end

@testitem "Square lattice with explicit Lx,Ly" begin
    using ManyVariableVariationalMonteCarlo
    # Choose a non-square-friendly n_sites = 15; set Lx=3, Ly=5
    n_sites = 15
    Lx, Ly = 3, 5
    ham = create_hubbard_hamiltonian(n_sites, 6, 1.0, 2.0; lattice_type = :square, Lx=Lx, Ly=Ly)

    @test ham.n_sites == n_sites
    # One U term per site
    @test length(ham.coulomb_intra_terms) == n_sites
    # For square with PBC: number of undirected NN bonds = 2*Lx*Ly; per bond 2 spins x 2 directions = 4
    @test length(ham.transfer_terms) == 8 * Lx * Ly
end

@testitem "Honeycomb with explicit Lx,Ly" begin
    using ManyVariableVariationalMonteCarlo
    # Honeycomb has 2 sites per unit cell; pick Lx=2, Ly=3 -> n_sites=12
    Lx, Ly = 2, 3
    n_sites = 2 * Lx * Ly
    J = 1.0
    ham = create_heisenberg_hamiltonian(n_sites, J; lattice_type = :honeycomb, Lx=Lx, Ly=Ly)

    @test ham.n_sites == n_sites
    @test ham.n_electrons == n_sites
    # Should have Hund terms added; basic sanity
    @test length(ham.hund_terms) > 0
end

@testitem "Complex coefficient handling" begin
    ham = Hamiltonian{ComplexF64}(2, 2)

    # Add complex transfer term
    add_transfer!(ham, 1.0 + 0.5im, 1, 0, 2, 0)

    @test ham.transfer_terms[1].coefficient == 1.0 + 0.5im

    # Test energy calculation with complex coefficients
    electron_config = [1, 2]
    electron_numbers = [1, 0, 0, 1]

    energy = calculate_hamiltonian(ham, electron_config, electron_numbers)
    @test typeof(energy) == ComplexF64
end
