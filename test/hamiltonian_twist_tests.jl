@testitem "hamiltonian: APBC and twist phases on boundary wraps" begin
    using ManyVariableVariationalMonteCarlo

    # Chain, n=4; check boundary wrap terms (1 <-> 4) phases
    n = 4
    ne = 2
    t = 1.0
    U = 0.0

    # No APBC, no twist: boundary coefficient should be -t (real)
    ham0 = create_hubbard_hamiltonian(n, ne, t, U; lattice_type = :chain, apbc = false)
    # APBC: boundary coefficient should be +t (since -t * (-1) = +t)
    ham_ap = create_hubbard_hamiltonian(n, ne, t, U; lattice_type = :chain, apbc = true)
    # Twist pi/2: boundary coefficient should be -t * exp(i*pi/2) = -i t
    ham_tw = create_hubbard_hamiltonian(
        n,
        ne,
        t,
        U;
        lattice_type = :chain,
        apbc = false,
        twist_x = pi/2,
    )

    # Extract boundary terms for spin up (spin=0) and check coefficients
    function get_coeff(ham, i, j, spin)
        for term in ham.transfer_terms
            if term.site_i == i &&
               term.site_j == j &&
               term.spin_i == spin &&
               term.spin_j == spin
                return term.coefficient
            end
        end
        return nothing
    end

    c0 = get_coeff(ham0, 1, n, 0)
    cap = get_coeff(ham_ap, 1, n, 0)
    ctw = get_coeff(ham_tw, 1, n, 0)

    @test c0 !== nothing && cap !== nothing && ctw !== nothing
    @test c0 ≈ -t
    @test cap ≈ +t
    @test isapprox(real(ctw), 0.0; atol = 1e-9) && isapprox(imag(ctw), -t; atol = 1e-9)

    # Square lattice wrap phases (2x2): check x- and y- wraps exist
    n2 = 4
    ham_sq = create_hubbard_hamiltonian(
        n2,
        ne,
        t,
        U;
        lattice_type = :square,
        apbc = false,
        twist_x = pi/3,
        twist_y = pi/4,
    )
    # Just ensure the Hamiltonian builds and contains transfer terms
    @test length(ham_sq.transfer_terms) > 0
end
