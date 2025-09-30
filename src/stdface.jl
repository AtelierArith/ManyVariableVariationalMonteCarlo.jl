"""
StdFace Standard Lattice Model Generators

Implements standard lattice model generators equivalent to the StdFace functionality
in the C reference implementation. Provides automated generation of Hamiltonian
parameters for common lattice geometries.

Ported from StdFace/*.c in the mVMC C reference implementation.
"""

using LinearAlgebra

"""
    LatticeType

Enumeration of supported lattice types.
"""
@enum LatticeType begin
    CHAIN_LATTICE
    SQUARE_LATTICE
    TRIANGULAR_LATTICE
    HONEYCOMB_LATTICE
    KAGOME_LATTICE
    LADDER_LATTICE
    PYROCHLORE_LATTICE
    ORTHORHOMBIC_LATTICE
end

"""
    ModelType

Enumeration of supported model types.
"""
@enum ModelType begin
    HUBBARD_MODEL
    SPIN_MODEL
    KONDO_MODEL
end

"""
    LatticeGeometry{T}

Represents the geometry of a lattice including unit cell vectors,
site positions, and neighbor relationships.
"""
struct LatticeGeometry{T<:Real}
    # Lattice parameters
    lattice_type::LatticeType
    dimensions::Int
    unit_cell_vectors::Matrix{T}  # dimensions × dimensions

    # Site information
    n_sites_unit_cell::Int
    site_positions::Matrix{T}  # n_sites_unit_cell × dimensions

    # System size
    L::Vector{Int}  # System size in each direction
    n_sites_total::Int

    # Neighbor information
    neighbor_vectors::Vector{Vector{T}}
    neighbor_distances::Vector{T}

    function LatticeGeometry{T}(
        lattice_type::LatticeType,
        dimensions::Int,
        unit_cell_vectors::Matrix{T},
        site_positions::Matrix{T},
        L::Vector{Int},
    ) where {T}
        n_sites_unit_cell = size(site_positions, 1)
        n_sites_total = n_sites_unit_cell * prod(L)

        # Generate neighbor vectors (will be filled by specific lattice functions)
        neighbor_vectors = Vector{T}[]
        neighbor_distances = T[]

        new{T}(
            lattice_type,
            dimensions,
            unit_cell_vectors,
            n_sites_unit_cell,
            site_positions,
            L,
            n_sites_total,
            neighbor_vectors,
            neighbor_distances,
        )
    end
end

"""
    StdFaceConfig{T}

Configuration for StdFace lattice generation.
"""
mutable struct StdFaceConfig{T<:Union{Real,Complex}}
    # Lattice parameters
    lattice_type::LatticeType
    model_type::ModelType
    L::Vector{Int}  # System size

    # Physical parameters
    t::T  # Hopping parameter
    t_prime::T  # Next-nearest neighbor hopping
    U::T  # On-site Coulomb interaction
    V::T  # Nearest-neighbor Coulomb interaction
    V_prime::T  # Next-nearest neighbor Coulomb interaction
    J::T  # Exchange interaction
    J_prime::T  # Next-nearest neighbor exchange
    mu::T  # Chemical potential
    h::T  # Magnetic field

    # Advanced parameters
    boundary_conditions::Vector{Bool}  # true = periodic, false = open
    twist_angles::Vector{Float64}  # Twist angles for boundary conditions

    function StdFaceConfig{T}(
        lattice_type::LatticeType,
        model_type::ModelType,
        L::Vector{Int},
    ) where {T}
        n_dim = length(L)
        new{T}(
            lattice_type,
            model_type,
            L,
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            zero(T),
            fill(true, n_dim),  # Default: periodic boundary conditions
            zeros(Float64, n_dim),  # Default: no twist
        )
    end
end

"""
    create_chain_lattice(L::Int; T=Float64)

Create a 1D chain lattice geometry.
"""
function create_chain_lattice(L::Int; T = Float64)
    unit_cell_vectors = reshape([1.0], 1, 1)
    site_positions = reshape([0.0], 1, 1)

    geometry = LatticeGeometry{T}(CHAIN_LATTICE, 1, unit_cell_vectors, site_positions, [L])

    # Add nearest neighbor vectors
    push!(geometry.neighbor_vectors, [1.0])
    push!(geometry.neighbor_vectors, [-1.0])
    push!(geometry.neighbor_distances, 1.0)
    push!(geometry.neighbor_distances, 1.0)

    return geometry
end

"""
    create_square_lattice(Lx::Int, Ly::Int; T=Float64)

Create a 2D square lattice geometry.
"""
function create_square_lattice(Lx::Int, Ly::Int; T = Float64)
    unit_cell_vectors = T[1.0 0.0; 0.0 1.0]
    site_positions = reshape(T[0.0, 0.0], 1, 2)

    geometry =
        LatticeGeometry{T}(SQUARE_LATTICE, 2, unit_cell_vectors, site_positions, [Lx, Ly])

    # Add nearest neighbor vectors
    push!(geometry.neighbor_vectors, T[1.0, 0.0])   # +x
    push!(geometry.neighbor_vectors, T[-1.0, 0.0])  # -x
    push!(geometry.neighbor_vectors, T[0.0, 1.0])   # +y
    push!(geometry.neighbor_vectors, T[0.0, -1.0])  # -y
    append!(geometry.neighbor_distances, [1.0, 1.0, 1.0, 1.0])

    # Add next-nearest neighbor vectors
    push!(geometry.neighbor_vectors, T[1.0, 1.0])   # +x+y
    push!(geometry.neighbor_vectors, T[1.0, -1.0])  # +x-y
    push!(geometry.neighbor_vectors, T[-1.0, 1.0])  # -x+y
    push!(geometry.neighbor_vectors, T[-1.0, -1.0]) # -x-y
    append!(geometry.neighbor_distances, [sqrt(2.0), sqrt(2.0), sqrt(2.0), sqrt(2.0)])

    return geometry
end

"""
    create_triangular_lattice(Lx::Int, Ly::Int; T=Float64)

Create a 2D triangular lattice geometry.
"""
function create_triangular_lattice(Lx::Int, Ly::Int; T = Float64)
    # Triangular lattice unit cell vectors
    a = 1.0
    unit_cell_vectors = T[a 0.0; a/2 a*sqrt(3)/2]
    site_positions = reshape(T[0.0, 0.0], 1, 2)

    geometry = LatticeGeometry{T}(
        TRIANGULAR_LATTICE,
        2,
        unit_cell_vectors,
        site_positions,
        [Lx, Ly],
    )

    # Add nearest neighbor vectors (6 neighbors)
    push!(geometry.neighbor_vectors, T[1.0, 0.0])                    # →
    push!(geometry.neighbor_vectors, T[-1.0, 0.0])                   # ←
    push!(geometry.neighbor_vectors, T[0.5, sqrt(3)/2])              # ↗
    push!(geometry.neighbor_vectors, T[-0.5, -sqrt(3)/2])            # ↙
    push!(geometry.neighbor_vectors, T[0.5, -sqrt(3)/2])             # ↘
    push!(geometry.neighbor_vectors, T[-0.5, sqrt(3)/2])             # ↖
    append!(geometry.neighbor_distances, fill(1.0, 6))

    return geometry
end

"""
    create_honeycomb_lattice(Lx::Int, Ly::Int; T=Float64)

Create a 2D honeycomb lattice geometry.
"""
function create_honeycomb_lattice(Lx::Int, Ly::Int; T = Float64)
    # Honeycomb lattice unit cell vectors
    a = 1.0
    unit_cell_vectors = T[3*a/2 0.0; 3*a/4 a*sqrt(3)/2]

    # Two sites per unit cell (A and B sublattices)
    site_positions = T[0.0 0.0; a/2 a*sqrt(3)/6]

    geometry = LatticeGeometry{T}(
        HONEYCOMB_LATTICE,
        2,
        unit_cell_vectors,
        site_positions,
        [Lx, Ly],
    )

    # Add nearest neighbor vectors (3 neighbors per site)
    # From A sublattice to B sublattice
    push!(geometry.neighbor_vectors, T[a/2, a*sqrt(3)/6])
    push!(geometry.neighbor_vectors, T[-a/2, a*sqrt(3)/6])
    push!(geometry.neighbor_vectors, T[0.0, -a*sqrt(3)/3])
    append!(geometry.neighbor_distances, fill(a*sqrt(3)/3, 3))

    return geometry
end

"""
    create_kagome_lattice(Lx::Int, Ly::Int; T=Float64)

Create a 2D kagome lattice geometry.
"""
function create_kagome_lattice(Lx::Int, Ly::Int; T = Float64)
    # Kagome lattice unit cell vectors
    a = 1.0
    unit_cell_vectors = T[2*a 0.0; a a*sqrt(3)]

    # Three sites per unit cell
    site_positions = T[
        0.0 0.0;
        a/2 a*sqrt(3)/2;
        3*a/2 a*sqrt(3)/2
    ]

    geometry =
        LatticeGeometry{T}(KAGOME_LATTICE, 2, unit_cell_vectors, site_positions, [Lx, Ly])

    # Add nearest neighbor vectors (complex connectivity)
    # Each site has 4 nearest neighbors
    push!(geometry.neighbor_vectors, T[a/2, a*sqrt(3)/2])
    push!(geometry.neighbor_vectors, T[-a/2, a*sqrt(3)/2])
    push!(geometry.neighbor_vectors, T[a, 0.0])
    push!(geometry.neighbor_vectors, T[-a, 0.0])
    append!(geometry.neighbor_distances, fill(a, 4))

    return geometry
end

"""
    create_ladder_lattice(L::Int, W::Int; T=Float64)

Create a ladder lattice geometry.
"""
function create_ladder_lattice(L::Int, W::Int = 2; T = Float64)
    unit_cell_vectors = T[1.0 0.0; 0.0 1.0]

    # W sites per unit cell (rungs of the ladder)
    site_positions = zeros(T, W, 2)
    for i = 1:W
        site_positions[i, :] = [0.0, (i-1)]
    end

    geometry = LatticeGeometry{T}(
        LADDER_LATTICE,
        2,
        unit_cell_vectors,
        site_positions,
        [L, 1],  # L rungs, 1 unit cell in y direction
    )

    # Add nearest neighbor vectors
    # Along the ladder (x-direction)
    push!(geometry.neighbor_vectors, T[1.0, 0.0])
    push!(geometry.neighbor_vectors, T[-1.0, 0.0])
    # Across the rungs (y-direction)
    push!(geometry.neighbor_vectors, T[0.0, 1.0])
    push!(geometry.neighbor_vectors, T[0.0, -1.0])
    append!(geometry.neighbor_distances, [1.0, 1.0, 1.0, 1.0])

    return geometry
end

"""
    generate_site_coordinates(geometry::LatticeGeometry{T}) where {T}

Generate coordinates for all sites in the lattice.
"""
function generate_site_coordinates(geometry::LatticeGeometry{T}) where {T}
    coords = Matrix{T}(undef, geometry.n_sites_total, geometry.dimensions)
    site_idx = 1

    # Generate coordinates for each unit cell
    for indices in Iterators.product([0:(L-1) for L in geometry.L]...)
        unit_cell_origin = sum(collect(indices) .* eachrow(geometry.unit_cell_vectors))

        for uc_site = 1:geometry.n_sites_unit_cell
            site_coord = unit_cell_origin + geometry.site_positions[uc_site, :]
            coords[site_idx, :] = site_coord
            site_idx += 1
        end
    end

    return coords
end

"""
    generate_neighbor_list(geometry::LatticeGeometry{T}, max_distance::T=2.0) where {T}

Generate neighbor list for all sites up to maximum distance.
"""
function generate_neighbor_list(
    geometry::LatticeGeometry{T},
    max_distance::T = 2.0,
) where {T}
    coords = generate_site_coordinates(geometry)
    neighbor_list = Vector{Vector{Int}}(undef, geometry.n_sites_total)

    for i = 1:geometry.n_sites_total
        neighbors = Int[]

        for j = 1:geometry.n_sites_total
            if i != j
                distance = norm(coords[i, :] - coords[j, :])
                if distance <= max_distance + 1e-10  # Small tolerance for numerical errors
                    push!(neighbors, j)
                end
            end
        end

        neighbor_list[i] = neighbors
    end

    return neighbor_list
end

"""
    create_stdface_hamiltonian(config::StdFaceConfig{T}, geometry::LatticeGeometry{T}) where {T}

Create a Hamiltonian based on StdFace configuration and lattice geometry.
"""
function create_stdface_hamiltonian(
    config::StdFaceConfig{T},
    geometry::LatticeGeometry{T},
) where {T}
    ham = Hamiltonian{T}(geometry.n_sites_total, 0)  # Will set n_electrons later
    coords = generate_site_coordinates(geometry)

    # Add hopping terms
    if config.model_type == HUBBARD_MODEL
        add_hubbard_terms!(ham, config, geometry, coords)
    elseif config.model_type == SPIN_MODEL
        add_spin_terms!(ham, config, geometry, coords)
    end

    return ham
end

"""
    add_hubbard_terms!(ham::Hamiltonian{T}, config::StdFaceConfig{T},
                      geometry::LatticeGeometry{T}, coords::Matrix{T}) where {T}

Add Hubbard model terms to the Hamiltonian.
"""
function add_hubbard_terms!(
    ham::Hamiltonian{T},
    config::StdFaceConfig{T},
    geometry::LatticeGeometry{T},
    coords::Matrix{T},
) where {T}
    n_sites = geometry.n_sites_total

    # On-site Coulomb interaction
    if abs(config.U) > 1e-10
        for i = 1:n_sites
            add_coulomb_intra!(ham, config.U, i)
        end
    end

    # Hopping terms
    if abs(config.t) > 1e-10
        neighbor_list = generate_neighbor_list(geometry, 1.1)  # Nearest neighbors

        for i = 1:n_sites
            for j in neighbor_list[i]
                if i < j  # Avoid double counting
                    # Add hopping for both spins
                    add_transfer!(ham, -config.t, i, 0, j, 0)  # spin up
                    add_transfer!(ham, -config.t, i, 1, j, 1)  # spin down
                end
            end
        end
    end

    # Next-nearest neighbor hopping
    if abs(config.t_prime) > 1e-10
        nnn_neighbor_list = generate_neighbor_list(geometry, 1.5)  # Next-nearest neighbors
        nn_neighbor_list = generate_neighbor_list(geometry, 1.1)   # Nearest neighbors

        for i = 1:n_sites
            for j in nnn_neighbor_list[i]
                if i < j && !(j in nn_neighbor_list[i])  # Next-nearest but not nearest
                    add_transfer!(ham, -config.t_prime, i, 0, j, 0)  # spin up
                    add_transfer!(ham, -config.t_prime, i, 1, j, 1)  # spin down
                end
            end
        end
    end

    # Nearest-neighbor Coulomb interaction
    if abs(config.V) > 1e-10
        neighbor_list = generate_neighbor_list(geometry, 1.1)

        for i = 1:n_sites
            for j in neighbor_list[i]
                if i < j
                    add_coulomb_inter!(ham, config.V, i, j)
                end
            end
        end
    end
end

"""
    add_spin_terms!(ham::Hamiltonian{T}, config::StdFaceConfig{T},
                   geometry::LatticeGeometry{T}, coords::Matrix{T}) where {T}

Add spin model terms to the Hamiltonian.
"""
function add_spin_terms!(
    ham::Hamiltonian{T},
    config::StdFaceConfig{T},
    geometry::LatticeGeometry{T},
    coords::Matrix{T},
) where {T}
    # Exchange interaction
    if abs(config.J) > 1e-10
        neighbor_list = generate_neighbor_list(geometry, 1.1)

        for i = 1:geometry.n_sites_total
            for j in neighbor_list[i]
                if i < j
                    add_hund_coupling!(ham, config.J, i, j)
                end
            end
        end
    end
end

"""
    create_heisenberg_hamiltonian(geometry::LatticeGeometry{S}, J::T) where {S,T}

Construct a Heisenberg Hamiltonian from a lattice `geometry` using
nearest-neighbor bonds inferred from the geometry's coordinates.
This supports arbitrary lattices for which `generate_neighbor_list`
returns sensible nearest neighbors (chain, square, triangular, honeycomb,
ladder, kagome, ...).
"""
function create_heisenberg_hamiltonian(
    geometry::LatticeGeometry{S},
    J::T,
) where {S<:Real,T<:Number}
    n = geometry.n_sites_total
    ham = Hamiltonian{T}(n, n)
    neighbors = generate_neighbor_list(geometry, S(1.1))
    for i = 1:n
        for j in neighbors[i]
            if i < j
                add_hund_coupling!(ham, J, i, j)
            end
        end
    end
    return ham
end

"""
    stdface_chain(L::Int, model::String; kwargs...)

Create a chain lattice with standard parameters.
"""
function stdface_chain(
    L::Int,
    model::String = "Hubbard";
    t = 1.0,
    U = 0.0,
    V = 0.0,
    J = 0.0,
    kwargs...,
)
    geometry = create_chain_lattice(L)

    model_type =
        model == "Hubbard" ? HUBBARD_MODEL : model == "Spin" ? SPIN_MODEL : HUBBARD_MODEL

    config = StdFaceConfig{typeof(t)}(CHAIN_LATTICE, model_type, [L])
    config.t = t
    config.U = U
    config.V = V
    config.J = J

    # Apply additional parameters
    for (key, value) in kwargs
        if hasfield(StdFaceConfig, key)
            setfield!(config, key, value)
        end
    end

    ham = create_stdface_hamiltonian(config, geometry)

    return ham, geometry, config
end

"""
    stdface_square(Lx::Int, Ly::Int, model::String="Hubbard"; kwargs...)

Create a square lattice with standard parameters.
"""
function stdface_square(
    Lx::Int,
    Ly::Int,
    model::String = "Hubbard";
    t = 1.0,
    U = 0.0,
    V = 0.0,
    t_prime = 0.0,
    kwargs...,
)
    geometry = create_square_lattice(Lx, Ly)

    model_type =
        model == "Hubbard" ? HUBBARD_MODEL : model == "Spin" ? SPIN_MODEL : HUBBARD_MODEL

    config = StdFaceConfig{typeof(t)}(SQUARE_LATTICE, model_type, [Lx, Ly])
    config.t = t
    config.U = U
    config.V = V
    config.t_prime = t_prime

    # Apply additional parameters
    for (key, value) in kwargs
        if hasfield(StdFaceConfig, key)
            setfield!(config, key, value)
        end
    end

    ham = create_stdface_hamiltonian(config, geometry)

    return ham, geometry, config
end

"""
    stdface_triangular(Lx::Int, Ly::Int, model::String="Hubbard"; kwargs...)

Create a triangular lattice with standard parameters.
"""
function stdface_triangular(
    Lx::Int,
    Ly::Int,
    model::String = "Hubbard";
    t = 1.0,
    U = 0.0,
    V = 0.0,
    J = 0.0,
    kwargs...,
)
    geometry = create_triangular_lattice(Lx, Ly)

    model_type =
        model == "Hubbard" ? HUBBARD_MODEL : model == "Spin" ? SPIN_MODEL : HUBBARD_MODEL

    config = StdFaceConfig{typeof(t)}(TRIANGULAR_LATTICE, model_type, [Lx, Ly])
    config.t = t
    config.U = U
    config.V = V
    config.J = J

    # Apply additional parameters
    for (key, value) in kwargs
        if hasfield(StdFaceConfig, key)
            setfield!(config, key, value)
        end
    end

    ham = create_stdface_hamiltonian(config, geometry)

    return ham, geometry, config
end

"""
    stdface_honeycomb(Lx::Int, Ly::Int, model::String="Hubbard"; kwargs...)

Create a honeycomb lattice with standard parameters.
"""
function stdface_honeycomb(
    Lx::Int,
    Ly::Int,
    model::String = "Hubbard";
    t = 1.0,
    U = 0.0,
    V = 0.0,
    kwargs...,
)
    geometry = create_honeycomb_lattice(Lx, Ly)

    model_type =
        model == "Hubbard" ? HUBBARD_MODEL : model == "Spin" ? SPIN_MODEL : HUBBARD_MODEL

    config = StdFaceConfig{typeof(t)}(HONEYCOMB_LATTICE, model_type, [Lx, Ly])
    config.t = t
    config.U = U
    config.V = V

    # Apply additional parameters
    for (key, value) in kwargs
        if hasfield(StdFaceConfig, key)
            setfield!(config, key, value)
        end
    end

    ham = create_stdface_hamiltonian(config, geometry)

    return ham, geometry, config
end

"""
    stdface_kagome(Lx::Int, Ly::Int, model::String="Hubbard"; kwargs...)

Create a kagome lattice with standard parameters.
"""
function stdface_kagome(
    Lx::Int,
    Ly::Int,
    model::String = "Hubbard";
    t = 1.0,
    U = 0.0,
    V = 0.0,
    J = 0.0,
    kwargs...,
)
    geometry = create_kagome_lattice(Lx, Ly)

    model_type =
        model == "Hubbard" ? HUBBARD_MODEL : model == "Spin" ? SPIN_MODEL : HUBBARD_MODEL

    config = StdFaceConfig{typeof(t)}(KAGOME_LATTICE, model_type, [Lx, Ly])
    config.t = t
    config.U = U
    config.V = V
    config.J = J

    # Apply additional parameters
    for (key, value) in kwargs
        if hasfield(StdFaceConfig, key)
            setfield!(config, key, value)
        end
    end

    ham = create_stdface_hamiltonian(config, geometry)

    return ham, geometry, config
end

"""
    stdface_ladder(L::Int, W::Int=2, model::String="Hubbard"; kwargs...)

Create a ladder lattice with standard parameters.
"""
function stdface_ladder(
    L::Int,
    W::Int = 2,
    model::String = "Hubbard";
    t = 1.0,
    t_perp = 1.0,
    U = 0.0,
    V = 0.0,
    kwargs...,
)
    geometry = create_ladder_lattice(L, W)

    model_type =
        model == "Hubbard" ? HUBBARD_MODEL : model == "Spin" ? SPIN_MODEL : HUBBARD_MODEL

    config = StdFaceConfig{typeof(t)}(LADDER_LATTICE, model_type, [L, 1])
    config.t = t
    config.U = U
    config.V = V

    # Apply additional parameters
    for (key, value) in kwargs
        if hasfield(StdFaceConfig, key)
            setfield!(config, key, value)
        end
    end

    ham = create_stdface_hamiltonian(config, geometry)

    # Add perpendicular hopping for ladder
    if abs(t_perp) > 1e-10 && W > 1
        for i = 1:L
            for w = 1:(W-1)
                site1 = (i-1)*W + w
                site2 = (i-1)*W + w + 1
                add_transfer!(ham, -t_perp, site1, 0, site2, 0)  # spin up
                add_transfer!(ham, -t_perp, site1, 1, site2, 1)  # spin down
            end
        end
    end

    return ham, geometry, config
end

"""
    lattice_summary(geometry::LatticeGeometry{T}) where {T}

Print a summary of the lattice geometry.
"""
function lattice_summary(geometry::LatticeGeometry{T}) where {T}
    println("Lattice Geometry Summary:")
    println("  Type: $(geometry.lattice_type)")
    println("  Dimensions: $(geometry.dimensions)")
    println("  System size: $(geometry.L)")
    println("  Sites per unit cell: $(geometry.n_sites_unit_cell)")
    println("  Total sites: $(geometry.n_sites_total)")
    println("  Neighbor vectors: $(length(geometry.neighbor_vectors))")

    if !isempty(geometry.neighbor_distances)
        unique_distances = unique(round.(geometry.neighbor_distances, digits = 6))
        println("  Neighbor distances: $(unique_distances)")
    end
end
