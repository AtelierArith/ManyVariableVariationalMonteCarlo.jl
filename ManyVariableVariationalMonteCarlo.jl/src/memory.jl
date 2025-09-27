"""
Memory management module for ManyVariableVariationalMonteCarlo.jl

Provides efficient memory allocation and management patterns
ported from the C reference implementation (setmemory.c, workspace.c).
"""

using LinearAlgebra

"""
    Workspace{T}

Manages preallocated workspace arrays for performance-critical computations.
Thread-safe workspace pools to avoid allocations in hot loops.
"""
mutable struct Workspace{T}
    data::Vector{T}
    size::Int
    position::Int

    function Workspace{T}(initial_size::Int = 1024) where {T}
        new{T}(Vector{T}(undef, initial_size), initial_size, 0)
    end
end

"""
    WorkspaceManager

Thread-safe management of multiple workspace pools for different data types.
"""
struct WorkspaceManager
    int_workspaces::Vector{Workspace{Int}}
    double_workspaces::Vector{Workspace{Float64}}
    complex_workspaces::Vector{Workspace{ComplexF64}}
    lock::ReentrantLock

    function WorkspaceManager(n_threads::Int = Threads.nthreads())
        int_ws = [Workspace{Int}() for _ = 1:n_threads]
        double_ws = [Workspace{Float64}() for _ = 1:n_threads]
        complex_ws = [Workspace{ComplexF64}() for _ = 1:n_threads]
        new(int_ws, double_ws, complex_ws, ReentrantLock())
    end
end

const GLOBAL_WORKSPACE = WorkspaceManager()

"""
    allocate_workspace!(ws::Workspace{T}, size::Int) where T

Allocate workspace memory of specified size, expanding if necessary.
Returns a view into the workspace data.
"""
function allocate_workspace!(ws::Workspace{T}, size::Int) where {T}
    if ws.position + size > ws.size
        # Expand workspace by factor of 2 or required size, whichever is larger
        new_size = max(ws.size * 2, ws.position + size)
        resize!(ws.data, new_size)
        ws.size = new_size
    end

    start_pos = ws.position + 1
    ws.position += size
    return view(ws.data, start_pos:(start_pos+size-1))
end

"""
    reset_workspace!(ws::Workspace)

Reset workspace position to beginning without deallocating memory.
"""
function reset_workspace!(ws::Workspace)
    ws.position = 0
end

"""
    get_workspace(::Type{T}, size::Int; thread_id::Int = Threads.threadid()) where T

Get workspace of specified type and size for current thread.
Thread-safe allocation from thread-local workspace pools.
"""
function get_workspace(::Type{Int}, size::Int; thread_id::Int = Threads.threadid())
    lock(GLOBAL_WORKSPACE.lock) do
        ws = GLOBAL_WORKSPACE.int_workspaces[thread_id]
        return allocate_workspace!(ws, size)
    end
end

function get_workspace(::Type{Float64}, size::Int; thread_id::Int = Threads.threadid())
    lock(GLOBAL_WORKSPACE.lock) do
        ws = GLOBAL_WORKSPACE.double_workspaces[thread_id]
        return allocate_workspace!(ws, size)
    end
end

function get_workspace(::Type{ComplexF64}, size::Int; thread_id::Int = Threads.threadid())
    lock(GLOBAL_WORKSPACE.lock) do
        ws = GLOBAL_WORKSPACE.complex_workspaces[thread_id]
        return allocate_workspace!(ws, size)
    end
end

"""
    reset_all_workspaces!()

Reset all workspace positions to beginning. Call between major computation phases.
"""
function reset_all_workspaces!()
    lock(GLOBAL_WORKSPACE.lock) do
        for ws in GLOBAL_WORKSPACE.int_workspaces
            reset_workspace!(ws)
        end
        for ws in GLOBAL_WORKSPACE.double_workspaces
            reset_workspace!(ws)
        end
        for ws in GLOBAL_WORKSPACE.complex_workspaces
            reset_workspace!(ws)
        end
    end
end

"""
    MemoryLayout

Defines memory allocation strategy for VMC global arrays.
Based on the C implementation's SetMemoryDef() function.
"""
struct MemoryLayout
    # System dimensions
    nsite::Int
    ne::Int
    nsize::Int

    # Hamiltonian terms
    ntransfer::Int
    ncoulomb_intra::Int
    ncoulomb_inter::Int
    nhund_coupling::Int
    npair_hopping::Int
    nexchange_coupling::Int

    # Variational parameters
    ngutzwiller::Int
    njastrow::Int
    ndoublon_holon_2site::Int
    ndoublon_holon_4site::Int

    # RBM parameters
    nrbm_hidden::Int
    nrbm_visible::Int

    function MemoryLayout(;
        nsite::Int,
        ne::Int,
        ntransfer::Int = 0,
        ncoulomb_intra::Int = 0,
        ncoulomb_inter::Int = 0,
        nhund_coupling::Int = 0,
        npair_hopping::Int = 0,
        nexchange_coupling::Int = 0,
        ngutzwiller::Int = nsite,
        njastrow::Int = nsite,
        ndoublon_holon_2site::Int = 0,
        ndoublon_holon_4site::Int = 0,
        nrbm_hidden::Int = 0,
        nrbm_visible::Int = 0,
    )
        new(
            nsite,
            ne,
            2 * ne,
            ntransfer,
            ncoulomb_intra,
            ncoulomb_inter,
            nhund_coupling,
            npair_hopping,
            nexchange_coupling,
            ngutzwiller,
            njastrow,
            ndoublon_holon_2site,
            ndoublon_holon_4site,
            nrbm_hidden,
            nrbm_visible,
        )
    end
end

"""
    allocate_global_arrays(layout::MemoryLayout)

Allocate all global arrays according to memory layout.
Returns a named tuple with preallocated arrays.
"""
function allocate_global_arrays(layout::MemoryLayout)
    # Site and spin arrays
    loc_spn = zeros(Int, layout.nsite)

    # Hamiltonian term arrays
    transfer_indices = Matrix{Int}(undef, layout.ntransfer, 4)
    transfer_params = Vector{ComplexF64}(undef, layout.ntransfer)

    coulomb_intra_indices = Vector{Int}(undef, layout.ncoulomb_intra)
    coulomb_intra_params = Vector{Float64}(undef, layout.ncoulomb_intra)

    coulomb_inter_indices = Matrix{Int}(undef, layout.ncoulomb_inter, 2)
    coulomb_inter_params = Vector{Float64}(undef, layout.ncoulomb_inter)

    hund_coupling_indices = Matrix{Int}(undef, layout.nhund_coupling, 2)
    hund_coupling_params = Vector{Float64}(undef, layout.nhund_coupling)

    pair_hopping_indices = Matrix{Int}(undef, layout.npair_hopping, 2)
    pair_hopping_params = Vector{Float64}(undef, layout.npair_hopping)

    exchange_coupling_indices = Matrix{Int}(undef, layout.nexchange_coupling, 2)
    exchange_coupling_params = Vector{Float64}(undef, layout.nexchange_coupling)

    # Variational parameter arrays
    gutzwiller_indices = Vector{Int}(undef, layout.ngutzwiller)
    gutzwiller_params = Vector{ComplexF64}(undef, layout.ngutzwiller)

    jastrow_indices = Matrix{Int}(undef, layout.njastrow, layout.nsite)
    jastrow_params = Matrix{ComplexF64}(undef, layout.njastrow, layout.nsite)

    doublon_holon_2site_indices =
        Matrix{Int}(undef, layout.ndoublon_holon_2site, 2 * layout.nsite)
    doublon_holon_2site_params = Vector{ComplexF64}(undef, layout.ndoublon_holon_2site)

    doublon_holon_4site_indices =
        Matrix{Int}(undef, layout.ndoublon_holon_4site, 4 * layout.nsite)
    doublon_holon_4site_params = Vector{ComplexF64}(undef, layout.ndoublon_holon_4site)

    # RBM arrays
    rbm_hidden_weights = Matrix{ComplexF64}(undef, layout.nrbm_hidden, layout.nrbm_visible)
    rbm_visible_bias = Vector{ComplexF64}(undef, layout.nrbm_visible)
    rbm_hidden_bias = Vector{ComplexF64}(undef, layout.nrbm_hidden)

    return (
        loc_spn = loc_spn,
        transfer_indices = transfer_indices,
        transfer_params = transfer_params,
        coulomb_intra_indices = coulomb_intra_indices,
        coulomb_intra_params = coulomb_intra_params,
        coulomb_inter_indices = coulomb_inter_indices,
        coulomb_inter_params = coulomb_inter_params,
        hund_coupling_indices = hund_coupling_indices,
        hund_coupling_params = hund_coupling_params,
        pair_hopping_indices = pair_hopping_indices,
        pair_hopping_params = pair_hopping_params,
        exchange_coupling_indices = exchange_coupling_indices,
        exchange_coupling_params = exchange_coupling_params,
        gutzwiller_indices = gutzwiller_indices,
        gutzwiller_params = gutzwiller_params,
        jastrow_indices = jastrow_indices,
        jastrow_params = jastrow_params,
        doublon_holon_2site_indices = doublon_holon_2site_indices,
        doublon_holon_2site_params = doublon_holon_2site_params,
        doublon_holon_4site_indices = doublon_holon_4site_indices,
        doublon_holon_4site_params = doublon_holon_4site_params,
        rbm_hidden_weights = rbm_hidden_weights,
        rbm_visible_bias = rbm_visible_bias,
        rbm_hidden_bias = rbm_hidden_bias,
    )
end

"""
    memory_summary(layout::MemoryLayout)

Estimate total memory usage for given layout configuration.
Returns memory usage in MB.
"""
function memory_summary(layout::MemoryLayout)
    int_bytes = sizeof(Int)
    float_bytes = sizeof(Float64)
    complex_bytes = sizeof(ComplexF64)

    total_bytes = 0

    # Site arrays
    total_bytes += layout.nsite * int_bytes  # loc_spn

    # Hamiltonian arrays
    total_bytes += layout.ntransfer * (4 * int_bytes + complex_bytes)
    total_bytes += layout.ncoulomb_intra * (int_bytes + float_bytes)
    total_bytes += layout.ncoulomb_inter * (2 * int_bytes + float_bytes)
    total_bytes += layout.nhund_coupling * (2 * int_bytes + float_bytes)
    total_bytes += layout.npair_hopping * (2 * int_bytes + float_bytes)
    total_bytes += layout.nexchange_coupling * (2 * int_bytes + float_bytes)

    # Variational parameter arrays
    total_bytes += layout.ngutzwiller * (int_bytes + complex_bytes)
    total_bytes += layout.njastrow * layout.nsite * (int_bytes + complex_bytes)
    total_bytes +=
        layout.ndoublon_holon_2site * (2 * layout.nsite * int_bytes + complex_bytes)
    total_bytes +=
        layout.ndoublon_holon_4site * (4 * layout.nsite * int_bytes + complex_bytes)

    # RBM arrays
    total_bytes += layout.nrbm_hidden * layout.nrbm_visible * complex_bytes  # weights
    total_bytes += layout.nrbm_visible * complex_bytes  # visible bias
    total_bytes += layout.nrbm_hidden * complex_bytes   # hidden bias

    return total_bytes / (1024^2)  # Convert to MB
end
