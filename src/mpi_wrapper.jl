"""
MPI Wrapper for mVMC C Compatibility

Provides MPI functionality matching the C implementation's usage patterns,
including communicator management, collective operations, and MultiDef mode support.

Ported from vmcmain.c and related C modules.
"""

# Check if MPI.jl is available, otherwise provide fallback
try
    using MPI
    const MPI_AVAILABLE = true
catch
    const MPI_AVAILABLE = false
    # Fallback definitions for when MPI is not available
    const MPI_COMM_WORLD = 0
    const MPI_SUCCESS = 0
    const MPI_INT = 0
    const MPI_DOUBLE = 0
    const MPI_DOUBLE_COMPLEX = 0
    const MPI_SUM = 0
    const MPI_MAX = 0
    const MPI_MIN = 0
end

"""
    MVMCMPIState

MPI state container matching the C implementation's MPI usage.
"""
mutable struct MVMCMPIState
    # MPI communicators (matching C variables)
    comm_world::Any  # MPI_Comm
    comm0::Any       # MPI_Comm
    comm1::Any       # MPI_Comm
    comm2::Any       # MPI_Comm

    # MPI process information
    rank_world::Int
    size_world::Int
    rank0::Int
    size0::Int
    rank1::Int
    size1::Int
    rank2::Int
    size2::Int

    # Process group information
    group1::Int
    group2::Int

    # MultiDef mode
    flag_multi_def::Bool
    n_multi_def::Int

    # Standard mode
    flag_standard::Bool

    # File directory for MultiDef mode
    file_dir_list::String

    function MVMCMPIState()
        if MPI_AVAILABLE
            comm_world = MPI.COMM_WORLD
            rank_world = MPI.Comm_rank(comm_world)
            size_world = MPI.Comm_size(comm_world)
        else
            comm_world = MPI_COMM_WORLD
            rank_world = 0
            size_world = 1
        end

        new(
            comm_world,  # comm_world
            comm_world,  # comm0 (initially same as world)
            nothing,     # comm1
            nothing,     # comm2
            rank_world,  # rank_world
            size_world,  # size_world
            rank_world,  # rank0
            size_world,  # size0
            0,           # rank1
            1,           # size1
            0,           # rank2
            1,           # size2
            0,           # group1
            0,           # group2
            false,       # flag_multi_def
            1,           # n_multi_def
            false,       # flag_standard
            ""           # file_dir_list
        )
    end
end

"""
    mpi_init!(state::MVMCMPIState, argc::Int, argv::Vector{String})

Initialize MPI.
Matches C function MPI_Init().

C実装参考: vmcmain.c 1行目から803行目まで
"""
function mpi_init!(state::MVMCMPIState, argc::Int, argv::Vector{String})
    if MPI_AVAILABLE
        MPI.Init()
        state.comm_world = MPI.COMM_WORLD
        state.rank_world = MPI.Comm_rank(state.comm_world)
        state.size_world = MPI.Comm_size(state.comm_world)
        state.rank0 = state.rank_world
        state.size0 = state.size_world
    else
        # Fallback for non-MPI mode
        state.rank_world = 0
        state.size_world = 1
        state.rank0 = 0
        state.size0 = 1
    end
end

"""
    mpi_finalize!(state::MVMCMPIState)

Finalize MPI.
Matches C function MPI_Finalize().
"""
function mpi_finalize!(state::MVMCMPIState)
    if MPI_AVAILABLE
        MPI.Finalize()
    end
end

"""
    mpi_barrier!(state::MVMCMPIState, comm::Any = nothing)

MPI barrier operation.
Matches C function MPI_Barrier().
"""
function mpi_barrier!(state::MVMCMPIState, comm::Any = nothing)
    if MPI_AVAILABLE
        if comm === nothing
            comm = state.comm0
        end
        MPI.Barrier(comm)
    end
end

"""
    mpi_comm_rank(comm::Any) -> Int

Get MPI rank.
Matches C function MPI_Comm_rank().
"""
function mpi_comm_rank(comm::Any)::Int
    if MPI_AVAILABLE
        return MPI.Comm_rank(comm)
    else
        return 0
    end
end

"""
    mpi_comm_size(comm::Any) -> Int

Get MPI size.
Matches C function MPI_Comm_size().
"""
function mpi_comm_size(comm::Any)::Int
    if MPI_AVAILABLE
        return MPI.Comm_size(comm)
    else
        return 1
    end
end

"""
    mpi_comm_dup!(state::MVMCMPIState, comm::Any)

Duplicate MPI communicator.
Matches C function MPI_Comm_dup().
"""
function mpi_comm_dup!(state::MVMCMPIState, comm::Any)
    if MPI_AVAILABLE
        state.comm0 = MPI.Comm_dup(comm)
        state.rank0 = MPI.Comm_rank(state.comm0)
        state.size0 = MPI.Comm_size(state.comm0)
    else
        state.comm0 = comm
        state.rank0 = 0
        state.size0 = 1
    end
end

"""
    mpi_comm_split!(state::MVMCMPIState, comm::Any, color::Int, key::Int)

Split MPI communicator.
Matches C function MPI_Comm_split().
"""
function mpi_comm_split!(state::MVMCMPIState, comm::Any, color::Int, key::Int)
    if MPI_AVAILABLE
        new_comm = MPI.Comm_split(comm, color, key)
        return new_comm
    else
        return comm
    end
end

"""
    mpi_allreduce!(state::MVMCMPIState, sendbuf::Vector{T}, recvbuf::Vector{T}, op::Int, comm::Any = nothing) where T

MPI allreduce operation.
Matches C function MPI_Allreduce().
"""
function mpi_allreduce!(state::MVMCMPIState, sendbuf::Vector{T}, recvbuf::Vector{T}, op::Int, comm::Any = nothing) where T
    if MPI_AVAILABLE
        if comm === nothing
            comm = state.comm0
        end

        # Map operation codes
        mpi_op = if op == 0  # MPI_SUM
            MPI.SUM
        elseif op == 1  # MPI_MAX
            MPI.MAX
        elseif op == 2  # MPI_MIN
            MPI.MIN
        else
            MPI.SUM
        end

        MPI.Allreduce!(sendbuf, recvbuf, mpi_op, comm)
    else
        # Fallback: just copy
        copy!(recvbuf, sendbuf)
    end
end

"""
    mpi_bcast!(state::MVMCMPIState, buffer::Vector{T}, root::Int, comm::Any = nothing) where T

MPI broadcast operation.
Matches C function MPI_Bcast().
"""
function mpi_bcast!(state::MVMCMPIState, buffer::Vector{T}, root::Int, comm::Any = nothing) where T
    if MPI_AVAILABLE
        if comm === nothing
            comm = state.comm0
        end
        MPI.Bcast!(buffer, root, comm)
    end
end

"""
    mpi_gather!(state::MVMCMPIState, sendbuf::Vector{T}, recvbuf::Vector{T}, root::Int, comm::Any = nothing) where T

MPI gather operation.
Matches C function MPI_Gather().
"""
function mpi_gather!(state::MVMCMPIState, sendbuf::Vector{T}, recvbuf::Vector{T}, root::Int, comm::Any = nothing) where T
    if MPI_AVAILABLE
        if comm === nothing
            comm = state.comm0
        end
        MPI.Gather!(sendbuf, recvbuf, root, comm)
    else
        # Fallback: just copy
        copy!(recvbuf, sendbuf)
    end
end

"""
    mpi_scatter!(state::MVMCMPIState, sendbuf::Vector{T}, recvbuf::Vector{T}, root::Int, comm::Any = nothing) where T

MPI scatter operation.
Matches C function MPI_Scatter().
"""
function mpi_scatter!(state::MVMCMPIState, sendbuf::Vector{T}, recvbuf::Vector{T}, root::Int, comm::Any = nothing) where T
    if MPI_AVAILABLE
        if comm === nothing
            comm = state.comm0
        end
        MPI.Scatter!(sendbuf, recvbuf, root, comm)
    else
        # Fallback: just copy
        copy!(recvbuf, sendbuf)
    end
end

"""
    init_multi_def_mode!(state::MVMCMPIState, n_multi_def::Int, file_dir_list::String)

Initialize MultiDef mode.
Matches C function initMultiDefMode().
"""
function init_multi_def_mode!(state::MVMCMPIState, n_multi_def::Int, file_dir_list::String)
    state.flag_multi_def = true
    state.n_multi_def = n_multi_def
    state.file_dir_list = file_dir_list

    # In MultiDef mode, we need to:
    # 1. Create subdirectories for each definition
    # 2. Split communicators appropriately
    # 3. Change working directory

    if state.rank_world == 0
        # Create directories for each definition
        for i in 1:n_multi_def
            dir_name = @sprintf("def_%03d", i)
            if !isdir(dir_name)
                mkdir(dir_name)
            end
        end
    end

    # Split communicator based on definition index
    def_index = (state.rank_world % n_multi_def) + 1
    state.comm0 = mpi_comm_split!(state, state.comm_world, def_index - 1, state.rank_world)
    state.rank0 = mpi_comm_rank(state.comm0)
    state.size0 = mpi_comm_size(state.comm0)

    # Change to appropriate directory
    if state.rank_world < n_multi_def
        def_dir = @sprintf("def_%03d", state.rank_world + 1)
        if isdir(def_dir)
            cd(def_dir)
        end
    end
end

"""
    setup_communicator_splitting!(state::MVMCMPIState)

Set up communicator splitting for VMC calculations.
Matches C pattern from vmcmain.c.
"""
function setup_communicator_splitting!(state::MVMCMPIState)
    # Split communicator for VMC calculations
    # This matches the C pattern where comm1 and comm2 are created
    # for different parts of the calculation

    if state.size0 > 1
        # Split into two groups
        color = state.rank0 < div(state.size0, 2) ? 0 : 1
        key = state.rank0

        state.comm1 = mpi_comm_split!(state, state.comm0, color, key)
        state.rank1 = mpi_comm_rank(state.comm1)
        state.size1 = mpi_comm_size(state.comm1)

        # Create comm2 for the second group
        if color == 0
            state.comm2 = mpi_comm_split!(state, state.comm0, 1, key)
        else
            state.comm2 = mpi_comm_split!(state, state.comm0, 0, key)
        end
        state.rank2 = mpi_comm_rank(state.comm2)
        state.size2 = mpi_comm_size(state.comm2)

        # Set group information
        state.group1 = color
        state.group2 = 1 - color
    else
        # Single process case
        state.comm1 = state.comm0
        state.comm2 = state.comm0
        state.rank1 = 0
        state.size1 = 1
        state.rank2 = 0
        state.size2 = 1
        state.group1 = 0
        state.group2 = 0
    end
end

"""
    print_mpi_info(state::MVMCMPIState)

Print MPI information for debugging.
"""
function print_mpi_info(state::MVMCMPIState)
    if state.rank_world == 0
        println("=== MPI Information ===")
        println("World: rank=$(state.rank_world)/$(state.size_world)")
        println("Comm0: rank=$(state.rank0)/$(state.size0)")
        if state.comm1 !== nothing
            println("Comm1: rank=$(state.rank1)/$(state.size1)")
        end
        if state.comm2 !== nothing
            println("Comm2: rank=$(state.rank2)/$(state.size2)")
        end
        println("MultiDef: $(state.flag_multi_def), N=$(state.n_multi_def)")
        println("Standard: $(state.flag_standard)")
        println("======================")
    end
end

"""
    mpi_abort!(state::MVMCMPIState, errorcode::Int)

MPI abort operation.
Matches C function MPI_Abort().
"""
function mpi_abort!(state::MVMCMPIState, errorcode::Int)
    if MPI_AVAILABLE
        MPI.Abort(state.comm_world, errorcode)
    else
        exit(errorcode)
    end
end

"""
    is_root_process(state::MVMCMPIState) -> Bool

Check if current process is root.
"""
function is_root_process(state::MVMCMPIState)::Bool
    return state.rank_world == 0
end

"""
    is_root_process_comm0(state::MVMCMPIState) -> Bool

Check if current process is root in comm0.
"""
function is_root_process_comm0(state::MVMCMPIState)::Bool
    return state.rank0 == 0
end

# Export functions and types
export MVMCMPIState, mpi_init!, mpi_finalize!, mpi_barrier!, mpi_comm_rank, mpi_comm_size,
       mpi_comm_dup!, mpi_comm_split!, mpi_allreduce!, mpi_bcast!, mpi_gather!, mpi_scatter!,
       init_multi_def_mode!, setup_communicator_splitting!, print_mpi_info, mpi_abort!,
       is_root_process, is_root_process_comm0
