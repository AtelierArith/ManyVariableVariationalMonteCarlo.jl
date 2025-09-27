@testset "Workspace Management" begin
    using Test
    using ManyVariableVariationalMonteCarlo: Workspace, allocate_workspace!, reset_workspace!
    using StableRNGs
    # Test basic workspace allocation
    ws = Workspace{Int}(10)
    @test ws.size == 10
    @test ws.position == 0
    # Test allocation within initial size
    view1 = allocate_workspace!(ws, 5)
    @test length(view1) == 5
    @test ws.position == 5
    # Test allocation requiring expansion
    view2 = allocate_workspace!(ws, 10)
    @test length(view2) == 10
    @test ws.size >= 15  # Should have expanded
    # Test workspace reset
    reset_workspace!(ws)
    @test ws.position == 0
end
@testset "WorkspaceManager" begin
    using Test
    using ManyVariableVariationalMonteCarlo: WorkspaceManager
    # Test workspace manager creation
    manager = WorkspaceManager(2)
    @test length(manager.int_workspaces) == 2
    @test length(manager.double_workspaces) == 2
    @test length(manager.complex_workspaces) == 2
end
@testset "Thread-safe workspace access" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Statistics
    # Test thread-safe workspace allocation
    int_ws = get_workspace(Int, 100)
    @test length(int_ws) == 100
    double_ws = get_workspace(Float64, 50)
    @test length(double_ws) == 50
    complex_ws = get_workspace(ComplexF64, 25)
    @test length(complex_ws) == 25
    # Test workspace reset
    reset_all_workspaces!()
    # No easy way to test reset without accessing internals
end
@testset "MemoryLayout" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Statistics
    # Test basic memory layout
    layout = MemoryLayout(nsite=10, ne=5)
    @test layout.nsite == 10
    @test layout.ne == 5
    @test layout.nsize == 10
    # Test with Hamiltonian terms
    layout = MemoryLayout(
        nsite=10, ne=5,
        ntransfer=20, ncoulomb_intra=10,
        ncoulomb_inter=15, nhund_coupling=8
    )
    @test layout.ntransfer == 20
    @test layout.ncoulomb_intra == 10
    @test layout.ncoulomb_inter == 15
    @test layout.nhund_coupling == 8
end
@testset "Global array allocation" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Statistics
    layout = MemoryLayout(
        nsite=4, ne=2,
        ntransfer=8, ncoulomb_intra=4,
        nrbm_hidden=10, nrbm_visible=8
    )
    arrays = allocate_global_arrays(layout)
    # Test array sizes
    @test length(arrays.loc_spn) == 4
    @test size(arrays.transfer_indices) == (8, 4)
    @test length(arrays.transfer_params) == 8
    @test length(arrays.coulomb_intra_indices) == 4
    @test length(arrays.coulomb_intra_params) == 4
    @test size(arrays.rbm_hidden_weights) == (10, 8)
    @test length(arrays.rbm_visible_bias) == 8
    @test length(arrays.rbm_hidden_bias) == 10
    # Test array types
    @test eltype(arrays.loc_spn) == Int
    @test eltype(arrays.transfer_params) == ComplexF64
    @test eltype(arrays.coulomb_intra_params) == Float64
    @test eltype(arrays.rbm_hidden_weights) == ComplexF64
end
@testset "Memory summary calculation" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Statistics
    layout = MemoryLayout(nsite=100, ne=50, ntransfer=200)
    memory_mb = memory_summary(layout)
    @test memory_mb > 0.0
    @test memory_mb < 1000.0  # Should be reasonable for this size
    # Test larger system
    large_layout = MemoryLayout(nsite=1000, ne=500, ntransfer=2000)
    large_memory_mb = memory_summary(large_layout)
    @test large_memory_mb > memory_mb
end
@testset "Memory layout edge cases" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Statistics
    # Test minimum system
    layout = MemoryLayout(nsite=2, ne=1)
    arrays = allocate_global_arrays(layout)
    @test length(arrays.loc_spn) == 2
    # Test with zero Hamiltonian terms
    layout = MemoryLayout(
        nsite=4, ne=2,
        ntransfer=0, ncoulomb_intra=0
    )
    arrays = allocate_global_arrays(layout)
    @test size(arrays.transfer_indices) == (0, 4)
    @test length(arrays.transfer_params) == 0
end
@testset "Performance characteristics" begin
    using Test
    using ManyVariableVariationalMonteCarlo
    using Statistics
    using ManyVariableVariationalMonteCarlo: Workspace, allocate_workspace!
    # Test that memory allocation is efficient
    layout = MemoryLayout(nsite=100, ne=50, ntransfer=200)
    # Should allocate quickly
    @test (@elapsed allocate_global_arrays(layout)) < 0.1
    # Should allocate minimal workspace overhead
    ws = Workspace{Float64}(1000)
    @test (@elapsed allocate_workspace!(ws, 500)) < 0.01
end