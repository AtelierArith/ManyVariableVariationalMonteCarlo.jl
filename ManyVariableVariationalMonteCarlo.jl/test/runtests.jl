using Test
using ManyVariableVariationalMonteCarlo

@testset "ManyVariableVariationalMonteCarlo.jl tests" begin
    include("simple_test.jl")
    include("config_tests.jl")
    include("linalg_tests.jl")
    include("linalg_enhanced_tests.jl")
    include("greens_tests.jl")
    include("projections_tests.jl")
    include("memory_tests.jl")
    include("rbm_tests.jl")
end

