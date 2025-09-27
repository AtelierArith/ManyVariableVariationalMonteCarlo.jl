module ManyVariableVariationalMonteCarlo

# Core modules
include("types.jl")
include("config.jl")
include("parameters.jl")
include("io.jl")
include("defs.jl")

# New infrastructure modules
include("memory.jl")
include("rng.jl")
include("linalg_simple.jl")

# Wavefunction components
include("slater.jl")
include("rbm.jl")

# Monte Carlo sampling
include("sampler.jl")

# Import functions from included modules
import .get_workspace, .reset_all_workspaces!

export SimulationConfig, FaceDefinition, facevalue, load_face_definition,
       ParameterLayout, ParameterFlags, ParameterMask, ParameterSet,
       initialize_parameters!, apply_opttrans_basis!,
       GreenFunctionEntry, GreenFunctionTable,
       read_initial_green, AMP_MAX,
       Namelist, load_namelist,
       TransferEntry, TransferTable, read_transfer_table,
       CoulombIntraEntry, CoulombIntraTable, read_coulomb_intra,
       InterAllEntry, InterAllTable, read_interall_table,
       GreenOneEntry, GreenOneTable, read_greenone_indices,
       # Memory management exports
       MemoryLayout, allocate_global_arrays, memory_summary,
       Workspace, WorkspaceManager, get_workspace, reset_all_workspaces!,
       # RNG exports
       VMCRng, ParallelRngManager, RngState, initialize_rng!, get_thread_rng,
       vmcrand, vmcrandn, vmcrand_int, vmcrand_bool,
       save_rng_state, restore_rng_state!, rng_info,
       benchmark_rng, test_rng_quality,
       # Linear algebra exports
       pfaffian, pfaffian_and_inverse, pfaffian_det_relation, pfaffian_skew_symmetric,
       is_antisymmetric, MatrixCalculation,
       sherman_morrison_update!, woodbury_update!, matrix_ratio,
       get_matrix_calculation, clear_matrix_calculations!, benchmark_linalg,
       PfaffianLimitError,
       # Slater determinant exports
       SlaterMatrix, SlaterDeterminant, initialize_slater!,
       compute_determinant!, compute_inverse!, update_slater!,
       two_electron_update!, get_determinant_value, get_log_determinant_value,
       is_valid, reset_slater!,
       # RBM exports
       RBMNetwork, initialize_rbm!, rbm_weight, log_rbm_weight,
       rbm_weight_phys, log_rbm_weight_phys, rbm_gradient, rbm_gradient_phys,
       update_rbm_weights!, update_rbm_weights_phys!, get_rbm_parameters,
       set_rbm_parameters!, rbm_parameter_count, reset_rbm!,
       # VMC sampling exports
       VMCConfig, VMCState, VMCResults, initialize_vmc_state!,
       propose_single_electron_move, propose_two_electron_move,
       accept_move!, reject_move!, metropolis_step!, run_vmc_sampling!,
       measure_energy, measure_observables, get_acceptance_rate, reset_vmc_state!

end # module ManyVariableVariationalMonteCarlo
