"""
Memory Management for mVMC C Compatibility

Provides memory allocation and management matching the C implementation's
memory layout and allocation patterns.

Ported from setmemory.c and workspace.c.
"""

using LinearAlgebra

"""
    MVMCMemoryManager

Memory manager matching the C implementation's memory allocation patterns.
"""
mutable struct MVMCMemoryManager
    # Memory allocation tracking
    allocated_arrays::Dict{String, Any}
    memory_usage::Dict{String, Int}
    total_memory_usage::Int

    # Workspace management
    workspace_arrays::Dict{String, Any}
    workspace_size::Int

    # Memory layout flags
    use_aligned_memory::Bool
    memory_alignment::Int

    function MVMCMemoryManager()
        new(
            Dict{String, Any}(),
            Dict{String, Int}(),
            0,
            Dict{String, Any}(),
            0,
            true,
            64  # 64-byte alignment for SIMD
        )
    end
end

"""
    set_memory_def!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate memory based on definition file counts.
Matches C function SetMemoryDef().

C実装参考: setmemory.c 1行目から503行目まで
"""
function set_memory_def!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    # Clear existing allocations
    clear_all_allocations!(manager)

    # Allocate arrays based on system parameters
    allocate_system_arrays!(manager, state)
    allocate_hamiltonian_arrays!(manager, state)
    allocate_variational_arrays!(manager, state)
    allocate_electron_arrays!(manager, state)
    allocate_slater_arrays!(manager, state)
    allocate_projection_arrays!(manager, state)
    allocate_sr_arrays!(manager, state)
    allocate_physical_arrays!(manager, state)
    allocate_workspace_arrays!(manager, state)

    # Update total memory usage
    update_memory_usage!(manager)
end

"""
    allocate_system_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for system parameters.
"""
function allocate_system_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    nsite = state.nsite
    ne = state.ne
    nsize = state.nsize

    # Local spin arrays
    if state.n_loc_spn > 0
        state.loc_spn = Vector{Int}(undef, nsite)
        manager.allocated_arrays["loc_spn"] = state.loc_spn
        manager.memory_usage["loc_spn"] = sizeof(state.loc_spn)
    end

    # Electron configuration arrays
    state.ele_idx = Vector{Int}(undef, nsize)
    state.ele_cfg = Vector{Int}(undef, nsite * 2)
    state.ele_num = Vector{Int}(undef, nsite * 2)
    state.ele_proj_cnt = Vector{Int}(undef, state.n_proj)
    state.ele_spn = Vector{Int}(undef, nsize)

    manager.allocated_arrays["ele_idx"] = state.ele_idx
    manager.allocated_arrays["ele_cfg"] = state.ele_cfg
    manager.allocated_arrays["ele_num"] = state.ele_num
    manager.allocated_arrays["ele_proj_cnt"] = state.ele_proj_cnt
    manager.allocated_arrays["ele_spn"] = state.ele_spn

    # Temporary arrays
    state.tmp_ele_idx = Vector{Int}(undef, nsize)
    state.tmp_ele_cfg = Vector{Int}(undef, nsite * 2)
    state.tmp_ele_num = Vector{Int}(undef, nsite * 2)
    state.tmp_ele_proj_cnt = Vector{Int}(undef, state.n_proj)
    state.tmp_ele_spn = Vector{Int}(undef, nsize)

    manager.allocated_arrays["tmp_ele_idx"] = state.tmp_ele_idx
    manager.allocated_arrays["tmp_ele_cfg"] = state.tmp_ele_cfg
    manager.allocated_arrays["tmp_ele_num"] = state.tmp_ele_num
    manager.allocated_arrays["tmp_ele_proj_cnt"] = state.tmp_ele_proj_cnt
    manager.allocated_arrays["tmp_ele_spn"] = state.tmp_ele_spn

    # Burn arrays
    state.burn_ele_idx = Vector{Int}(undef, nsize)
    state.burn_ele_cfg = Vector{Int}(undef, nsite * 2)
    state.burn_ele_num = Vector{Int}(undef, nsite * 2)
    state.burn_ele_proj_cnt = Vector{Int}(undef, state.n_proj)
    state.burn_ele_spn = Vector{Int}(undef, nsize)

    manager.allocated_arrays["burn_ele_idx"] = state.burn_ele_idx
    manager.allocated_arrays["burn_ele_cfg"] = state.burn_ele_cfg
    manager.allocated_arrays["burn_ele_num"] = state.burn_ele_num
    manager.allocated_arrays["burn_ele_proj_cnt"] = state.burn_ele_proj_cnt
    manager.allocated_arrays["burn_ele_spn"] = state.burn_ele_spn
end

"""
    allocate_hamiltonian_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for Hamiltonian terms.
"""
function allocate_hamiltonian_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    # Transfer terms
    if state.n_transfer > 0
        state.transfer = zeros(Int, state.n_transfer, 4)
        state.para_transfer = Vector{ComplexF64}(undef, state.n_transfer)
        manager.allocated_arrays["transfer"] = state.transfer
        manager.allocated_arrays["para_transfer"] = state.para_transfer
    end

    # Coulomb intra terms
    if state.n_coulomb_intra > 0
        state.coulomb_intra = Vector{Int}(undef, state.n_coulomb_intra)
        state.para_coulomb_intra = Vector{Float64}(undef, state.n_coulomb_intra)
        manager.allocated_arrays["coulomb_intra"] = state.coulomb_intra
        manager.allocated_arrays["para_coulomb_intra"] = state.para_coulomb_intra
    end

    # Coulomb inter terms
    if state.n_coulomb_inter > 0
        state.coulomb_inter = zeros(Int, state.n_coulomb_inter, 2)
        state.para_coulomb_inter = Vector{Float64}(undef, state.n_coulomb_inter)
        manager.allocated_arrays["coulomb_inter"] = state.coulomb_inter
        manager.allocated_arrays["para_coulomb_inter"] = state.para_coulomb_inter
    end

    # Hund coupling terms
    if state.n_hund_coupling > 0
        state.hund_coupling = zeros(Int, state.n_hund_coupling, 2)
        state.para_hund_coupling = Vector{Float64}(undef, state.n_hund_coupling)
        manager.allocated_arrays["hund_coupling"] = state.hund_coupling
        manager.allocated_arrays["para_hund_coupling"] = state.para_hund_coupling
    end

    # Pair hopping terms
    if state.n_pair_hopping > 0
        state.pair_hopping = zeros(Int, state.n_pair_hopping, 2)
        state.para_pair_hopping = Vector{Float64}(undef, state.n_pair_hopping)
        manager.allocated_arrays["pair_hopping"] = state.pair_hopping
        manager.allocated_arrays["para_pair_hopping"] = state.para_pair_hopping
    end

    # Exchange coupling terms
    if state.n_exchange_coupling > 0
        state.exchange_coupling = zeros(Int, state.n_exchange_coupling, 2)
        state.para_exchange_coupling = Vector{Float64}(undef, state.n_exchange_coupling)
        manager.allocated_arrays["exchange_coupling"] = state.exchange_coupling
        manager.allocated_arrays["para_exchange_coupling"] = state.para_exchange_coupling
    end

    # InterAll terms
    if state.n_inter_all > 0
        state.inter_all = zeros(Int, state.n_inter_all, 8)
        state.para_inter_all = Vector{ComplexF64}(undef, state.n_inter_all)
        manager.allocated_arrays["inter_all"] = state.inter_all
        manager.allocated_arrays["para_inter_all"] = state.para_inter_all
    end
end

"""
    allocate_variational_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for variational parameters.
"""
function allocate_variational_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    # Gutzwiller parameters
    if state.n_gutzwiller_idx > 0
        state.gutzwiller_idx = Vector{Int}(undef, state.n_gutzwiller_idx)
        manager.allocated_arrays["gutzwiller_idx"] = state.gutzwiller_idx
    end

    # Jastrow parameters
    if state.n_jastrow_idx > 0
        state.jastrow_idx = zeros(Int, state.n_jastrow_idx, 2)
        manager.allocated_arrays["jastrow_idx"] = state.jastrow_idx
    end

    # Doublon-Holon parameters
    if state.n_doublon_holon_2site_idx > 0
        state.doublon_holon_2site_idx = zeros(Int, state.n_doublon_holon_2site_idx, 2 * state.nsite)
        manager.allocated_arrays["doublon_holon_2site_idx"] = state.doublon_holon_2site_idx
    end

    if state.n_doublon_holon_4site_idx > 0
        state.doublon_holon_4site_idx = zeros(Int, state.n_doublon_holon_4site_idx, 4 * state.nsite)
        manager.allocated_arrays["doublon_holon_4site_idx"] = state.doublon_holon_4site_idx
    end

    # Orbital parameters
    if state.n_orbital_idx > 0
        state.orbital_idx = zeros(Int, state.n_orbital_idx, 2)
        state.orbital_sgn = zeros(Int, state.n_orbital_idx, 2)
        manager.allocated_arrays["orbital_idx"] = state.orbital_idx
        manager.allocated_arrays["orbital_sgn"] = state.orbital_sgn
    end

    # RBM parameters
    if state.n_neuron > 0
        allocate_rbm_arrays!(manager, state)
    end

    # Quantum projection transformation
    if state.n_qp_trans > 0
        state.qp_trans = zeros(Int, state.n_qp_trans, state.nsite)
        state.qp_trans_inv = zeros(Int, state.n_qp_trans, state.nsite)
        state.qp_trans_sgn = zeros(Int, state.n_qp_trans, state.nsite)
        state.para_qp_trans = Vector{ComplexF64}(undef, state.n_qp_trans)
        manager.allocated_arrays["qp_trans"] = state.qp_trans
        manager.allocated_arrays["qp_trans_inv"] = state.qp_trans_inv
        manager.allocated_arrays["qp_trans_sgn"] = state.qp_trans_sgn
        manager.allocated_arrays["para_qp_trans"] = state.para_qp_trans
    end

    # Optimal transformation
    if state.n_qp_opt_trans > 0
        state.qp_opt_trans = zeros(Int, state.n_qp_opt_trans, state.nsite)
        state.qp_opt_trans_sgn = zeros(Int, state.n_qp_opt_trans, state.nsite)
        state.para_qp_opt_trans = Vector{Float64}(undef, state.n_qp_opt_trans)
        manager.allocated_arrays["qp_opt_trans"] = state.qp_opt_trans
        manager.allocated_arrays["qp_opt_trans_sgn"] = state.qp_opt_trans_sgn
        manager.allocated_arrays["para_qp_opt_trans"] = state.para_qp_opt_trans
    end

    # Main parameter arrays
    if state.n_para > 0
        state.para = Vector{ComplexF64}(undef, state.n_para)
        state.proj = Vector{ComplexF64}(undef, state.n_proj)
        state.slater = Vector{ComplexF64}(undef, state.n_slater)
        state.opt_trans = Vector{ComplexF64}(undef, state.n_opt_trans)
        manager.allocated_arrays["para"] = state.para
        manager.allocated_arrays["proj"] = state.proj
        manager.allocated_arrays["slater"] = state.slater
        manager.allocated_arrays["opt_trans"] = state.opt_trans
    end
end

"""
    allocate_rbm_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate RBM-specific arrays.
"""
function allocate_rbm_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    # General RBM
    if state.n_general_rbm_hidden_layer_idx > 0
        state.general_rbm_hidden_layer_idx = Vector{Int}(undef, state.n_general_rbm_hidden_layer_idx)
        manager.allocated_arrays["general_rbm_hidden_layer_idx"] = state.general_rbm_hidden_layer_idx
    end

    if state.n_general_rbm_phys_layer_idx > 0
        state.general_rbm_phys_layer_idx = Vector{Int}(undef, state.n_general_rbm_phys_layer_idx)
        manager.allocated_arrays["general_rbm_phys_layer_idx"] = state.general_rbm_phys_layer_idx
    end

    if state.n_general_rbm_phys_hidden_idx > 0
        state.general_rbm_phys_hidden_idx = zeros(Int, state.n_general_rbm_phys_hidden_idx, state.nsite * 2)
        manager.allocated_arrays["general_rbm_phys_hidden_idx"] = state.general_rbm_phys_hidden_idx
    end

    # Charge RBM
    if state.n_charge_rbm_hidden_layer_idx > 0
        state.charge_rbm_hidden_layer_idx = Vector{Int}(undef, state.n_charge_rbm_hidden_layer_idx)
        manager.allocated_arrays["charge_rbm_hidden_layer_idx"] = state.charge_rbm_hidden_layer_idx
    end

    if state.n_charge_rbm_phys_layer_idx > 0
        state.charge_rbm_phys_layer_idx = Vector{Int}(undef, state.n_charge_rbm_phys_layer_idx)
        manager.allocated_arrays["charge_rbm_phys_layer_idx"] = state.charge_rbm_phys_layer_idx
    end

    if state.n_charge_rbm_phys_hidden_idx > 0
        state.charge_rbm_phys_hidden_idx = zeros(Int, state.n_charge_rbm_phys_hidden_idx, state.nsite)
        manager.allocated_arrays["charge_rbm_phys_hidden_idx"] = state.charge_rbm_phys_hidden_idx
    end

    # Spin RBM
    if state.n_spin_rbm_hidden_layer_idx > 0
        state.spin_rbm_hidden_layer_idx = Vector{Int}(undef, state.n_spin_rbm_hidden_layer_idx)
        manager.allocated_arrays["spin_rbm_hidden_layer_idx"] = state.spin_rbm_hidden_layer_idx
    end

    if state.n_spin_rbm_phys_layer_idx > 0
        state.spin_rbm_phys_layer_idx = Vector{Int}(undef, state.n_spin_rbm_phys_layer_idx)
        manager.allocated_arrays["spin_rbm_phys_layer_idx"] = state.spin_rbm_phys_layer_idx
    end

    if state.n_spin_rbm_phys_hidden_idx > 0
        state.spin_rbm_phys_hidden_idx = zeros(Int, state.n_spin_rbm_phys_hidden_idx, state.nsite)
        manager.allocated_arrays["spin_rbm_phys_hidden_idx"] = state.spin_rbm_phys_hidden_idx
    end

    # RBM parameters
    if state.n_rbm > 0
        state.rbm = Vector{ComplexF64}(undef, state.n_rbm)
        state.rbm_cnt = Vector{ComplexF64}(undef, state.n_neuron)
        manager.allocated_arrays["rbm"] = state.rbm
        manager.allocated_arrays["rbm_cnt"] = state.rbm_cnt
    end
end

"""
    allocate_electron_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for electron configuration.
"""
function allocate_electron_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    nsite = state.nsite
    ne = state.ne
    nsize = state.nsize

    # Electron configuration arrays
    state.ele_idx = Vector{Int}(undef, nsize)
    state.ele_cfg = Vector{Int}(undef, nsite * 2)
    state.ele_num = Vector{Int}(undef, nsite * 2)
    state.ele_proj_cnt = Vector{Int}(undef, state.n_proj)
    state.ele_spn = Vector{Int}(undef, nsize)

    # RBM counting arrays
    if state.n_neuron > 0
        state.rbm_cnt = Vector{ComplexF64}(undef, state.n_neuron)
        state.ele_proj_bf_cnt = Vector{Int}(undef, state.n_proj_bf)
        manager.allocated_arrays["rbm_cnt"] = state.rbm_cnt
        manager.allocated_arrays["ele_proj_bf_cnt"] = state.ele_proj_bf_cnt
    end

    # Backflow arrays
    if state.n_back_flow_idx > 0
        state.back_flow_idx = zeros(Int, state.n_back_flow_idx, state.nsite)
        state.pos_bf = zeros(Int, state.n_range, state.nsite)
        state.range_idx = zeros(Int, state.n_range, state.nsite)
        state.bf_sub_idx = zeros(Int, state.nsite, state.nsite)
        manager.allocated_arrays["back_flow_idx"] = state.back_flow_idx
        manager.allocated_arrays["pos_bf"] = state.pos_bf
        manager.allocated_arrays["range_idx"] = state.range_idx
        manager.allocated_arrays["bf_sub_idx"] = state.bf_sub_idx
    end

    # Wavefunction arrays
    state.log_sq_pf_full_slater = Vector{Float64}(undef, state.nvmc_sample)
    state.smp_slt_elm_bf_real = Vector{Float64}(undef, state.nvmc_sample)
    state.smp_eta_flag = Vector{Int}(undef, state.nvmc_sample)
    state.smp_eta = Vector{Float64}(undef, state.nvmc_sample)

    manager.allocated_arrays["log_sq_pf_full_slater"] = state.log_sq_pf_full_slater
    manager.allocated_arrays["smp_slt_elm_bf_real"] = state.smp_slt_elm_bf_real
    manager.allocated_arrays["smp_eta_flag"] = state.smp_eta_flag
    manager.allocated_arrays["smp_eta"] = state.smp_eta
end

"""
    allocate_slater_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for Slater determinant calculations.
"""
function allocate_slater_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    nsite = state.nsite
    ne = state.ne
    nqp_full = state.nqp_full

    # Slater determinant arrays
    if nqp_full > 0
        state.slater_elm = Array{ComplexF64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
        state.inv_m = Array{ComplexF64, 3}(undef, nqp_full, ne, ne)
        state.pf_m = Vector{ComplexF64}(undef, nqp_full)

        # Real versions
        state.slater_elm_real = Array{Float64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
        state.inv_m_real = Array{Float64, 3}(undef, nqp_full, ne, ne)
        state.pf_m_real = Vector{Float64}(undef, nqp_full)

        # Backflow versions
        state.slater_elm_bf = Array{ComplexF64, 3}(undef, nqp_full, nsite * 2, nsite * 2)
        state.slater_elm_bf_real = Array{Float64, 3}(undef, nqp_full, nsite * 2, nsite * 2)

        manager.allocated_arrays["slater_elm"] = state.slater_elm
        manager.allocated_arrays["inv_m"] = state.inv_m
        manager.allocated_arrays["pf_m"] = state.pf_m
        manager.allocated_arrays["slater_elm_real"] = state.slater_elm_real
        manager.allocated_arrays["inv_m_real"] = state.inv_m_real
        manager.allocated_arrays["pf_m_real"] = state.pf_m_real
        manager.allocated_arrays["slater_elm_bf"] = state.slater_elm_bf
        manager.allocated_arrays["slater_elm_bf_real"] = state.slater_elm_bf_real
    end
end

"""
    allocate_projection_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for quantum projection.
"""
function allocate_projection_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    nqp_full = state.nqp_full
    nqp_fix = state.nqp_fix
    nsp_gauss_leg = state.nsp_gauss_leg

    # Quantum projection weights
    if nqp_full > 0
        state.qp_full_weight = Vector{ComplexF64}(undef, nqp_full)
        manager.allocated_arrays["qp_full_weight"] = state.qp_full_weight
    end

    if nqp_fix > 0
        state.qp_fix_weight = Vector{ComplexF64}(undef, nqp_fix)
        manager.allocated_arrays["qp_fix_weight"] = state.qp_fix_weight
    end

    # Gauss-Legendre quadrature
    if nsp_gauss_leg > 0
        state.spgl_cos = Vector{ComplexF64}(undef, nsp_gauss_leg)
        state.spgl_sin = Vector{ComplexF64}(undef, nsp_gauss_leg)
        state.spgl_cos_sin = Vector{ComplexF64}(undef, nsp_gauss_leg)
        state.spgl_cos_cos = Vector{ComplexF64}(undef, nsp_gauss_leg)
        state.spgl_sin_sin = Vector{ComplexF64}(undef, nsp_gauss_leg)

        manager.allocated_arrays["spgl_cos"] = state.spgl_cos
        manager.allocated_arrays["spgl_sin"] = state.spgl_sin
        manager.allocated_arrays["spgl_cos_sin"] = state.spgl_cos_sin
        manager.allocated_arrays["spgl_cos_cos"] = state.spgl_cos_cos
        manager.allocated_arrays["spgl_sin_sin"] = state.spgl_sin_sin
    end
end

"""
    allocate_sr_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for stochastic reconfiguration.
"""
function allocate_sr_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    n_para = state.n_para
    nvmc_sample = state.nvmc_sample

    # SR matrices
    if n_para > 0
        state.sr_opt_oo = Matrix{ComplexF64}(undef, n_para + 1, n_para + 1)
        state.sr_opt_ho = Vector{ComplexF64}(undef, n_para + 1)
        state.sr_opt_o = Vector{ComplexF64}(undef, n_para + 1)
        state.sr_opt_o_store = Matrix{ComplexF64}(undef, n_para + 1, nvmc_sample)

        # Real versions
        state.sr_opt_oo_real = Matrix{Float64}(undef, n_para + 1, n_para + 1)
        state.sr_opt_ho_real = Vector{Float64}(undef, n_para + 1)
        state.sr_opt_o_real = Vector{Float64}(undef, n_para + 1)
        state.sr_opt_o_store_real = Matrix{Float64}(undef, n_para + 1, nvmc_sample)

        # SR data
        state.sr_opt_data = Vector{ComplexF64}(undef, 2 + n_para)

        manager.allocated_arrays["sr_opt_oo"] = state.sr_opt_oo
        manager.allocated_arrays["sr_opt_ho"] = state.sr_opt_ho
        manager.allocated_arrays["sr_opt_o"] = state.sr_opt_o
        manager.allocated_arrays["sr_opt_o_store"] = state.sr_opt_o_store
        manager.allocated_arrays["sr_opt_oo_real"] = state.sr_opt_oo_real
        manager.allocated_arrays["sr_opt_ho_real"] = state.sr_opt_ho_real
        manager.allocated_arrays["sr_opt_o_real"] = state.sr_opt_o_real
        manager.allocated_arrays["sr_opt_o_store_real"] = state.sr_opt_o_store_real
        manager.allocated_arrays["sr_opt_data"] = state.sr_opt_data
    end
end

"""
    allocate_physical_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate arrays for physical quantities.
"""
function allocate_physical_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    # Green function arrays
    if state.n_cis_ajs > 0
        state.phys_cis_ajs = Vector{ComplexF64}(undef, state.n_cis_ajs)
        state.local_cis_ajs = Vector{ComplexF64}(undef, state.n_cis_ajs)
        manager.allocated_arrays["phys_cis_ajs"] = state.phys_cis_ajs
        manager.allocated_arrays["local_cis_ajs"] = state.local_cis_ajs
    end

    if state.n_cis_ajs_ckt_alt > 0
        state.phys_cis_ajs_ckt_alt = Vector{ComplexF64}(undef, state.n_cis_ajs_ckt_alt)
        manager.allocated_arrays["phys_cis_ajs_ckt_alt"] = state.phys_cis_ajs_ckt_alt
    end

    if state.n_cis_ajs_ckt_alt_dc > 0
        state.phys_cis_ajs_ckt_alt_dc = Vector{ComplexF64}(undef, state.n_cis_ajs_ckt_alt_dc)
        state.local_cis_ajs_ckt_alt_dc = Vector{ComplexF64}(undef, state.n_cis_ajs_ckt_alt_dc)
        manager.allocated_arrays["phys_cis_ajs_ckt_alt_dc"] = state.phys_cis_ajs_ckt_alt_dc
        manager.allocated_arrays["local_cis_ajs_ckt_alt_dc"] = state.local_cis_ajs_ckt_alt_dc
    end

    # Large scale calculation arrays
    if state.nlanczos_mode > 0
        n_ls_ham = 2
        state.qqqq = Array{ComplexF64, 4}(undef, n_ls_ham, n_ls_ham, n_ls_ham, n_ls_ham)
        state.lsl_q = Matrix{ComplexF64}(undef, n_ls_ham, n_ls_ham)
        state.qqqq_real = Array{Float64, 4}(undef, n_ls_ham, n_ls_ham, n_ls_ham, n_ls_ham)
        state.lsl_q_real = Matrix{Float64}(undef, n_ls_ham, n_ls_ham)

        manager.allocated_arrays["qqqq"] = state.qqqq
        manager.allocated_arrays["lsl_q"] = state.lsl_q
        manager.allocated_arrays["qqqq_real"] = state.qqqq_real
        manager.allocated_arrays["lsl_q_real"] = state.lsl_q_real
    end
end

"""
    allocate_workspace_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)

Allocate workspace arrays.
"""
function allocate_workspace_arrays!(manager::MVMCMemoryManager, state::MVMCGlobalState)
    # Workspace arrays for calculations
    # These would be allocated based on the specific calculation needs
    # For now, we'll set up basic workspace management

    manager.workspace_size = 0
    manager.workspace_arrays = Dict{String, Any}()
end

"""
    clear_all_allocations!(manager::MVMCMemoryManager)

Clear all allocated arrays.
"""
function clear_all_allocations!(manager::MVMCMemoryManager)
    # Clear allocated arrays
    for (name, array) in manager.allocated_arrays
        # Arrays will be garbage collected automatically
    end

    empty!(manager.allocated_arrays)
    empty!(manager.memory_usage)
    empty!(manager.workspace_arrays)

    manager.total_memory_usage = 0
    manager.workspace_size = 0
end

"""
    update_memory_usage!(manager::MVMCMemoryManager)

Update memory usage statistics.
"""
function update_memory_usage!(manager::MVMCMemoryManager)
    total = 0
    for (name, array) in manager.allocated_arrays
        size_bytes = sizeof(array)
        manager.memory_usage[name] = size_bytes
        total += size_bytes
    end

    manager.total_memory_usage = total
end

"""
    print_memory_summary(manager::MVMCMemoryManager)

Print memory usage summary.
"""
function print_memory_summary(manager::MVMCMemoryManager)
    println("=== Memory Usage Summary ===")
    println("Total memory: $(manager.total_memory_usage) bytes")
    println("Allocated arrays: $(length(manager.allocated_arrays))")

    for (name, size_bytes) in manager.memory_usage
        println("  $name: $size_bytes bytes")
    end

    println("===========================")
end

# Export functions and types
export MVMCMemoryManager, set_memory_def!, clear_all_allocations!,
       update_memory_usage!, print_memory_summary
