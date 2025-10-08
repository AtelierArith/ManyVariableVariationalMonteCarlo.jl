"""
Global State Management for mVMC C Compatibility

Provides Julia equivalents of global variables from global.h in the C implementation.
Manages configuration flags, system parameters, file handles, and timing infrastructure.

Ported from global.h and related C modules.
"""

using Printf
using LinearAlgebra

# Constants matching C implementation
const D_FileNameMax = 256
const NTimer = 1000

"""
    MVMCGlobalState

Global state container matching the C global.h variables.
This provides a Julia equivalent of the global variables used throughout mVMC.
"""
mutable struct MVMCGlobalState
    # File naming
    c_data_file_head::String
    c_para_file_head::String

    # Calculation modes
    nvmc_cal_mode::Int  # 0: optimization, 1: physics calculation
    nlanczos_mode::Int  # 0: none, 1: energy only, 2: Green functions

    # Storage and solver options
    nstore_o::Int  # 0: normal, other: store
    nsrcg::Int     # 0: (Sca)LAPACK, other: CG

    # Data output control
    n_data_idx_start::Int
    n_data_qty_smp::Int

    # System parameters
    nsite::Int      # Number of sites
    ne::Int         # Number of electrons with up spin
    nup::Int        # Number of electrons with up spin
    nsize::Int      # Total number of electrons = 2*Ne
    nsite2::Int     # 2*Nsite
    nz::Int         # Connectivity
    two_sz::Int     # 2*Sz

    # Quantum projection parameters
    nsp_gauss_leg::Int    # Number of Gauss-Legendre points
    nsp_stot::Int         # S of Spin projection
    nmp_trans::Int        # Number of quantum projection for translation and point group
    nqp_full::Int         # Total number of quantum projection
    nqp_fix::Int          # For QPFixWeight

    # Stochastic reconfiguration parameters
    nsr_opt_itr_step::Int    # Number of SR method steps
    nsr_opt_itr_smp::Int     # Number of SR method steps for average
    nsr_opt_fix_smp::Int     # Number of SR method steps with fixed samples
    dsr_opt_red_cut::Float64 # SR stabilizing factor for truncation
    dsr_opt_sta_del::Float64 # SR stabilizing factor for diagonal modification
    dsr_opt_step_dt::Float64 # Step width of SR method
    nsr_opt_cg_max_iter::Int # Maximum iterations in SR-CG method
    dsr_opt_cg_tol::Float64 # Tolerance for SR-CG method

    # Monte Carlo parameters
    nvmc_warm_up::Int      # MC steps for warming up
    nvmc_interval::Int     # Sampling interval [MCS]
    nvmc_sample::Int       # Number of samples
    n_ex_update_path::Int  # Update by exchange hopping
    n_block_update_size::Int # Size of block Pfaffian update

    # Random number generation
    rnd_seed::Int
    n_split_size::Int      # Number of inner MPI processes

    # Total length of definition arrays
    n_total_def_int::Int
    n_total_def_double::Int

    # Local spin configuration
    n_loc_spn::Int
    loc_spn::Union{Vector{Int}, Nothing}

    # Hamiltonian terms
    n_transfer::Int
    transfer::Union{Matrix{Int}, Nothing}
    para_transfer::Union{Vector{ComplexF64}, Nothing}

    n_coulomb_intra::Int
    coulomb_intra::Union{Vector{Int}, Nothing}
    para_coulomb_intra::Union{Vector{Float64}, Nothing}

    n_coulomb_inter::Int
    coulomb_inter::Union{Matrix{Int}, Nothing}
    para_coulomb_inter::Union{Vector{Float64}, Nothing}

    n_hund_coupling::Int
    hund_coupling::Union{Matrix{Int}, Nothing}
    para_hund_coupling::Union{Vector{Float64}, Nothing}

    n_pair_hopping::Int
    pair_hopping::Union{Matrix{Int}, Nothing}
    para_pair_hopping::Union{Vector{Float64}, Nothing}

    n_exchange_coupling::Int
    exchange_coupling::Union{Matrix{Int}, Nothing}
    para_exchange_coupling::Union{Vector{Float64}, Nothing}

    n_inter_all::Int
    inter_all::Union{Matrix{Int}, Nothing}
    para_inter_all::Union{Vector{ComplexF64}, Nothing}

    # Variational parameters
    n_gutzwiller_idx::Int
    gutzwiller_idx::Union{Vector{Int}, Nothing}
    n_jastrow_idx::Int
    jastrow_idx::Union{Matrix{Int}, Nothing}
    n_doublon_holon_2site_idx::Int
    doublon_holon_2site_idx::Union{Matrix{Int}, Nothing}
    n_doublon_holon_4site_idx::Int
    doublon_holon_4site_idx::Union{Matrix{Int}, Nothing}
    n_orbital_idx::Int
    orbital_idx::Union{Matrix{Int}, Nothing}
    orbital_sgn::Union{Matrix{Int}, Nothing}
    i_flg_orbital_general::Int
    i_n_orbital_parallel::Int
    i_n_orbital_anti_parallel::Int

    # RBM parameters
    n_neuron::Int
    n_neuron_general::Int
    n_neuron_charge::Int
    n_neuron_spin::Int
    n_rbm_hidden_layer_idx::Int
    n_rbm_phys_layer_idx::Int
    n_rbm_phys_hidden_idx::Int
    n_general_rbm_hidden_layer_idx::Int
    general_rbm_hidden_layer_idx::Union{Vector{Int}, Nothing}
    n_general_rbm_phys_layer_idx::Int
    general_rbm_phys_layer_idx::Union{Vector{Int}, Nothing}
    n_general_rbm_phys_hidden_idx::Int
    general_rbm_phys_hidden_idx::Union{Matrix{Int}, Nothing}
    n_charge_rbm_hidden_layer_idx::Int
    charge_rbm_hidden_layer_idx::Union{Vector{Int}, Nothing}
    n_charge_rbm_phys_layer_idx::Int
    charge_rbm_phys_layer_idx::Union{Vector{Int}, Nothing}
    n_charge_rbm_phys_hidden_idx::Int
    charge_rbm_phys_hidden_idx::Union{Matrix{Int}, Nothing}
    n_spin_rbm_hidden_layer_idx::Int
    spin_rbm_hidden_layer_idx::Union{Vector{Int}, Nothing}
    n_spin_rbm_phys_layer_idx::Int
    spin_rbm_phys_layer_idx::Union{Vector{Int}, Nothing}
    n_spin_rbm_phys_hidden_idx::Int
    spin_rbm_phys_hidden_idx::Union{Matrix{Int}, Nothing}
    n_block_size_rbm_ratio::Int

    # Quantum projection transformation
    n_qp_trans::Int
    qp_trans::Union{Matrix{Int}, Nothing}
    qp_trans_inv::Union{Matrix{Int}, Nothing}
    qp_trans_sgn::Union{Matrix{Int}, Nothing}
    para_qp_trans::Union{Vector{ComplexF64}, Nothing}

    # Optimal transformation
    n_qp_opt_trans::Int
    qp_opt_trans::Union{Matrix{Int}, Nothing}
    qp_opt_trans_sgn::Union{Matrix{Int}, Nothing}
    para_qp_opt_trans::Union{Vector{Float64}, Nothing}

    # Green functions
    n_cis_ajs::Int
    cis_ajs_idx::Union{Matrix{Int}, Nothing}
    n_cis_ajs_ckt_alt::Int
    cis_ajs_ckt_alt_idx::Union{Matrix{Int}, Nothing}
    n_cis_ajs_ckt_alt_dc::Int
    cis_ajs_ckt_alt_dc_idx::Union{Matrix{Int}, Nothing}
    i_one_body_g_idx::Union{Matrix{Int}, Nothing}

    # Optimization flags
    opt_flag::Union{Vector{Int}, Nothing}
    all_complex_flag::Int
    flag_rbm::Int
    ap_flag::Int  # 0: periodic, 1: anti-periodic

    # Shift flags
    flag_shift_gj::Int
    flag_shift_dh2::Int
    flag_shift_dh4::Int
    flag_opt_trans::Int
    flag_binary::Int
    n_file_flush_interval::Int

    # Variational parameters
    n_para::Int
    n_proj::Int
    n_rbm::Int
    n_proj_bf::Int
    n_slater::Int
    n_opt_trans::Int
    eta_flag::Union{Matrix{Int}, Nothing}
    para::Union{Vector{ComplexF64}, Nothing}
    proj::Union{Vector{ComplexF64}, Nothing}
    rbm::Union{Vector{ComplexF64}, Nothing}
    proj_bf::Union{Vector{ComplexF64}, Nothing}
    slater::Union{Vector{ComplexF64}, Nothing}
    opt_trans::Union{Vector{ComplexF64}, Nothing}
    eta::Union{Matrix{ComplexF64}, Nothing}

    # Back flow
    n_back_flow_idx::Int
    back_flow_idx::Union{Matrix{Int}, Nothing}
    n_range::Int
    pos_bf::Union{Matrix{Int}, Nothing}
    range_idx::Union{Matrix{Int}, Nothing}
    n_bf_idx_total::Int
    n_range_idx::Int
    bf_sub_idx::Union{Matrix{Int}, Nothing}

    # Electron configuration
    ele_idx::Union{Vector{Int}, Nothing}
    ele_cfg::Union{Vector{Int}, Nothing}
    ele_num::Union{Vector{Int}, Nothing}
    ele_proj_cnt::Union{Vector{Int}, Nothing}
    ele_spn::Union{Vector{Int}, Nothing}
    rbm_cnt::Union{Vector{ComplexF64}, Nothing}
    ele_proj_bf_cnt::Union{Vector{Int}, Nothing}
    log_sq_pf_full_slater::Union{Vector{Float64}, Nothing}
    smp_slt_elm_bf_real::Union{Vector{Float64}, Nothing}
    smp_eta_flag::Union{Vector{Int}, Nothing}
    smp_eta::Union{Vector{Float64}, Nothing}

    # Temporary arrays
    tmp_ele_idx::Union{Vector{Int}, Nothing}
    tmp_ele_cfg::Union{Vector{Int}, Nothing}
    tmp_ele_num::Union{Vector{Int}, Nothing}
    tmp_ele_proj_cnt::Union{Vector{Int}, Nothing}
    tmp_ele_spn::Union{Vector{Int}, Nothing}
    tmp_ele_proj_bf_cnt::Union{Vector{Int}, Nothing}
    tmp_rbm_cnt::Union{Vector{ComplexF64}, Nothing}

    # Burn arrays
    burn_ele_idx::Union{Vector{Int}, Nothing}
    burn_ele_cfg::Union{Vector{Int}, Nothing}
    burn_ele_num::Union{Vector{Int}, Nothing}
    burn_ele_proj_cnt::Union{Vector{Int}, Nothing}
    burn_ele_spn::Union{Vector{Int}, Nothing}
    burn_rbm_cnt::Union{Vector{ComplexF64}, Nothing}
    burn_flag::Int

    # Slater elements
    slater_elm::Union{Array{ComplexF64, 3}, Nothing}
    inv_m::Union{Array{ComplexF64, 3}, Nothing}
    pf_m::Union{Vector{ComplexF64}, Nothing}
    slater_elm_real::Union{Array{Float64, 3}, Nothing}
    inv_m_real::Union{Array{Float64, 3}, Nothing}
    pf_m_real::Union{Vector{Float64}, Nothing}
    slater_elm_bf::Union{Array{ComplexF64, 3}, Nothing}
    slater_elm_bf_real::Union{Array{Float64, 3}, Nothing}

    # Quantum projection
    qp_full_weight::Union{Vector{ComplexF64}, Nothing}
    qp_fix_weight::Union{Vector{ComplexF64}, Nothing}
    spgl_cos::Union{Vector{ComplexF64}, Nothing}
    spgl_sin::Union{Vector{ComplexF64}, Nothing}
    spgl_cos_sin::Union{Vector{ComplexF64}, Nothing}
    spgl_cos_cos::Union{Vector{ComplexF64}, Nothing}
    spgl_sin_sin::Union{Vector{ComplexF64}, Nothing}

    # Stochastic reconfiguration
    sr_opt_size::Int
    sr_opt_oo::Union{Matrix{ComplexF64}, Nothing}
    sr_opt_ho::Union{Vector{ComplexF64}, Nothing}
    sr_opt_o::Union{Vector{ComplexF64}, Nothing}
    sr_opt_o_store::Union{Matrix{ComplexF64}, Nothing}
    sr_opt_oo_real::Union{Matrix{Float64}, Nothing}
    sr_opt_ho_real::Union{Vector{Float64}, Nothing}
    sr_opt_o_real::Union{Vector{Float64}, Nothing}
    sr_opt_o_store_real::Union{Matrix{Float64}, Nothing}
    sr_opt_data::Union{Vector{ComplexF64}, Nothing}

    # Physical quantities
    wc::ComplexF64
    etot::ComplexF64
    etot2::ComplexF64
    dbtot::ComplexF64
    dbtot2::ComplexF64
    phys_cis_ajs::Union{Vector{ComplexF64}, Nothing}
    phys_cis_ajs_ckt_alt::Union{Vector{ComplexF64}, Nothing}
    phys_cis_ajs_ckt_alt_dc::Union{Vector{ComplexF64}, Nothing}
    local_cis_ajs::Union{Vector{ComplexF64}, Nothing}
    local_cis_ajs_ckt_alt_dc::Union{Vector{ComplexF64}, Nothing}
    sz_tot::ComplexF64
    sz_tot2::ComplexF64

    # Large scale calculation
    qqqq::Union{Array{ComplexF64, 4}, Nothing}
    lsl_q::Union{Matrix{ComplexF64}, Nothing}
    qqqq_real::Union{Array{Float64, 4}, Nothing}
    lsl_q_real::Union{Matrix{Float64}, Nothing}
    q_cis_ajs_q::Union{Array{ComplexF64, 3}, Nothing}
    q_cis_ajs_ckt_alt_q::Union{Array{ComplexF64, 3}, Nothing}
    q_cis_ajs_ckt_alt_q_dc::Union{Array{ComplexF64, 3}, Nothing}
    lsl_cis_ajs::Union{Matrix{ComplexF64}, Nothing}
    q_cis_ajs_q_real::Union{Array{Float64, 3}, Nothing}
    q_cis_ajs_ckt_alt_q_real::Union{Array{Float64, 3}, Nothing}
    q_cis_ajs_ckt_alt_q_dc_real::Union{Array{Float64, 3}, Nothing}
    lsl_cis_ajs_real::Union{Matrix{Float64}, Nothing}

    # Timing
    timer::Vector{Float64}
    timer_start::Vector{Float64}
    ccc::Vector{Float64}

    # SR optimization flag
    sr_flag::Int

    # OpenMP
    n_thread::Int

    # LAPACK
    lapack_lwork::Int

    # Counter for vmc_make
    counter::Vector{Int}
    counter_max::Int

    function MVMCGlobalState()
        new(
            "zvo",  # c_data_file_head
            "zqp",  # c_para_file_head
            0,      # nvmc_cal_mode
            0,      # nlanczos_mode
            0,      # nstore_o
            0,      # nsrcg
            0,      # n_data_idx_start
            1,      # n_data_qty_smp
            0,      # nsite
            0,      # ne
            0,      # nup
            0,      # nsize
            0,      # nsite2
            0,      # nz
            0,      # two_sz
            0,      # nsp_gauss_leg
            0,      # nsp_stot
            0,      # nmp_trans
            0,      # nqp_full
            0,      # nqp_fix
            0,      # nsr_opt_itr_step
            0,      # nsr_opt_itr_smp
            0,      # nsr_opt_fix_smp
            0.0,    # dsr_opt_red_cut
            0.0,    # dsr_opt_sta_del
            0.0,    # dsr_opt_step_dt
            0,      # nsr_opt_cg_max_iter
            0.0,    # dsr_opt_cg_tol
            0,      # nvmc_warm_up
            0,      # nvmc_interval
            0,      # nvmc_sample
            0,      # n_ex_update_path
            0,      # n_block_update_size
            0,      # rnd_seed
            0,      # n_split_size
            0,      # n_total_def_int
            0,      # n_total_def_double
            0,      # n_loc_spn
            nothing, # loc_spn
            0,      # n_transfer
            nothing, # transfer
            nothing, # para_transfer
            0,      # n_coulomb_intra
            nothing, # coulomb_intra
            nothing, # para_coulomb_intra
            0,      # n_coulomb_inter
            nothing, # coulomb_inter
            nothing, # para_coulomb_inter
            0,      # n_hund_coupling
            nothing, # hund_coupling
            nothing, # para_hund_coupling
            0,      # n_pair_hopping
            nothing, # pair_hopping
            nothing, # para_pair_hopping
            0,      # n_exchange_coupling
            nothing, # exchange_coupling
            nothing, # para_exchange_coupling
            0,      # n_inter_all
            nothing, # inter_all
            nothing, # para_inter_all
            0,      # n_gutzwiller_idx
            nothing, # gutzwiller_idx
            0,      # n_jastrow_idx
            nothing, # jastrow_idx
            0,      # n_doublon_holon_2site_idx
            nothing, # doublon_holon_2site_idx
            0,      # n_doublon_holon_4site_idx
            nothing, # doublon_holon_4site_idx
            0,      # n_orbital_idx
            nothing, # orbital_idx
            nothing, # orbital_sgn
            0,      # i_flg_orbital_general
            0,      # i_n_orbital_parallel
            0,      # i_n_orbital_anti_parallel
            0,      # n_neuron
            0,      # n_neuron_general
            0,      # n_neuron_charge
            0,      # n_neuron_spin
            0,      # n_rbm_hidden_layer_idx
            0,      # n_rbm_phys_layer_idx
            0,      # n_rbm_phys_hidden_idx
            0,      # n_general_rbm_hidden_layer_idx
            nothing, # general_rbm_hidden_layer_idx
            0,      # n_general_rbm_phys_layer_idx
            nothing, # general_rbm_phys_layer_idx
            0,      # n_general_rbm_phys_hidden_idx
            nothing, # general_rbm_phys_hidden_idx
            0,      # n_charge_rbm_hidden_layer_idx
            nothing, # charge_rbm_hidden_layer_idx
            0,      # n_charge_rbm_phys_layer_idx
            nothing, # charge_rbm_phys_layer_idx
            0,      # n_charge_rbm_phys_hidden_idx
            nothing, # charge_rbm_phys_hidden_idx
            0,      # n_spin_rbm_hidden_layer_idx
            nothing, # spin_rbm_hidden_layer_idx
            0,      # n_spin_rbm_phys_layer_idx
            nothing, # spin_rbm_phys_layer_idx
            0,      # n_spin_rbm_phys_hidden_idx
            nothing, # spin_rbm_phys_hidden_idx
            0,      # n_block_size_rbm_ratio
            0,      # n_qp_trans
            nothing, # qp_trans
            nothing, # qp_trans_inv
            nothing, # qp_trans_sgn
            nothing, # para_qp_trans
            0,      # n_qp_opt_trans
            nothing, # qp_opt_trans
            nothing, # qp_opt_trans_sgn
            nothing, # para_qp_opt_trans
            0,      # n_cis_ajs
            nothing, # cis_ajs_idx
            0,      # n_cis_ajs_ckt_alt
            nothing, # cis_ajs_ckt_alt_idx
            0,      # n_cis_ajs_ckt_alt_dc
            nothing, # cis_ajs_ckt_alt_dc_idx
            nothing, # i_one_body_g_idx
            nothing, # opt_flag
            0,      # all_complex_flag
            0,      # flag_rbm
            0,      # ap_flag
            0,      # flag_shift_gj
            0,      # flag_shift_dh2
            0,      # flag_shift_dh4
            0,      # flag_opt_trans
            0,      # flag_binary
            1,      # n_file_flush_interval
            0,      # n_para
            0,      # n_proj
            0,      # n_rbm
            0,      # n_proj_bf
            0,      # n_slater
            0,      # n_opt_trans
            nothing, # eta_flag
            nothing, # para
            nothing, # proj
            nothing, # rbm
            nothing, # proj_bf
            nothing, # slater
            nothing, # opt_trans
            nothing, # eta
            0,      # n_back_flow_idx
            nothing, # back_flow_idx
            0,      # n_range
            nothing, # pos_bf
            nothing, # range_idx
            0,      # n_bf_idx_total
            0,      # n_range_idx
            nothing, # bf_sub_idx
            nothing, # ele_idx
            nothing, # ele_cfg
            nothing, # ele_num
            nothing, # ele_proj_cnt
            nothing, # ele_spn
            nothing, # rbm_cnt
            nothing, # ele_proj_bf_cnt
            nothing, # log_sq_pf_full_slater
            nothing, # smp_slt_elm_bf_real
            nothing, # smp_eta_flag
            nothing, # smp_eta
            nothing, # tmp_ele_idx
            nothing, # tmp_ele_cfg
            nothing, # tmp_ele_num
            nothing, # tmp_ele_proj_cnt
            nothing, # tmp_ele_spn
            nothing, # tmp_ele_proj_bf_cnt
            nothing, # tmp_rbm_cnt
            nothing, # burn_ele_idx
            nothing, # burn_ele_cfg
            nothing, # burn_ele_num
            nothing, # burn_ele_proj_cnt
            nothing, # burn_ele_spn
            nothing, # burn_rbm_cnt
            0,      # burn_flag
            nothing, # slater_elm
            nothing, # inv_m
            nothing, # pf_m
            nothing, # slater_elm_real
            nothing, # inv_m_real
            nothing, # pf_m_real
            nothing, # slater_elm_bf
            nothing, # slater_elm_bf_real
            nothing, # qp_full_weight
            nothing, # qp_fix_weight
            nothing, # spgl_cos
            nothing, # spgl_sin
            nothing, # spgl_cos_sin
            nothing, # spgl_cos_cos
            nothing, # spgl_sin_sin
            0,      # sr_opt_size
            nothing, # sr_opt_oo
            nothing, # sr_opt_ho
            nothing, # sr_opt_o
            nothing, # sr_opt_o_store
            nothing, # sr_opt_oo_real
            nothing, # sr_opt_ho_real
            nothing, # sr_opt_o_real
            nothing, # sr_opt_o_store_real
            nothing, # sr_opt_data
            ComplexF64(0.0), # wc
            ComplexF64(0.0), # etot
            ComplexF64(0.0), # etot2
            ComplexF64(0.0), # dbtot
            ComplexF64(0.0), # dbtot2
            nothing, # phys_cis_ajs
            nothing, # phys_cis_ajs_ckt_alt
            nothing, # phys_cis_ajs_ckt_alt_dc
            nothing, # local_cis_ajs
            nothing, # local_cis_ajs_ckt_alt_dc
            ComplexF64(0.0), # sz_tot
            ComplexF64(0.0), # sz_tot2
            nothing, # qqqq
            nothing, # lsl_q
            nothing, # qqqq_real
            nothing, # lsl_q_real
            nothing, # q_cis_ajs_q
            nothing, # q_cis_ajs_ckt_alt_q
            nothing, # q_cis_ajs_ckt_alt_q_dc
            nothing, # lsl_cis_ajs
            nothing, # q_cis_ajs_q_real
            nothing, # q_cis_ajs_ckt_alt_q_real
            nothing, # q_cis_ajs_ckt_alt_q_dc_real
            nothing, # lsl_cis_ajs_real
            zeros(Float64, NTimer), # timer
            zeros(Float64, NTimer), # timer_start
            zeros(Float64, 100),    # ccc
            0,      # sr_flag
            1,      # n_thread
            0,      # lapack_lwork
            zeros(Int, 6), # counter
            6       # counter_max
        )
    end
end

"""
    init_timer!(state::MVMCGlobalState)

Initialize timing infrastructure.
Matches C function InitTimer().

C実装参考: global.h 1行目から100行目まで
"""
function init_timer!(state::MVMCGlobalState)
    fill!(state.timer, 0.0)
    fill!(state.timer_start, 0.0)
    fill!(state.ccc, 0.0)
end

"""
    start_timer!(state::MVMCGlobalState, timer_id::Int)

Start a timer.
Matches C function StartTimer().
"""
function start_timer!(state::MVMCGlobalState, timer_id::Int)
    if 1 <= timer_id <= NTimer
        state.timer_start[timer_id] = time()
    end
end

"""
    stop_timer!(state::MVMCGlobalState, timer_id::Int)

Stop a timer and accumulate time.
Matches C function StopTimer().
"""
function stop_timer!(state::MVMCGlobalState, timer_id::Int)
    if 1 <= timer_id <= NTimer
        state.timer[timer_id] += time() - state.timer_start[timer_id]
    end
end

"""
    get_timer(state::MVMCGlobalState, timer_id::Int) -> Float64

Get accumulated timer value.
"""
function get_timer(state::MVMCGlobalState, timer_id::Int)::Float64
    if 1 <= timer_id <= NTimer
        return state.timer[timer_id]
    end
    return 0.0
end

"""
    set_system_parameters!(state::MVMCGlobalState, nsite::Int, ne::Int, nup::Int)

Set basic system parameters.
"""
function set_system_parameters!(state::MVMCGlobalState, nsite::Int, ne::Int, nup::Int)
    state.nsite = nsite
    state.ne = ne
    state.nup = nup
    state.nsize = 2 * ne
    state.nsite2 = 2 * nsite
    state.two_sz = nup - ne
end

"""
    set_calculation_mode!(state::MVMCGlobalState, nvmc_cal_mode::Int, nlanczos_mode::Int)

Set calculation modes.
"""
function set_calculation_mode!(state::MVMCGlobalState, nvmc_cal_mode::Int, nlanczos_mode::Int)
    state.nvmc_cal_mode = nvmc_cal_mode
    state.nlanczos_mode = nlanczos_mode
end

"""
    set_file_options!(state::MVMCGlobalState, data_file_head::String, flag_binary::Bool, flush_interval::Int)

Set file output options.
"""
function set_file_options!(state::MVMCGlobalState, data_file_head::String, flag_binary::Bool, flush_interval::Int)
    state.c_data_file_head = data_file_head
    state.flag_binary = flag_binary ? 1 : 0
    state.n_file_flush_interval = flush_interval
end

"""
    set_sr_parameters!(state::MVMCGlobalState, itr_step::Int, itr_smp::Int, fix_smp::Int, red_cut::Float64, sta_del::Float64, step_dt::Float64)

Set stochastic reconfiguration parameters.
"""
function set_sr_parameters!(state::MVMCGlobalState, itr_step::Int, itr_smp::Int, fix_smp::Int, red_cut::Float64, sta_del::Float64, step_dt::Float64)
    state.nsr_opt_itr_step = itr_step
    state.nsr_opt_itr_smp = itr_smp
    state.nsr_opt_fix_smp = fix_smp
    state.dsr_opt_red_cut = red_cut
    state.dsr_opt_sta_del = sta_del
    state.dsr_opt_step_dt = step_dt
end

"""
    set_mc_parameters!(state::MVMCGlobalState, warm_up::Int, interval::Int, sample::Int, seed::Int)

Set Monte Carlo parameters.
"""
function set_mc_parameters!(state::MVMCGlobalState, warm_up::Int, interval::Int, sample::Int, seed::Int)
    state.nvmc_warm_up = warm_up
    state.nvmc_interval = interval
    state.nvmc_sample = sample
    state.rnd_seed = seed
end

"""
    print_state_summary(state::MVMCGlobalState)

Print a summary of the global state.
"""
function print_state_summary(state::MVMCGlobalState)
    println("=== MVMC Global State Summary ===")
    println("System: Nsite=$(state.nsite), Ne=$(state.ne), Nup=$(state.nup), 2Sz=$(state.two_sz)")
    println("Mode: NVMCCalMode=$(state.nvmc_cal_mode), NLanczosMode=$(state.nlanczos_mode)")
    println("Files: $(state.c_data_file_head), Binary=$(state.flag_binary)")
    println("SR: Steps=$(state.nsr_opt_itr_step), Samples=$(state.nsr_opt_itr_smp)")
    println("MC: WarmUp=$(state.nvmc_warm_up), Interval=$(state.nvmc_interval), Sample=$(state.nvmc_sample)")
    println("Parameters: NPara=$(state.n_para), NProj=$(state.n_proj), NSlater=$(state.n_slater)")
    println("=================================")
end

# Export functions and types
export MVMCGlobalState, init_timer!, start_timer!, stop_timer!, get_timer,
       set_system_parameters!, set_calculation_mode!, set_file_options!,
       set_sr_parameters!, set_mc_parameters!, print_state_summary
