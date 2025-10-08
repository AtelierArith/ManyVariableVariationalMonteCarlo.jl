"""
Hamiltonian Calculation System for mVMC C Compatibility

Translates the Hamiltonian calculation modules (calham*.c) to Julia,
maintaining exact compatibility with C numerical methods and energy calculations.

Ported from:
- calham.c: General Hamiltonian calculations
- calham_real.c: Real number version
- calham_fsz.c: Fixed Sz sector version
- calham_fsz_real.c: Fixed Sz sector real version
"""

using LinearAlgebra
using ..GlobalState: global_state, Nsite, Ne, Nup, Nsize, Nsite2, NQPFull, NQPFix,
                     NTransfer, Transfer, ParaTransfer, NCoulombIntra, CoulombIntra, ParaCoulombIntra,
                     NCoulombInter, CoulombInter, ParaCoulombInter, NHundCoupling, HundCoupling, ParaHundCoupling,
                     NPairHopping, PairHopping, ParaPairHopping, NExchangeCoupling, ExchangeCoupling, ParaExchangeCoupling,
                     NInterAll, InterAll, ParaInterAll, NProj, EleIdx, EleCfg, EleNum, EleProjCnt, RBMCnt

# Import local Green function functions
using ..LocGrnCompat: GreenFunc1, GreenFunc2, GreenFunc1_real, GreenFunc2_real

# Constants matching C implementation
const EPS = 1.0e-14
const MAX_ITER = 100

"""
    calculate_hamiltonian(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                        ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, rbm_cnt::Vector{ComplexF64})

Calculate total Hamiltonian energy.
Matches C function CalculateHamiltonian.

C実装参考: calham.c 32行目から522行目まで
"""
function calculate_hamiltonian(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                              ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, rbm_cnt::Vector{ComplexF64})
    # Calculate number operator terms (fast)
    e0 = calculate_hamiltonian0(ele_num)

    # Calculate Green function terms
    e1 = calculate_hamiltonian1(ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)

    return e0 + e1
end

"""
    calculate_hamiltonian0(ele_num::Vector{Int})

Calculate number operator terms (CoulombIntra, CoulombInter, Hund).
Matches C function CalculateHamiltonian0.

C実装参考: calham.c 37行目から78行目まで
"""
function calculate_hamiltonian0(ele_num::Vector{Int})
    n0 = ele_num[1:Nsite]  # Up electrons
    n1 = ele_num[(Nsite+1):Nsite2]  # Down electrons
    e = 0.0

    # CoulombIntra terms
    for idx in 1:NCoulombIntra
        ri = CoulombIntra[idx]
        e += ParaCoulombIntra[idx] * n0[ri] * n1[ri]
    end

    # CoulombInter terms
    for idx in 1:NCoulombInter
        ri = CoulombInter[idx][1]
        rj = CoulombInter[idx][2]
        e += ParaCoulombInter[idx] * (n0[ri] + n1[ri]) * (n0[rj] + n1[rj])
    end

    # Hund coupling terms
    for idx in 1:NHundCoupling
        ri = HundCoupling[idx][1]
        rj = HundCoupling[idx][2]
        e -= ParaHundCoupling[idx] * (n0[ri] * n0[rj] + n1[ri] * n1[rj])
    end

    return e
end

"""
    calculate_hamiltonian1(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                          ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Calculate Green function terms.
Matches C function CalculateHamiltonian1.

C実装参考: calham.c 79行目から522行目まで
"""
function calculate_hamiltonian1(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                               ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    e = ComplexF64(0.0)

    # Transfer terms
    for idx in 1:NTransfer
        ri = Transfer[idx][1]
        rj = Transfer[idx][2]
        s = Transfer[idx][3]
        t = Transfer[idx][4]

        e += ParaTransfer[idx] * GreenFunc1(ri, rj, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Pair hopping terms
    for idx in 1:NPairHopping
        ri = PairHopping[idx][1]
        rj = PairHopping[idx][2]

        e += ParaPairHopping[idx] * GreenFunc2(ri, rj, ri, rj, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Exchange coupling terms
    for idx in 1:NExchangeCoupling
        ri = ExchangeCoupling[idx][1]
        rj = ExchangeCoupling[idx][2]

        tmp = GreenFunc2(ri, rj, rj, ri, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        tmp += GreenFunc2(ri, rj, rj, ri, 1, 0, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        e += ParaExchangeCoupling[idx] * tmp
    end

    # InterAll terms
    for idx in 1:NInterAll
        ri = InterAll[idx][1]
        rj = InterAll[idx][2]
        s = InterAll[idx][3]
        rk = InterAll[idx][4]
        rl = InterAll[idx][5]
        t = InterAll[idx][6]

        e += ParaInterAll[idx] * GreenFunc2(ri, rj, rk, rl, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end

"""
    calculate_hamiltonian2(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                          ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Calculate Hamiltonian for Lanczos mode.
Matches C function CalculateHamiltonian2.
"""
function calculate_hamiltonian2(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    e = ComplexF64(0.0)

    # Pair hopping terms
    for idx in 1:NPairHopping
        ri = PairHopping[idx][1]
        rj = PairHopping[idx][2]

        e += ParaPairHopping[idx] * GreenFunc2(ri, rj, ri, rj, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Exchange coupling terms
    for idx in 1:NExchangeCoupling
        ri = ExchangeCoupling[idx][1]
        rj = ExchangeCoupling[idx][2]

        tmp = GreenFunc2(ri, rj, rj, ri, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        tmp += GreenFunc2(ri, rj, rj, ri, 1, 0, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        e += ParaExchangeCoupling[idx] * tmp
    end

    # InterAll terms
    for idx in 1:NInterAll
        ri = InterAll[idx][1]
        rj = InterAll[idx][2]
        s = InterAll[idx][3]
        rk = InterAll[idx][4]
        rl = InterAll[idx][5]
        t = InterAll[idx][6]

        e += ParaInterAll[idx] * GreenFunc2(ri, rj, rk, rl, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end

"""
    calculate_double_occupation(ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                               ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Calculate double occupation.
Matches C function CalculateDoubleOccupation.
"""
function calculate_double_occupation(ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                    ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    n0 = ele_num[1:Nsite]  # Up electrons
    n1 = ele_num[(Nsite+1):Nsite2]  # Down electrons

    double_occ = 0.0
    for ri in 1:Nsite
        double_occ += n0[ri] * n1[ri]
    end

    return double_occ
end

"""
    calculate_hamiltonian_bf_fcmp(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                  ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_proj_bf_cnt::Vector{Int})

Calculate Hamiltonian with backflow effects.
Matches C function CalculateHamiltonianBF_fcmp.
"""
function calculate_hamiltonian_bf_fcmp(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                        ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_proj_bf_cnt::Vector{Int})
    # Calculate number operator terms (fast)
    e0 = calculate_hamiltonian0(ele_num)

    # Calculate Green function terms with backflow
    e1 = calculate_hamiltonian1_bf(ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt, ele_proj_bf_cnt)

    return e0 + e1
end

"""
    calculate_hamiltonian1_bf(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                              ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_proj_bf_cnt::Vector{Int})

Calculate Green function terms with backflow.
"""
function calculate_hamiltonian1_bf(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                   ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_proj_bf_cnt::Vector{Int})
    e = ComplexF64(0.0)

    # Transfer terms with backflow
    for idx in 1:NTransfer
        ri = Transfer[idx][1]
        rj = Transfer[idx][2]
        s = Transfer[idx][3]
        t = Transfer[idx][4]

        e += ParaTransfer[idx] * GreenFunc1(ri, rj, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Pair hopping terms with backflow
    for idx in 1:NPairHopping
        ri = PairHopping[idx][1]
        rj = PairHopping[idx][2]

        e += ParaPairHopping[idx] * GreenFunc2(ri, rj, ri, rj, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Exchange coupling terms with backflow
    for idx in 1:NExchangeCoupling
        ri = ExchangeCoupling[idx][1]
        rj = ExchangeCoupling[idx][2]

        tmp = GreenFunc2(ri, rj, rj, ri, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        tmp += GreenFunc2(ri, rj, rj, ri, 1, 0, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        e += ParaExchangeCoupling[idx] * tmp
    end

    # InterAll terms with backflow
    for idx in 1:NInterAll
        ri = InterAll[idx][1]
        rj = InterAll[idx][2]
        s = InterAll[idx][3]
        rk = InterAll[idx][4]
        rl = InterAll[idx][5]
        t = InterAll[idx][6]

        e += ParaInterAll[idx] * GreenFunc2(ri, rj, rk, rl, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end

# Real number versions

"""
    calculate_hamiltonian_real(ip::Float64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                              ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Real version of Hamiltonian calculation.
Matches C function CalculateHamiltonian_real.
"""
function calculate_hamiltonian_real(ip::Float64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                    ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    # Calculate number operator terms (fast)
    e0 = calculate_hamiltonian0_real(ele_num)

    # Calculate Green function terms
    e1 = calculate_hamiltonian1_real(ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)

    return e0 + e1
end

"""
    calculate_hamiltonian0_real(ele_num::Vector{Int})

Real version of number operator terms.
Matches C function CalculateHamiltonian0_real.
"""
function calculate_hamiltonian0_real(ele_num::Vector{Int})
    n0 = ele_num[1:Nsite]  # Up electrons
    n1 = ele_num[(Nsite+1):Nsite2]  # Down electrons
    e = 0.0

    # CoulombIntra terms
    for idx in 1:NCoulombIntra
        ri = CoulombIntra[idx]
        e += ParaCoulombIntra[idx] * n0[ri] * n1[ri]
    end

    # CoulombInter terms
    for idx in 1:NCoulombInter
        ri = CoulombInter[idx][1]
        rj = CoulombInter[idx][2]
        e += ParaCoulombInter[idx] * (n0[ri] + n1[ri]) * (n0[rj] + n1[rj])
    end

    # Hund coupling terms
    for idx in 1:NHundCoupling
        ri = HundCoupling[idx][1]
        rj = HundCoupling[idx][2]
        e -= ParaHundCoupling[idx] * (n0[ri] * n0[rj] + n1[ri] * n1[rj])
    end

    return e
end

"""
    calculate_hamiltonian1_real(ip::Float64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Real version of Green function terms.
Matches C function CalculateHamiltonian1_real.
"""
function calculate_hamiltonian1_real(ip::Float64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                     ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    e = 0.0

    # Transfer terms
    for idx in 1:NTransfer
        ri = Transfer[idx][1]
        rj = Transfer[idx][2]
        s = Transfer[idx][3]
        t = Transfer[idx][4]

        e += real(ParaTransfer[idx]) * GreenFunc1_real(ri, rj, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Pair hopping terms
    for idx in 1:NPairHopping
        ri = PairHopping[idx][1]
        rj = PairHopping[idx][2]

        e += ParaPairHopping[idx] * GreenFunc2_real(ri, rj, ri, rj, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Exchange coupling terms
    for idx in 1:NExchangeCoupling
        ri = ExchangeCoupling[idx][1]
        rj = ExchangeCoupling[idx][2]

        tmp = GreenFunc2_real(ri, rj, rj, ri, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        tmp += GreenFunc2_real(ri, rj, rj, ri, 1, 0, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        e += ParaExchangeCoupling[idx] * tmp
    end

    # InterAll terms
    for idx in 1:NInterAll
        ri = InterAll[idx][1]
        rj = InterAll[idx][2]
        s = InterAll[idx][3]
        rk = InterAll[idx][4]
        rl = InterAll[idx][5]
        t = InterAll[idx][6]

        e += real(ParaInterAll[idx]) * GreenFunc2_real(ri, rj, rk, rl, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end

# Fixed Sz sector versions

"""
    calculate_hamiltonian_fsz(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                              ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_spn::Vector{Int})

Fixed Sz sector version of Hamiltonian calculation.
Matches C function CalculateHamiltonian_fsz.
"""
function calculate_hamiltonian_fsz(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                   ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_spn::Vector{Int})
    # Calculate number operator terms (fast)
    e0 = calculate_hamiltonian0_fsz(ele_num, ele_spn)

    # Calculate Green function terms
    e1 = calculate_hamiltonian1_fsz(ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt, ele_spn)

    return e0 + e1
end

"""
    calculate_hamiltonian0_fsz(ele_num::Vector{Int}, ele_spn::Vector{Int})

Fixed Sz sector version of number operator terms.
Matches C function CalculateHamiltonian0_fsz.
"""
function calculate_hamiltonian0_fsz(ele_num::Vector{Int}, ele_spn::Vector{Int})
    e = 0.0

    # CoulombIntra terms
    for idx in 1:NCoulombIntra
        ri = CoulombIntra[idx]
        e += ParaCoulombIntra[idx] * ele_num[ri] * ele_num[ri + Nsite]
    end

    # CoulombInter terms
    for idx in 1:NCoulombInter
        ri = CoulombInter[idx][1]
        rj = CoulombInter[idx][2]
        e += ParaCoulombInter[idx] * (ele_num[ri] + ele_num[ri + Nsite]) * (ele_num[rj] + ele_num[rj + Nsite])
    end

    # Hund coupling terms
    for idx in 1:NHundCoupling
        ri = HundCoupling[idx][1]
        rj = HundCoupling[idx][2]
        e -= ParaHundCoupling[idx] * (ele_num[ri] * ele_num[rj] + ele_num[ri + Nsite] * ele_num[rj + Nsite])
    end

    return e
end

"""
    calculate_hamiltonian1_fsz(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                              ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_spn::Vector{Int})

Fixed Sz sector version of Green function terms.
Matches C function CalculateHamiltonian1_fsz.
"""
function calculate_hamiltonian1_fsz(ip::ComplexF64, ele_idx::Vector{Int}, ele_cfg::Vector{Int},
                                    ele_num::Vector{Int}, ele_proj_cnt::Vector{Int}, ele_spn::Vector{Int})
    e = ComplexF64(0.0)

    # Transfer terms
    for idx in 1:NTransfer
        ri = Transfer[idx][1]
        rj = Transfer[idx][2]
        s = Transfer[idx][3]
        t = Transfer[idx][4]

        e += ParaTransfer[idx] * GreenFunc1(ri, rj, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Pair hopping terms
    for idx in 1:NPairHopping
        ri = PairHopping[idx][1]
        rj = PairHopping[idx][2]

        e += ParaPairHopping[idx] * GreenFunc2(ri, rj, ri, rj, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Exchange coupling terms
    for idx in 1:NExchangeCoupling
        ri = ExchangeCoupling[idx][1]
        rj = ExchangeCoupling[idx][2]

        tmp = GreenFunc2(ri, rj, rj, ri, 0, 1, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        tmp += GreenFunc2(ri, rj, rj, ri, 1, 0, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        e += ParaExchangeCoupling[idx] * tmp
    end

    # InterAll terms
    for idx in 1:NInterAll
        ri = InterAll[idx][1]
        rj = InterAll[idx][2]
        s = InterAll[idx][3]
        rk = InterAll[idx][4]
        rl = InterAll[idx][5]
        t = InterAll[idx][6]

        e += ParaInterAll[idx] * GreenFunc2(ri, rj, rk, rl, s, t, ip, ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end

# Utility functions

"""
    calculate_kinetic_energy(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Calculate kinetic energy from transfer terms.
"""
function calculate_kinetic_energy(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    e = ComplexF64(0.0)

    for idx in 1:NTransfer
        ri = Transfer[idx][1]
        rj = Transfer[idx][2]
        s = Transfer[idx][3]
        t = Transfer[idx][4]

        e += ParaTransfer[idx] * GreenFunc1(ri, rj, s, t, ComplexF64(1.0), ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end

"""
    calculate_potential_energy(ele_num::Vector{Int})

Calculate potential energy from number operators.
"""
function calculate_potential_energy(ele_num::Vector{Int})
    return calculate_hamiltonian0(ele_num)
end

"""
    calculate_interaction_energy(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})

Calculate interaction energy from Green functions.
"""
function calculate_interaction_energy(ele_idx::Vector{Int}, ele_cfg::Vector{Int}, ele_num::Vector{Int}, ele_proj_cnt::Vector{Int})
    e = ComplexF64(0.0)

    # Pair hopping terms
    for idx in 1:NPairHopping
        ri = PairHopping[idx][1]
        rj = PairHopping[idx][2]

        e += ParaPairHopping[idx] * GreenFunc2(ri, rj, ri, rj, 0, 1, ComplexF64(1.0), ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    # Exchange coupling terms
    for idx in 1:NExchangeCoupling
        ri = ExchangeCoupling[idx][1]
        rj = ExchangeCoupling[idx][2]

        tmp = GreenFunc2(ri, rj, rj, ri, 0, 1, ComplexF64(1.0), ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        tmp += GreenFunc2(ri, rj, rj, ri, 1, 0, ComplexF64(1.0), ele_idx, ele_cfg, ele_num, ele_proj_cnt)
        e += ParaExchangeCoupling[idx] * tmp
    end

    # InterAll terms
    for idx in 1:NInterAll
        ri = InterAll[idx][1]
        rj = InterAll[idx][2]
        s = InterAll[idx][3]
        rk = InterAll[idx][4]
        rl = InterAll[idx][5]
        t = InterAll[idx][6]

        e += ParaInterAll[idx] * GreenFunc2(ri, rj, rk, rl, s, t, ComplexF64(1.0), ele_idx, ele_cfg, ele_num, ele_proj_cnt)
    end

    return e
end
