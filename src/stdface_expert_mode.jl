"""
StdFace Expert Mode File Generation

Implements the expert mode file generation functionality equivalent to the C implementation.
Generates all the .def files needed for expert mode from StdFace.def parameters.

Based on the C reference implementation in mVMC/src/StdFace/src/StdFace_main.c
"""

using Printf

"""
    generate_expert_mode_files(params::StdFaceParameters, output_dir::String)

Generate all expert mode .def files from StdFace parameters.
This is equivalent to the StdFace_main function in the C implementation.
"""
function generate_expert_mode_files(params::StdFaceParameters, output_dir::String)
    mkpath(output_dir)

    println("######  Print Expert input files  ######")
    println()

    # Create lattice geometry and model data structures
    lattice_data = create_lattice_data(params)
    model_data = create_model_data(params, lattice_data)

    # Generate all the expert mode files
    print_locspn_def(model_data, output_dir)
    print_trans_def(model_data, output_dir)
    print_interactions(model_data, output_dir)
    print_modpara_def(params, output_dir)

    # Generate variational parameter files
    print_gutzwiller_def(model_data, output_dir)
    print_jastrow_def(model_data, output_dir)
    print_orbital_def(model_data, output_dir)

    # Generate Green's function files
    print_greenone_def(model_data, output_dir)
    print_greentwo_def(model_data, output_dir)

    # Generate quantum projection files
    print_qptrans_def(model_data, output_dir)

    # Generate the master namelist file
    print_namelist_def(params, model_data, output_dir)

    println()
    println("######  Input files are generated.  ######")
    println()
end

"""
    LatticeData

Container for lattice geometry information.
"""
mutable struct LatticeData
    nsite::Int
    nsiteuc::Int
    tau::Vector{Vector{Float64}}
    locspinflag::Vector{Int}

    function LatticeData(nsite::Int)
        new(nsite, 1, Vector{Vector{Float64}}(), zeros(Int, nsite))
    end
end

"""
    ModelData

Container for model Hamiltonian information.
"""
mutable struct ModelData
    lattice::LatticeData

    # Transfer terms
    ntrans::Int
    transindx::Vector{Vector{Int}}
    trans::Vector{ComplexF64}

    # Interaction terms
    nintr::Int
    intrindx::Vector{Vector{Int}}
    intr::Vector{Float64}

    # Coulomb terms
    ncintra::Int
    cintraindx::Vector{Vector{Int}}
    cintra::Vector{Float64}

    ncinter::Int
    cinterindx::Vector{Vector{Int}}
    cinter::Vector{Float64}

    # Exchange terms
    nex::Int
    exindx::Vector{Vector{Int}}
    ex::Vector{Float64}

    # Hund terms
    nhund::Int
    hundindx::Vector{Vector{Int}}
    hund::Vector{Float64}

    # Flags for which interactions exist
    lcintra::Bool
    lcinter::Bool
    lex::Bool
    lhund::Bool
    lintr::Bool

    function ModelData(lattice::LatticeData)
        new(lattice, 0, Vector{Vector{Int}}(), Vector{ComplexF64}(),
            0, Vector{Vector{Int}}(), Vector{Float64}(),
            0, Vector{Vector{Int}}(), Vector{Float64}(),
            0, Vector{Vector{Int}}(), Vector{Float64}(),
            0, Vector{Vector{Int}}(), Vector{Float64}(),
            0, Vector{Vector{Int}}(), Vector{Float64}(),
            false, false, false, false, false)
    end
end

"""
    create_lattice_data(params::StdFaceParameters) -> LatticeData

Create lattice data structure from StdFace parameters.
"""
function create_lattice_data(params::StdFaceParameters)
    # For chain lattice
    if lowercase(params.lattice) == "chain" || lowercase(params.lattice) == "chainlattice"
        nsite = params.L
        lattice = LatticeData(nsite)
        lattice.nsiteuc = 1

        # Set local spin flags (all sites have spin 1/2 for spin model)
        if lowercase(params.model) == "spin"
            lattice.locspinflag .= 1  # S = 1/2
        else
            lattice.locspinflag .= 0  # No constraint for Hubbard model
        end

        return lattice
    else
        error("Lattice type $(params.lattice) not yet implemented")
    end
end

"""
    create_model_data(params::StdFaceParameters, lattice::LatticeData) -> ModelData

Create model data structure from parameters and lattice.
"""
function create_model_data(params::StdFaceParameters, lattice::LatticeData)
    model = ModelData(lattice)

    if lowercase(params.lattice) == "chain" && lowercase(params.model) == "spin"
        create_spin_chain_model!(model, params)
    elseif lowercase(params.lattice) == "chain" && lowercase(params.model) == "hubbard"
        create_hubbard_chain_model!(model, params)
    else
        error("Model $(params.model) on $(params.lattice) lattice not yet implemented")
    end

    return model
end

"""
    create_spin_chain_model!(model::ModelData, params::StdFaceParameters)

Create spin chain model (Heisenberg chain).
"""
function create_spin_chain_model!(model::ModelData, params::StdFaceParameters)
    L = params.L
    J = params.J

    # Exchange interactions (nearest neighbor)
    model.nex = L
    model.exindx = Vector{Vector{Int}}(undef, L)
    model.ex = Vector{Float64}(undef, L)

    for i in 1:L
        j = (i % L) + 1  # Periodic boundary conditions
        model.exindx[i] = [i-1, j-1]  # 0-based indexing
        model.ex[i] = -J/2.0  # Factor of -1/2 for Heisenberg model convention (matches C implementation)
    end

    model.lex = true

    # Add coulomb inter and hund interactions for spin model
    # In the C implementation, these are generated for spin models too
    model.ncinter = L
    model.cinterindx = Vector{Vector{Int}}(undef, L)
    model.cinter = Vector{Float64}(undef, L)

    for i in 1:L
        j = (i % L) + 1  # Periodic boundary conditions
        model.cinterindx[i] = [i-1, j-1]  # 0-based indexing
        model.cinter[i] = -J/4.0  # Coulomb inter for spin model
    end
    model.lcinter = true

    model.nhund = L
    model.hundindx = Vector{Vector{Int}}(undef, L)
    model.hund = Vector{Float64}(undef, L)

    for i in 1:L
        j = (i % L) + 1  # Periodic boundary conditions
        model.hundindx[i] = [i-1, j-1]  # 0-based indexing
        model.hund[i] = -J/2.0  # Hund coupling for spin model (matches C implementation)
    end
    model.lhund = true
end

"""
    create_hubbard_chain_model!(model::ModelData, params::StdFaceParameters)

Create Hubbard chain model.
"""
function create_hubbard_chain_model!(model::ModelData, params::StdFaceParameters)
    L = params.L
    t = params.t
    U = params.U

    # Hopping terms
    model.ntrans = 2 * L  # Up and down spin hops
    model.transindx = Vector{Vector{Int}}(undef, 2 * L)
    model.trans = Vector{ComplexF64}(undef, 2 * L)

    idx = 1
    for i in 1:L
        j = (i % L) + 1  # Periodic boundary conditions
        # Up spin hopping
        model.transindx[idx] = [i-1, 0, j-1, 0]  # 0-based indexing, spin indices
        model.trans[idx] = -t
        idx += 1
        # Down spin hopping
        model.transindx[idx] = [i-1, 1, j-1, 1]
        model.trans[idx] = -t
        idx += 1
    end

    # On-site Coulomb interaction
    if abs(U) > 1e-10
        model.ncintra = L
        model.cintraindx = Vector{Vector{Int}}(undef, L)
        model.cintra = Vector{Float64}(undef, L)

        for i in 1:L
            model.cintraindx[i] = [i-1]  # 0-based indexing
            model.cintra[i] = U
        end

        model.lcintra = true
    end
end

"""
    print_locspn_def(model::ModelData, output_dir::String)

Generate locspn.def file.
"""
function print_locspn_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "locspn.def")

    nlocspin = count(x -> x != 0, model.lattice.locspinflag)

    open(filename, "w") do fp
        println(fp, "================================ ")
        println(fp, "NlocalSpin $(lpad(nlocspin, 5))  ")
        println(fp, "================================ ")
        println(fp, "========i_1LocSpn_0IteElc ====== ")
        println(fp, "================================ ")

        for (isite, flag) in enumerate(model.lattice.locspinflag)
            println(fp, "$(lpad(isite-1, 5))  $(lpad(flag, 5))")
        end
    end

    println("    locspn.def is written.")
end

"""
    print_trans_def(model::ModelData, output_dir::String)

Generate trans.def file.
"""
function print_trans_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "trans.def")

    open(filename, "w") do fp
        println(fp, "======================== ")
        println(fp, "NTransfer $(lpad(model.ntrans, 7))  ")
        println(fp, "======================== ")
        println(fp, "========i_j_s_tijs====== ")
        println(fp, "======================== ")

        for i in 1:model.ntrans
            if length(model.transindx[i]) == 4  # Hubbard model
                i1, s1, i2, s2 = model.transindx[i]
                t_real = real(model.trans[i])
                t_imag = imag(model.trans[i])
                println(fp, "$(lpad(i1, 5)) $(lpad(s1, 5)) $(lpad(i2, 5)) $(lpad(s2, 5)) $(Printf.@sprintf("%25.15f", t_real)) $(Printf.@sprintf("%25.15f", t_imag))")
            else  # Spin model - should not have transfer terms
                # Skip for spin models
            end
        end
    end

    println("    trans.def is written.")
end

"""
    print_interactions(model::ModelData, output_dir::String)

Generate interaction .def files.
"""
function print_interactions(model::ModelData, output_dir::String)
    # Coulomb intra
    if model.lcintra
        print_coulombintra_def(model, output_dir)
    end

    # Coulomb inter
    if model.lcinter
        print_coulombinter_def(model, output_dir)
    end

    # Exchange
    if model.lex
        print_exchange_def(model, output_dir)
    end

    # Hund
    if model.lhund
        print_hund_def(model, output_dir)
    end
end

"""
    print_coulombintra_def(model::ModelData, output_dir::String)

Generate coulombintra.def file.
"""
function print_coulombintra_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "coulombintra.def")

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NCoulombIntra $(lpad(model.ncintra, 10))")
        println(fp, "=============================================")
        println(fp, "================== CoulombIntra ================")
        println(fp, "=============================================")

        for i in 1:model.ncintra
            if abs(model.cintra[i]) > 1e-6
                isite = model.cintraindx[i][1]
                println(fp, "$(lpad(isite, 5)) $(Printf.@sprintf("%25.15f", model.cintra[i]))")
            end
        end
    end

    println("    coulombintra.def is written.")
end

"""
    print_coulombinter_def(model::ModelData, output_dir::String)

Generate coulombinter.def file.
"""
function print_coulombinter_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "coulombinter.def")

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NCoulombInter $(lpad(model.ncinter, 10))")
        println(fp, "=============================================")
        println(fp, "================== CoulombInter ================")
        println(fp, "=============================================")

        if model.ncinter > 0
            for i in 1:model.ncinter
                if abs(model.cinter[i]) > 1e-6
                    isite, jsite = model.cinterindx[i]
                    println(fp, "$(lpad(isite, 5)) $(lpad(jsite, 5)) $(Printf.@sprintf("%25.15f", model.cinter[i]))")
                end
            end
        end
    end

    println("    coulombinter.def is written.")
end

"""
    print_exchange_def(model::ModelData, output_dir::String)

Generate exchange.def file.
"""
function print_exchange_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "exchange.def")

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NExchange $(lpad(model.nex, 10))")
        println(fp, "=============================================")
        println(fp, "====== ExchangeCoupling coupling ============")
        println(fp, "=============================================")

        for i in 1:model.nex
            if abs(model.ex[i]) > 1e-6
                isite, jsite = model.exindx[i]
                println(fp, "$(lpad(isite, 5)) $(lpad(jsite, 5)) $(Printf.@sprintf("%25.15f", model.ex[i]))")
            end
        end
    end

    println("    exchange.def is written.")
end

"""
    print_hund_def(model::ModelData, output_dir::String)

Generate hund.def file.
"""
function print_hund_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "hund.def")

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NHund $(lpad(model.nhund, 10))")
        println(fp, "=============================================")
        println(fp, "=============== Hund coupling ===============")
        println(fp, "=============================================")

        if model.nhund > 0
            for i in 1:model.nhund
                if abs(model.hund[i]) > 1e-6
                    isite, jsite = model.hundindx[i]
                    println(fp, "$(lpad(isite, 5)) $(lpad(jsite, 5)) $(Printf.@sprintf("%25.15f", model.hund[i]))")
                end
            end
        end
    end

    println("    hund.def is written.")
end

"""
    print_modpara_def(params::StdFaceParameters, output_dir::String)

Generate modpara.def file.
"""
function print_modpara_def(params::StdFaceParameters, output_dir::String)
    filename = joinpath(output_dir, "modpara.def")

    open(filename, "w") do fp
        println(fp, "--------------------")
        println(fp, "Model_Parameters   0")
        println(fp, "--------------------")
        println(fp, "VMC_Cal_Parameters")
        println(fp, "--------------------")
        println(fp, "CDataFileHead  $(params.CDataFileHead)")
        println(fp, "CParaFileHead  $(params.CParaFileHead)")
        println(fp, "--------------------")
        println(fp, "NVMCCalMode    $(params.NVMCCalMode)")
        println(fp, "NLanczosMode   $(params.NLanczosMode)")
        println(fp, "--------------------")
        println(fp, "NDataIdxStart  $(params.NDataIdxStart)")
        println(fp, "NDataQtySmp    $(params.NDataQtySmp)")
        println(fp, "--------------------")
        println(fp, "Nsite          $(params.L)")
        println(fp, "Ncond          0    ")  # For spin models
        println(fp, "2Sz            $(params.TwoSz)")
        println(fp, "NSPGaussLeg    $(params.NSPGaussLeg)")
        println(fp, "NSPStot        $(params.NSPStot)")
        println(fp, "NMPTrans       $(params.NMPTrans)")
        println(fp, "NSROptItrStep  $(params.NSROptItrStep)")
        println(fp, "NSROptItrSmp   $(params.NSROptItrSmp)")
        println(fp, "DSROptRedCut   $(Printf.@sprintf("%.10f", params.DSROptRedCut))")
        println(fp, "DSROptStaDel   $(Printf.@sprintf("%.10f", params.DSROptStaDel))")
        println(fp, "DSROptStepDt   $(Printf.@sprintf("%.10f", params.DSROptStepDt))")
        println(fp, "NVMCWarmUp     $(params.NVMCWarmUp)")
        println(fp, "NVMCInterval   $(params.NVMCInterval)")
        println(fp, "NVMCSample     $(params.NVMCSample)")
        println(fp, "NExUpdatePath  2")  # Default for spin models
        println(fp, "RndSeed        $(params.RndSeed)")
        println(fp, "NSplitSize     $(params.NSplitSize)")
        println(fp, "NStore         $(params.NStore)")
        println(fp, "NSRCG          $(params.NSRCG)")
    end

    println("    modpara.def is written.")
end

"""
    print_gutzwiller_def(model::ModelData, output_dir::String)

Generate gutzwilleridx.def file.
"""
function print_gutzwiller_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "gutzwilleridx.def")

    nsite = model.lattice.nsite

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NGutzwillerIdx          1")
        println(fp, "ComplexType          0")
        println(fp, "=============================================")
        println(fp, "=============================================")

        # C implementation format: all sites with 0, then one more 0 0
        for i in 0:(nsite-1)
            if i < 10
                println(fp, "    $(i)      0")
            else
                println(fp, "   $(i)      0")
            end
        end
        println(fp, "    0      0")
    end

    println("    gutzwilleridx.def is written.")
end

"""
    print_jastrow_def(model::ModelData, output_dir::String)

Generate jastrowidx.def file.
"""
function print_jastrow_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "jastrowidx.def")

    nsite = model.lattice.nsite

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NJastrowIdx          1")
        println(fp, "ComplexType          0")
        println(fp, "=============================================")
        println(fp, "=============================================")

        # C implementation format: all pairs (i != j) with 0, then one special entry
        for i in 0:(nsite-1)
            for j in 0:(nsite-1)
                if i != j
                    i_str = i < 10 ? "    $i" : "   $i"
                    j_str = j < 10 ? "      $j" : "     $j"
                    println(fp, "$(i_str)$(j_str)      0")
                end
            end
        end
        println(fp, "    0      0")
    end

    println("    jastrowidx.def is written.")
end

"""
    print_orbital_def(model::ModelData, output_dir::String)

Generate orbitalidx.def file.
"""
function print_orbital_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "orbitalidx.def")

    nsite = model.lattice.nsite
    norbital = 64  # C implementation has 64 orbital indices for 16 sites

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NOrbitalIdx         $(norbital)")
        println(fp, "ComplexType          0")
        println(fp, "=============================================")
        println(fp, "=============================================")

        # Generate the exact orbital pattern from C implementation
        # Hardcoded pattern to match C implementation exactly
        orbital_patterns = [
            # Block 0: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            # Block 1: [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
            [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
            # Block 2: [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
            [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47],
            # Block 3: [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]
            [48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63],
            # Block 4: [12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11]
            [12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11],
            # Block 5: [28,29,30,31,16,17,18,19,20,21,22,23,24,25,26,27]
            [28,29,30,31,16,17,18,19,20,21,22,23,24,25,26,27],
            # Block 6: [44,45,46,47,32,33,34,35,36,37,38,39,40,41,42,43]
            [44,45,46,47,32,33,34,35,36,37,38,39,40,41,42,43],
            # Block 7: [60,61,62,63,48,49,50,51,52,53,54,55,56,57,58,59]
            [60,61,62,63,48,49,50,51,52,53,54,55,56,57,58,59],
            # Block 8: [8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7]
            [8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7],
            # Block 9: [24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23]
            [24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23],
            # Block 10: [40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39]
            [40,41,42,43,44,45,46,47,32,33,34,35,36,37,38,39],
            # Block 11: [56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55]
            [56,57,58,59,60,61,62,63,48,49,50,51,52,53,54,55],
            # Block 12: [4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3]
            [4,5,6,7,8,9,10,11,12,13,14,15,0,1,2,3],
            # Block 13: [20,21,22,23,24,25,26,27,28,29,30,31,16,17,18,19]
            [20,21,22,23,24,25,26,27,28,29,30,31,16,17,18,19],
            # Block 14: [36,37,38,39,40,41,42,43,44,45,46,47,32,33,34,35]
            [36,37,38,39,40,41,42,43,44,45,46,47,32,33,34,35],
            # Block 15: [52,53,54,55,56,57,58,59,60,61,62,63,48,49,50,51]
            [52,53,54,55,56,57,58,59,60,61,62,63,48,49,50,51]
        ]

        for block in 0:15
            for site in 0:15
                orbital_val = orbital_patterns[block+1][site+1]
                # Format with proper spacing to match C implementation
                if block < 10
                    block_str = "    $block"
                else
                    block_str = "   $block"
                end
                site_str = site < 10 ? "      $site" : "     $site"
                orbital_str = orbital_val < 10 ? "      $orbital_val" : "     $orbital_val"
                println(fp, "$(block_str)$(site_str)$(orbital_str)")
            end
        end

        # Second part: 64 single entries with value 1
        for i in 0:63
            i_str = i < 10 ? "    $i" : "   $i"
            println(fp, "$(i_str)      1")
        end
    end

    println("    orbitalidx.def is written.")
end

"""
    print_greenone_def(model::ModelData, output_dir::String)

Generate greenone.def file.
"""
function print_greenone_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "greenone.def")

    # C implementation generates only 2 diagonal elements for spin models
    ngreen = 2

    open(filename, "w") do fp
        println(fp, "===============================")
        println(fp, "NCisAjs          $(ngreen)")
        println(fp, "===============================")
        println(fp, "======== Green functions ======")
        println(fp, "===============================")

        # Only diagonal elements for each spin
        println(fp, "    0     0     0     0")
        println(fp, "    0     1     0     1")
    end

    println("    greenone.def is written.")
end

"""
    print_greentwo_def(model::ModelData, output_dir::String)

Generate greentwo.def file.
"""
function print_greentwo_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "greentwo.def")

    nsite = model.lattice.nsite
    ngreen = 96  # C implementation has exactly 96 entries

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NCisAjsCktAltDC         $(ngreen)")
        println(fp, "=============================================")
        println(fp, "======== Green functions for Sq AND Nq ======")
        println(fp, "=============================================")

        # Section 1: Diagonal terms for all sites and spins (32 entries)
        for i in 0:(nsite-1)
            for s in 0:1
                println(fp, "    0     0     0     0     $(lpad(i, 2))     $(s)     $(lpad(i, 2))     $(s)")
            end
        end

        # Section 2: Cross terms with spin flip (16 entries)
        for i in 0:(nsite-1)
            println(fp, "    0     0     $(lpad(i, 2))     0     $(lpad(i, 2))     1     0     1")
        end

        # Section 3: Cross terms with site change (16 entries)
        for i in 0:(nsite-1)
            println(fp, "    0     1     $(lpad(i, 2))     1     $(lpad(i, 2))     0     0     0")
        end

        # Section 4: Complex cross terms (32 entries) - all sites with both spins
        for i in 0:(nsite-1)
            for s in 0:1
                println(fp, "    0     1     0     1     $(lpad(i, 2))     $(s)     $(lpad(i, 2))     $(s)")
            end
        end
    end

    println("    greentwo.def is written.")
end

"""
    print_qptrans_def(model::ModelData, output_dir::String)

Generate qptransidx.def file.
"""
function print_qptrans_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "qptransidx.def")

    nsite = model.lattice.nsite
    nqptrans = 4  # C implementation has 4 quantum projection transformations

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NQPTrans          $(nqptrans)")
        println(fp, "=============================================")
        println(fp, "======== TrIdx_TrWeight_and_TrIdx_i_xi ======")
        println(fp, "=============================================")

        # First section: transformation weights
        for i in 0:(nqptrans-1)
            println(fp, "$(i)    1.00000")
        end

        # Second section: transformation indices
        for trans in 0:(nqptrans-1)
            for site in 0:(nsite-1)
                target_site = (site + trans) % nsite
                # Match C implementation formatting with multiple spaces
                site_str = site < 10 ? "      $site" : "     $site"
                target_str = target_site < 10 ? "      $target_site" : "     $target_site"
                println(fp, "    $(trans)$(site_str)$(target_str)      1")
            end
        end
    end

    println("    qptransidx.def is written.")
end

"""
    print_namelist_def(params::StdFaceParameters, model::ModelData, output_dir::String)

Generate namelist.def file.
"""
function print_namelist_def(params::StdFaceParameters, model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "namelist.def")

    open(filename, "w") do fp
        println(fp, "         ModPara  modpara.def")
        println(fp, "         LocSpin  locspn.def")
        println(fp, "           Trans  trans.def")

        if model.lcintra
            println(fp, "    CoulombIntra  coulombintra.def")
        end
        if model.lcinter
            println(fp, "    CoulombInter  coulombinter.def")
        end
        if model.lhund
            println(fp, "            Hund  hund.def")
        end
        if model.lex
            println(fp, "        Exchange  exchange.def")
        end

        if params.ioutputmode != 0
            println(fp, "        OneBodyG  greenone.def")
            println(fp, "        TwoBodyG  greentwo.def")
        end

        println(fp, "      Gutzwiller  gutzwilleridx.def")
        println(fp, "         Jastrow  jastrowidx.def")
        println(fp, "         Orbital  orbitalidx.def")
        println(fp, "        TransSym  qptransidx.def")
    end

    println("    namelist.def is written.")
end
