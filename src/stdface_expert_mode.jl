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
        model.ex[i] = -J/2.0  # Factor of -1/2 for Heisenberg model convention
    end

    model.lex = true

    # Add empty coulomb inter and hund interactions for completeness
    # (these are needed in the namelist.def)
    model.ncinter = 0
    model.cinterindx = Vector{Vector{Int}}()
    model.cinter = Vector{Float64}()
    model.lcinter = true  # Generate empty file

    model.nhund = 0
    model.hundindx = Vector{Vector{Int}}()
    model.hund = Vector{Float64}()
    model.lhund = true  # Generate empty file
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
        println(fp, "================================")
        println(fp, "NlocalSpin $(lpad(nlocspin, 5))")
        println(fp, "================================")
        println(fp, "========i_1LocSpn_0IteElc ======")
        println(fp, "================================")

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
        println(fp, "========================")
        println(fp, "NTransfer $(lpad(model.ntrans, 7))")
        println(fp, "========================")
        println(fp, "========i_j_s_tijs======")
        println(fp, "========================")

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
        println(fp, "================== Hund ================")
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
        println(fp, "Nsite          $(params.L)")
        println(fp, "Ncond          $(params.L)")  # For half-filling
        println(fp, "2Sz            $(params.TwoSz)")
        println(fp, "NSROptItrStep  $(params.NSROptItrStep)")
        println(fp, "NSROptItrSmp   $(params.NSROptItrSmp)")
        println(fp, "DSROptRedCut   $(params.DSROptRedCut)")
        println(fp, "DSROptStaDel   $(params.DSROptStaDel)")
        println(fp, "DSROptStepDt   $(params.DSROptStepDt)")
        println(fp, "NVMCWarmUp     $(params.NVMCWarmUp)")
        println(fp, "NVMCInterval   $(params.NVMCInterval)")
        println(fp, "NVMCSample     $(params.NVMCSample)")
        println(fp, "NExUpdatePath  0")
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
        println(fp, "NGutzwillerIdx    $(nsite)")
        println(fp, "ComplexType    0")
        println(fp, "=============================================")
        println(fp, "=============================================")

        for i in 0:(nsite-1)
            println(fp, "    $(i)")
        end
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
    npairs = nsite * (nsite - 1) ÷ 2

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NJastrowIdx    $(npairs)")
        println(fp, "ComplexType    0")
        println(fp, "=============================================")
        println(fp, "=============================================")

        for i in 0:(nsite-1)
            for j in (i+1):(nsite-1)
                println(fp, "    $(i)    $(j)")
            end
        end
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
    norbital = nsite * nsite

    open(filename, "w") do fp
        println(fp, "=============================================")
        println(fp, "NOrbitalIdx    $(norbital)")
        println(fp, "ComplexType    0")
        println(fp, "=============================================")
        println(fp, "=============================================")

        for i in 0:(nsite-1)
            for j in 0:(nsite-1)
                println(fp, "    $(i)    $(j)")
            end
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

    nsite = model.lattice.nsite
    ngreen = 2 * nsite * nsite  # Up and down spins

    open(filename, "w") do fp
        println(fp, "===============================")
        println(fp, "NCisAjs $(ngreen)")
        println(fp, "===============================")
        println(fp, "======== Green functions ======")
        println(fp, "===============================")

        for spin in 0:1
            for i in 0:(nsite-1)
                for j in 0:(nsite-1)
                    println(fp, "$(lpad(i, 5)) $(lpad(spin, 5)) $(lpad(j, 5)) $(lpad(spin, 5))")
                end
            end
        end
    end

    println("    greenone.def is written.")
end

"""
    print_greentwo_def(model::ModelData, output_dir::String)

Generate greentwo.def file.
"""
function print_greentwo_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "greentwo.def")

    # For now, generate empty file - two-body Green's functions are complex
    open(filename, "w") do fp
        println(fp, "===============================")
        println(fp, "NCisAjsCktAltDC    0")
        println(fp, "===============================")
        println(fp, "====== Two-body Green ========")
        println(fp, "===============================")
    end

    println("    greentwo.def is written.")
end

"""
    print_qptrans_def(model::ModelData, output_dir::String)

Generate qptransidx.def file.
"""
function print_qptrans_def(model::ModelData, output_dir::String)
    filename = joinpath(output_dir, "qptransidx.def")

    # For now, generate empty file - quantum projection is complex
    open(filename, "w") do fp
        println(fp, "===============================")
        println(fp, "NQPTrans    0")
        println(fp, "===============================")
        println(fp, "======== Trans Sym ============")
        println(fp, "===============================")
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
