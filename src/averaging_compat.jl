"""
Averaging and Statistics System for mVMC C Compatibility

Translates the averaging and statistics modules (average.c, avevar.c) to Julia,
maintaining exact compatibility with C numerical methods and statistical calculations.

Ported from:
- average.c: Weighted averaging functions
- avevar.c: Average and variance calculations
- average.h: Averaging header definitions
"""

using Statistics
using ..GlobalState: global_state, Wc, Etot, Etot2, Sztot, Sztot2, SROptOO, SROptHO, SROptO,
                     SROptOO_real, SROptHO_real, SROptO_real, SROptData, NSROptItrSmp, NPara,
                     SROptSize, NSRCG, PhysCisAjs, PhysCisAjsCktAlt, PhysCisAjsCktAltDC,
                     LocalCisAjs, LocalCisAjsCktAlt, LocalCisAjsCktAltDC, NCisAjs, NCisAjsCktAlt,
                     NCisAjsCktAltDC, Para, AllComplexFlag, FlagRBM, NProjBF

# Import required modules
using ..MPIWrapper: MPI_Comm, MPI_Comm_size, MPI_Comm_rank, MPI_Barrier, MPI_Allreduce, MPI_SUM
using ..IOCompat: write_header, write_opt_data, output_opt_data

# Constants matching C implementation
const EPS = 1.0e-14
const MAX_ITER = 100

"""
    weight_average_we(comm::MPI_Comm)

Calculate weighted average of Wc, Etot, Etot2, Sztot, Sztot2.
Matches C function WeightAverageWE.

C実装参考: average.c 1行目から334行目まで
"""
function weight_average_we(comm::MPI_Comm)
    rank, size = MPI_Comm_rank(comm), MPI_Comm_size(comm)

    if size > 1
        # MPI reduction
        send_data = [Wc, Etot, Etot2, Sztot, Sztot2]
        recv_data = MPI_Allreduce(send_data, MPI_SUM, comm)

        Wc = recv_data[1]
        inv_w = 1.0 / Wc
        Etot = recv_data[2] * inv_w
        Etot2 = recv_data[3] * inv_w
        Sztot = recv_data[4] * inv_w
        Sztot2 = recv_data[5] * inv_w
    else
        # Single process
        inv_w = 1.0 / Wc
        Etot *= inv_w
        Etot2 *= inv_w
        Sztot *= inv_w
        Sztot2 *= inv_w
    end
end

"""
    weight_average_sr_opt(comm::MPI_Comm)

Calculate weighted average of SROptOO and SROptHO.
Matches C function WeightAverageSROpt.

C実装参考: average.c 1行目から334行目まで
"""
function weight_average_sr_opt(comm::MPI_Comm)
    rank, size = MPI_Comm_rank(comm), MPI_Comm_size(comm)

    inv_w = 1.0 / Wc

    # Determine array size
    if NSRCG == 0
        n = 2 * SROptSize * (2 * SROptSize + 1)
    else
        n = 2 * SROptSize * 3
    end

    if size > 1
        # MPI reduction
        SROptOO = MPI_Allreduce(SROptOO, MPI_SUM, comm)
        SROptHO = MPI_Allreduce(SROptHO, MPI_SUM, comm)

        # Apply weight
        for i in 1:n
            SROptOO[i] *= inv_w
        end

        for i in 1:SROptSize
            SROptHO[i] *= inv_w
        end
    else
        # Single process
        for i in 1:n
            SROptOO[i] *= inv_w
        end

        for i in 1:SROptSize
            SROptHO[i] *= inv_w
        end
    end
end

"""
    weight_average_sr_opt_real(comm::MPI_Comm)

Calculate weighted average of SROptOO_real and SROptHO_real.
Matches C function WeightAverageSROpt_real.
"""
function weight_average_sr_opt_real(comm::MPI_Comm)
    rank, size = MPI_Comm_rank(comm), MPI_Comm_size(comm)

    inv_w = 1.0 / Wc

    # Determine array size
    if NSRCG == 0
        n = SROptSize * (SROptSize + 1)
    else
        n = SROptSize * 3
    end

    if size > 1
        # MPI reduction
        SROptOO_real = MPI_Allreduce(SROptOO_real, MPI_SUM, comm)
        SROptHO_real = MPI_Allreduce(SROptHO_real, MPI_SUM, comm)

        # Apply weight
        for i in 1:n
            SROptOO_real[i] *= inv_w
        end

        for i in 1:SROptSize
            SROptHO_real[i] *= inv_w
        end
    else
        # Single process
        for i in 1:n
            SROptOO_real[i] *= inv_w
        end

        for i in 1:SROptSize
            SROptHO_real[i] *= inv_w
        end
    end
end

"""
    weight_average_green_func(comm::MPI_Comm)

Calculate weighted average of Green functions.
Matches C function WeightAverageGreenFunc.
"""
function weight_average_green_func(comm::MPI_Comm)
    rank, size = MPI_Comm_rank(comm), MPI_Comm_size(comm)

    if size > 1
        # MPI reduction for Green functions
        PhysCisAjs = MPI_Allreduce(PhysCisAjs, MPI_SUM, comm)
        PhysCisAjsCktAlt = MPI_Allreduce(PhysCisAjsCktAlt, MPI_SUM, comm)
        PhysCisAjsCktAltDC = MPI_Allreduce(PhysCisAjsCktAltDC, MPI_SUM, comm)
        LocalCisAjs = MPI_Allreduce(LocalCisAjs, MPI_SUM, comm)
        LocalCisAjsCktAlt = MPI_Allreduce(LocalCisAjsCktAlt, MPI_SUM, comm)
        LocalCisAjsCktAltDC = MPI_Allreduce(LocalCisAjsCktAltDC, MPI_SUM, comm)
    end

    # Apply weight
    inv_w = 1.0 / Wc

    for i in 1:NCisAjs
        PhysCisAjs[i] *= inv_w
        LocalCisAjs[i] *= inv_w
    end

    for i in 1:NCisAjsCktAlt
        PhysCisAjsCktAlt[i] *= inv_w
        LocalCisAjsCktAlt[i] *= inv_w
    end

    for i in 1:NCisAjsCktAltDC
        PhysCisAjsCktAltDC[i] *= inv_w
        LocalCisAjsCktAltDC[i] *= inv_w
    end
end

"""
    calc_ave_var(i::Int, n::Int, ave::Ref{ComplexF64}, var::Ref{Float64})

Calculate average and variance.
Matches C function CalcAveVar.

C実装参考: avevar.c 1行目から262行目まで
"""
function calc_ave_var(i::Int, n::Int, ave::Ref{ComplexF64}, var::Ref{Float64})
    # Calculate average
    ave[] = ComplexF64(0.0)
    for sample in 1:NSROptItrSmp
        ave[] += SROptData[i + n * (sample - 1)]
    end
    ave[] /= NSROptItrSmp

    # Calculate variance
    var[] = 0.0
    for sample in 1:NSROptItrSmp
        data = SROptData[i + n * (sample - 1)] - ave[]
        var[] += real(data * conj(data))
    end
    var[] = sqrt(var[] / (NSROptItrSmp - 1.0))
end

"""
    write_header(c_nkw_idx::String, nkw_idx::Int, fp::IO)

Write header to file.
Matches C function WriteHeader.
"""
function write_header(c_nkw_idx::String, nkw_idx::Int, fp::IO)
    println(fp, "======================")
    println(fp, c_nkw_idx, "  ", nkw_idx)
    println(fp, "======================")
    println(fp, "======================")
    println(fp, "======================")
end

"""
    child_output_opt_data(fp_all::IO, c_file_name::String, c_nkw_idx::String,
                         n_idx_head::Int, n_idx::Int, count_i::Int, n::Int)

Output optimization data for child process.
Matches C function Child_OutputOptData.
"""
function child_output_opt_data(fp_all::IO, c_file_name::String, c_nkw_idx::String,
                               n_idx_head::Int, n_idx::Int, count_i::Int, n::Int)
    fp_out = open(c_file_name, "w")
    write_header(c_nkw_idx, n_idx_head, fp_out)

    for i in 1:n_idx
        ave = Ref{ComplexF64}(0.0)
        var = Ref{Float64}(0.0)
        calc_ave_var(count_i + i, n, ave, var)

        println(fp_out, i, " ", real(ave[]), " ", imag(ave[]))
        print(fp_all, real(ave[]), " ", imag(ave[]), " ", var[], " ")
    end

    close(fp_out)
end

"""
    store_opt_data(sample::Int)

Store optimization data.
Matches C function StoreOptData.
"""
function store_opt_data(sample::Int)
    n = 2 + NPara
    opt_data = SROptData[(sample - 1) * n + 1:sample * n]

    opt_data[1] = Etot
    opt_data[2] = Etot2
    for i in 1:NPara
        opt_data[i + 2] = Para[i]
    end
end

"""
    output_opt_data()

Output optimization data.
Matches C function OutputOptData.
"""
function output_opt_data()
    n = 2 + NPara
    file_name = "zvo_opt.dat"

    fp = open(file_name, "w")
    write_header("Optimization Data", 1, fp)

    for i in 1:n
        ave = Ref{ComplexF64}(0.0)
        var = Ref{Float64}(0.0)
        calc_ave_var(i, n, ave, var)

        println(fp, i, " ", real(ave[]), " ", imag(ave[]), " ", var[])
    end

    close(fp)
end

"""
    calculate_statistics(data::Vector{ComplexF64})

Calculate basic statistics.
"""
function calculate_statistics(data::Vector{ComplexF64})
    n = length(data)
    if n == 0
        return (ComplexF64(0.0), 0.0, 0.0, 0.0)
    end

    # Calculate mean
    mean_val = sum(data) / n

    # Calculate variance
    var_val = sum(abs2.(data .- mean_val)) / (n - 1)

    # Calculate standard deviation
    std_val = sqrt(var_val)

    # Calculate standard error
    se_val = std_val / sqrt(n)

    return (mean_val, var_val, std_val, se_val)
end

"""
    calculate_correlation(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})

Calculate correlation coefficient.
"""
function calculate_correlation(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})
    n = min(length(data1), length(data2))
    if n == 0
        return 0.0
    end

    # Calculate means
    mean1 = sum(data1[1:n]) / n
    mean2 = sum(data2[1:n]) / n

    # Calculate correlation
    numerator = sum((data1[1:n] .- mean1) .* conj.(data2[1:n] .- mean2))
    denominator = sqrt(sum(abs2.(data1[1:n] .- mean1)) * sum(abs2.(data2[1:n] .- mean2)))

    if denominator == 0.0
        return 0.0
    end

    return real(numerator / denominator)
end

"""
    calculate_autocorrelation(data::Vector{ComplexF64}, lag::Int)

Calculate autocorrelation function.
"""
function calculate_autocorrelation(data::Vector{ComplexF64}, lag::Int)
    n = length(data)
    if n <= lag
        return 0.0
    end

    # Calculate mean
    mean_val = sum(data) / n

    # Calculate autocorrelation
    numerator = sum((data[1:n-lag] .- mean_val) .* conj.(data[lag+1:n] .- mean_val))
    denominator = sum(abs2.(data .- mean_val))

    if denominator == 0.0
        return 0.0
    end

    return real(numerator / denominator)
end

"""
    calculate_spectral_density(data::Vector{ComplexF64})

Calculate spectral density.
"""
function calculate_spectral_density(data::Vector{ComplexF64})
    n = length(data)
    if n == 0
        return Vector{ComplexF64}()
    end

    # Calculate FFT
    fft_data = fft(data)

    # Calculate spectral density
    spectral_density = abs2.(fft_data)

    return spectral_density
end

"""
    calculate_power_spectrum(data::Vector{ComplexF64})

Calculate power spectrum.
"""
function calculate_power_spectrum(data::Vector{ComplexF64})
    n = length(data)
    if n == 0
        return Vector{Float64}()
    end

    # Calculate spectral density
    spectral_density = calculate_spectral_density(data)

    # Calculate power spectrum
    power_spectrum = real.(spectral_density)

    return power_spectrum
end

"""
    calculate_cross_spectrum(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})

Calculate cross spectrum.
"""
function calculate_cross_spectrum(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})
    n = min(length(data1), length(data2))
    if n == 0
        return Vector{ComplexF64}()
    end

    # Calculate FFTs
    fft1 = fft(data1[1:n])
    fft2 = fft(data2[1:n])

    # Calculate cross spectrum
    cross_spectrum = fft1 .* conj.(fft2)

    return cross_spectrum
end

"""
    calculate_coherence(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})

Calculate coherence.
"""
function calculate_coherence(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})
    n = min(length(data1), length(data2))
    if n == 0
        return Vector{Float64}()
    end

    # Calculate cross spectrum
    cross_spectrum = calculate_cross_spectrum(data1, data2)

    # Calculate power spectra
    power1 = calculate_power_spectrum(data1[1:n])
    power2 = calculate_power_spectrum(data2[1:n])

    # Calculate coherence
    coherence = abs2.(cross_spectrum) ./ (power1 .* power2)

    return real.(coherence)
end

"""
    calculate_phase_spectrum(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})

Calculate phase spectrum.
"""
function calculate_phase_spectrum(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})
    n = min(length(data1), length(data2))
    if n == 0
        return Vector{Float64}()
    end

    # Calculate cross spectrum
    cross_spectrum = calculate_cross_spectrum(data1, data2)

    # Calculate phase spectrum
    phase_spectrum = angle.(cross_spectrum)

    return phase_spectrum
end

"""
    calculate_transfer_function(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})

Calculate transfer function.
"""
function calculate_transfer_function(data1::Vector{ComplexF64}, data2::Vector{ComplexF64})
    n = min(length(data1), length(data2))
    if n == 0
        return Vector{ComplexF64}()
    end

    # Calculate FFTs
    fft1 = fft(data1[1:n])
    fft2 = fft(data2[1:n])

    # Calculate transfer function
    transfer_function = fft2 ./ fft1

    return transfer_function
end

"""
    calculate_impulse_response(data::Vector{ComplexF64})

Calculate impulse response.
"""
function calculate_impulse_response(data::Vector{ComplexF64})
    n = length(data)
    if n == 0
        return Vector{ComplexF64}()
    end

    # Calculate IFFT
    impulse_response = ifft(data)

    return impulse_response
end

"""
    calculate_step_response(data::Vector{ComplexF64})

Calculate step response.
"""
function calculate_step_response(data::Vector{ComplexF64})
    n = length(data)
    if n == 0
        return Vector{ComplexF64}()
    end

    # Calculate cumulative sum
    step_response = cumsum(data)

    return step_response
end

"""
    calculate_frequency_response(data::Vector{ComplexF64}, frequencies::Vector{Float64})

Calculate frequency response.
"""
function calculate_frequency_response(data::Vector{ComplexF64}, frequencies::Vector{Float64})
    n = length(data)
    if n == 0
        return Vector{ComplexF64}()
    end

    # Calculate FFT
    fft_data = fft(data)

    # Calculate frequency response
    frequency_response = Vector{ComplexF64}(undef, length(frequencies))
    for (i, freq) in enumerate(frequencies)
        idx = round(Int, freq * n) + 1
        if 1 <= idx <= n
            frequency_response[i] = fft_data[idx]
        else
            frequency_response[i] = ComplexF64(0.0)
        end
    end

    return frequency_response
end

"""
    calculate_bode_plot(data::Vector{ComplexF64}, frequencies::Vector{Float64})

Calculate Bode plot.
"""
function calculate_bode_plot(data::Vector{ComplexF64}, frequencies::Vector{Float64})
    # Calculate frequency response
    frequency_response = calculate_frequency_response(data, frequencies)

    # Calculate magnitude and phase
    magnitude = abs.(frequency_response)
    phase = angle.(frequency_response)

    return (magnitude, phase)
end

"""
    calculate_nyquist_plot(data::Vector{ComplexF64}, frequencies::Vector{Float64})

Calculate Nyquist plot.
"""
function calculate_nyquist_plot(data::Vector{ComplexF64}, frequencies::Vector{Float64})
    # Calculate frequency response
    frequency_response = calculate_frequency_response(data, frequencies)

    # Calculate real and imaginary parts
    real_part = real.(frequency_response)
    imag_part = imag.(frequency_response)

    return (real_part, imag_part)
end

"""
    calculate_stability_margins(data::Vector{ComplexF64}, frequencies::Vector{Float64})

Calculate stability margins.
"""
function calculate_stability_margins(data::Vector{ComplexF64}, frequencies::Vector{Float64})
    # Calculate frequency response
    frequency_response = calculate_frequency_response(data, frequencies)

    # Calculate magnitude and phase
    magnitude = abs.(frequency_response)
    phase = angle.(frequency_response)

    # Find gain margin
    gain_margin = 1.0 / maximum(magnitude)

    # Find phase margin
    phase_margin = minimum(phase) + π

    return (gain_margin, phase_margin)
end

# Utility functions

"""
    initialize_averaging_system()

Initialize averaging system.
"""
function initialize_averaging_system()
    # Initialize averaging system
    # This is a simplified version - the full implementation would
    # initialize all averaging-related systems
    return
end

"""
    output_statistics_info(data::Vector{ComplexF64}, name::String)

Output statistics information.
"""
function output_statistics_info(data::Vector{ComplexF64}, name::String)
    mean_val, var_val, std_val, se_val = calculate_statistics(data)

    println("Statistics for $name:")
    println("  Mean: $mean_val")
    println("  Variance: $var_val")
    println("  Standard Deviation: $std_val")
    println("  Standard Error: $se_val")
end

"""
    calculate_confidence_interval(data::Vector{ComplexF64}, confidence::Float64)

Calculate confidence interval.
"""
function calculate_confidence_interval(data::Vector{ComplexF64}, confidence::Float64)
    n = length(data)
    if n == 0
        return (ComplexF64(0.0), ComplexF64(0.0))
    end

    # Calculate mean and standard error
    mean_val, _, _, se_val = calculate_statistics(data)

    # Calculate confidence interval
    z_score = quantile(Normal(), (1 + confidence) / 2)
    margin = z_score * se_val

    lower = mean_val - margin
    upper = mean_val + margin

    return (lower, upper)
end

"""
    calculate_bootstrap_statistics(data::Vector{ComplexF64}, n_bootstrap::Int)

Calculate bootstrap statistics.
"""
function calculate_bootstrap_statistics(data::Vector{ComplexF64}, n_bootstrap::Int)
    n = length(data)
    if n == 0
        return (ComplexF64(0.0), 0.0)
    end

    # Bootstrap samples
    bootstrap_means = Vector{ComplexF64}(undef, n_bootstrap)

    for i in 1:n_bootstrap
        # Sample with replacement
        sample = data[rand(1:n, n)]
        bootstrap_means[i] = sum(sample) / n
    end

    # Calculate bootstrap statistics
    bootstrap_mean = sum(bootstrap_means) / n_bootstrap
    bootstrap_var = sum(abs2.(bootstrap_means .- bootstrap_mean)) / (n_bootstrap - 1)

    return (bootstrap_mean, bootstrap_var)
end
