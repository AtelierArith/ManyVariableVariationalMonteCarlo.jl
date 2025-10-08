"""
Visualization and Analysis Tools for ManyVariableVariationalMonteCarlo.jl

Implements comprehensive visualization and analysis functionality including:
- Energy convergence plots
- Parameter evolution visualization
- Correlation function plots
- Wavefunction analysis
- Statistical analysis and reporting
- Interactive plotting capabilities

Ported from visualization concepts in the C reference implementation.
"""

using Plots
using StatsPlots
using PlotlyJS
using Statistics
using LinearAlgebra
using Dates

"""
    PlotConfig

Configuration for plotting and visualization.
"""
mutable struct PlotConfig
    # Plot settings
    plot_engine::String
    plot_size::Tuple{Int,Int}
    plot_dpi::Int

    # Color scheme
    color_scheme::String
    line_width::Float64
    marker_size::Float64

    # Output settings
    save_plots::Bool
    output_dir::String
    output_format::String

    # Interactive settings
    interactive_mode::Bool
    show_plots::Bool

    function PlotConfig(;
        plot_engine::String = "gr",
        plot_size::Tuple{Int,Int} = (800, 600),
        plot_dpi::Int = 300,
        color_scheme::String = "default",
        line_width::Float64 = 2.0,
        marker_size::Float64 = 4.0,
        save_plots::Bool = true,
        output_dir::String = "plots",
        output_format::String = "png",
        interactive_mode::Bool = false,
        show_plots::Bool = true,
    )
        new(
            plot_engine,
            plot_size,
            plot_dpi,
            color_scheme,
            line_width,
            marker_size,
            save_plots,
            output_dir,
            output_format,
            interactive_mode,
            show_plots,
        )
    end
end

"""
    VisualizationManager

Manages visualization and analysis operations.
"""
mutable struct VisualizationManager
    # Configuration
    config::PlotConfig

    # Data storage
    energy_history::Vector{Float64}
    parameter_history::Vector{Vector{Float64}}
    gradient_history::Vector{Vector{Float64}}
    observable_history::Vector{Vector{Float64}}

    # Plot objects
    current_plots::Dict{String,Any}

    # Statistics
    total_plots::Int
    plot_generation_time::Float64

    function VisualizationManager(; config::PlotConfig = PlotConfig())
        new(
            config,
            Float64[],
            Vector{Float64}[],
            Vector{Float64}[],
            Vector{Float64}[],
            Dict{String,Any}(),
            0,
            0.0,
        )
    end
end

"""
    plot_energy_convergence(manager::VisualizationManager,
                           energy_data::Vector{Float64};
                           title::String = "Energy Convergence")

Plot energy convergence over iterations.

C実装参考: vmcmain.c 1行目から803行目まで
"""
function plot_energy_convergence(
    manager::VisualizationManager,
    energy_data::Vector{Float64};
    title::String = "Energy Convergence",
)
    # Create plot
    p = plot(
        energy_data,
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = title,
        xlabel = "Iteration",
        ylabel = "Energy",
        legend = :topright,
        size = manager.config.plot_size,
    )

    # Add moving average
    if length(energy_data) > 10
        window_size = min(50, length(energy_data) ÷ 10)
        moving_avg =
            [mean(energy_data[max(1, i-window_size+1):i]) for i = 1:length(energy_data)]
        plot!(
            p,
            moving_avg,
            linewidth = manager.config.line_width + 1,
            linestyle = :dash,
            label = "Moving Average ($window_size)",
        )
    end

    # Add convergence line
    if length(energy_data) > 1
        final_energy = energy_data[end]
        hline!(p, [final_energy], linewidth = 1, linestyle = :dot, label = "Final Energy")
    end

    # Save plot
    if manager.config.save_plots
        save_plot(manager, p, "energy_convergence")
    end

    # Show plot
    if manager.config.show_plots
        display(p)
    end

    # Store plot
    manager.current_plots["energy_convergence"] = p
    manager.total_plots += 1

    return p
end

"""
    plot_parameter_evolution(manager::VisualizationManager,
                           parameter_data::Vector{Vector{Float64}};
                           title::String = "Parameter Evolution")

Plot parameter evolution over iterations.
"""
function plot_parameter_evolution(
    manager::VisualizationManager,
    parameter_data::Vector{Vector{Float64}};
    title::String = "Parameter Evolution",
)
    if isempty(parameter_data)
        return nothing
    end

    # Create plot
    p = plot(
        title = title,
        xlabel = "Iteration",
        ylabel = "Parameter Value",
        legend = :topright,
        size = manager.config.plot_size,
    )

    # Plot each parameter
    n_params = length(parameter_data[1])
    for i = 1:min(n_params, 10)  # Limit to first 10 parameters
        param_values = [data[i] for data in parameter_data]
        plot!(
            p,
            param_values,
            linewidth = manager.config.line_width,
            markersize = manager.config.marker_size,
            label = "Parameter $i",
        )
    end

    # Add parameter statistics
    if length(parameter_data) > 1
        param_means = [mean([data[i] for data in parameter_data]) for i = 1:n_params]
        param_stds = [std([data[i] for data in parameter_data]) for i = 1:n_params]

        # Plot mean ± std
        plot!(
            p,
            param_means,
            linewidth = manager.config.line_width + 1,
            linestyle = :dash,
            label = "Mean",
        )

        # Add error bars
        plot!(
            p,
            param_means,
            yerror = param_stds,
            linewidth = 0,
            markersize = 0,
            label = "±1σ",
        )
    end

    # Save plot
    if manager.config.save_plots
        save_plot(manager, p, "parameter_evolution")
    end

    # Show plot
    if manager.config.show_plots
        display(p)
    end

    # Store plot
    manager.current_plots["parameter_evolution"] = p
    manager.total_plots += 1

    return p
end

"""
    plot_correlation_functions(manager::VisualizationManager,
                              correlation_data::Vector{Vector{Float64}};
                              title::String = "Correlation Functions")

Plot correlation functions.
"""
function plot_correlation_functions(
    manager::VisualizationManager,
    correlation_data::Vector{Vector{Float64}};
    title::String = "Correlation Functions",
)
    if isempty(correlation_data)
        return nothing
    end

    # Create plot
    p = plot(
        title = title,
        xlabel = "Distance",
        ylabel = "Correlation",
        legend = :topright,
        size = manager.config.plot_size,
    )

    # Plot each correlation function
    for (i, corr_data) in enumerate(correlation_data)
        distances = collect(0:(length(corr_data)-1))
        plot!(
            p,
            distances,
            corr_data,
            linewidth = manager.config.line_width,
            markersize = manager.config.marker_size,
            label = "Correlation $i",
        )
    end

    # Add theoretical curves if available
    if length(correlation_data) > 0
        # Add exponential decay as reference
        distances = collect(0:(length(correlation_data[1])-1))
        exp_decay = exp.(-distances ./ 2.0)
        plot!(
            p,
            distances,
            exp_decay,
            linewidth = manager.config.line_width,
            linestyle = :dash,
            label = "Exp. Decay",
        )
    end

    # Save plot
    if manager.config.save_plots
        save_plot(manager, p, "correlation_functions")
    end

    # Show plot
    if manager.config.show_plots
        display(p)
    end

    # Store plot
    manager.current_plots["correlation_functions"] = p
    manager.total_plots += 1

    return p
end

"""
    plot_acceptance_rates(manager::VisualizationManager,
                         acceptance_data::Vector{Float64};
                         title::String = "Acceptance Rates")

Plot acceptance rates over iterations.
"""
function plot_acceptance_rates(
    manager::VisualizationManager,
    acceptance_data::Vector{Float64};
    title::String = "Acceptance Rates",
)
    # Create plot
    p = plot(
        acceptance_data,
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = title,
        xlabel = "Iteration",
        ylabel = "Acceptance Rate",
        legend = :topright,
        size = manager.config.plot_size,
    )

    # Add target acceptance rate
    target_rate = 0.5
    hline!(p, [target_rate], linewidth = 1, linestyle = :dash, label = "Target Rate")

    # Add moving average
    if length(acceptance_data) > 10
        window_size = min(50, length(acceptance_data) ÷ 10)
        moving_avg = [
            mean(acceptance_data[max(1, i-window_size+1):i]) for
            i = 1:length(acceptance_data)
        ]
        plot!(
            p,
            moving_avg,
            linewidth = manager.config.line_width + 1,
            linestyle = :dash,
            label = "Moving Average",
        )
    end

    # Save plot
    if manager.config.save_plots
        save_plot(manager, p, "acceptance_rates")
    end

    # Show plot
    if manager.config.show_plots
        display(p)
    end

    # Store plot
    manager.current_plots["acceptance_rates"] = p
    manager.total_plots += 1

    return p
end

"""
    plot_wavefunction_analysis(manager::VisualizationManager,
                              wavefunction_data::Vector{ComplexF64};
                              title::String = "Wavefunction Analysis")

Plot wavefunction analysis.
"""
function plot_wavefunction_analysis(
    manager::VisualizationManager,
    wavefunction_data::Vector{ComplexF64};
    title::String = "Wavefunction Analysis",
)
    # Create subplots
    p1 = plot(
        real.(wavefunction_data),
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = "Real Part",
        xlabel = "Sample",
        ylabel = "Real(ψ)",
        legend = false,
        size = (manager.config.plot_size[1] ÷ 2, manager.config.plot_size[2]),
    )

    p2 = plot(
        imag.(wavefunction_data),
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = "Imaginary Part",
        xlabel = "Sample",
        ylabel = "Imag(ψ)",
        legend = false,
        size = (manager.config.plot_size[1] ÷ 2, manager.config.plot_size[2]),
    )

    p3 = plot(
        abs.(wavefunction_data),
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = "Magnitude",
        xlabel = "Sample",
        ylabel = "|ψ|",
        legend = false,
        size = (manager.config.plot_size[1] ÷ 2, manager.config.plot_size[2]),
    )

    p4 = plot(
        angle.(wavefunction_data),
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = "Phase",
        xlabel = "Sample",
        ylabel = "arg(ψ)",
        legend = false,
        size = (manager.config.plot_size[1] ÷ 2, manager.config.plot_size[2]),
    )

    # Combine subplots
    p = plot(
        p1,
        p2,
        p3,
        p4,
        layout = (2, 2),
        size = manager.config.plot_size,
        title = title,
    )

    # Save plot
    if manager.config.save_plots
        save_plot(manager, p, "wavefunction_analysis")
    end

    # Show plot
    if manager.config.show_plots
        display(p)
    end

    # Store plot
    manager.current_plots["wavefunction_analysis"] = p
    manager.total_plots += 1

    return p
end

"""
    plot_statistical_analysis(manager::VisualizationManager,
                             data::Vector{Float64};
                             title::String = "Statistical Analysis")

Plot statistical analysis of data.
"""
function plot_statistical_analysis(
    manager::VisualizationManager,
    data::Vector{Float64};
    title::String = "Statistical Analysis",
)
    # Create subplots
    p1 = histogram(
        data,
        bins = 50,
        title = "Histogram",
        xlabel = "Value",
        ylabel = "Frequency",
        legend = false,
        size = (manager.config.plot_size[1] ÷ 2, manager.config.plot_size[2]),
    )

    p2 = plot(
        data,
        linewidth = manager.config.line_width,
        markersize = manager.config.marker_size,
        title = "Time Series",
        xlabel = "Sample",
        ylabel = "Value",
        legend = false,
        size = (manager.config.plot_size[1] ÷ 2, manager.config.plot_size[2]),
    )

    # Add statistics
    mean_val = mean(data)
    std_val = std(data)

    # Add mean and std lines
    hline!(p1, [mean_val], linewidth = 2, linestyle = :dash, label = "Mean")
    hline!(
        p1,
        [mean_val + std_val, mean_val - std_val],
        linewidth = 1,
        linestyle = :dot,
        label = "±1σ",
    )

    hline!(p2, [mean_val], linewidth = 2, linestyle = :dash, label = "Mean")
    hline!(
        p2,
        [mean_val + std_val, mean_val - std_val],
        linewidth = 1,
        linestyle = :dot,
        label = "±1σ",
    )

    # Combine subplots
    p = plot(p1, p2, layout = (1, 2), size = manager.config.plot_size, title = title)

    # Save plot
    if manager.config.save_plots
        save_plot(manager, p, "statistical_analysis")
    end

    # Show plot
    if manager.config.show_plots
        display(p)
    end

    # Store plot
    manager.current_plots["statistical_analysis"] = p
    manager.total_plots += 1

    return p
end

"""
    save_plot(manager::VisualizationManager, plot_obj, filename::String)

Save plot to file.
"""
function save_plot(manager::VisualizationManager, plot_obj, filename::String)
    # Create output directory if it doesn't exist
    if !isdir(manager.config.output_dir)
        mkpath(manager.config.output_dir)
    end

    # Generate filename
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    full_filename = joinpath(
        manager.config.output_dir,
        "$(filename)_$(timestamp).$(manager.config.output_format)",
    )

    # Save plot
    savefig(plot_obj, full_filename)

    println("Plot saved to: $full_filename")
end

"""
    generate_report(manager::VisualizationManager,
                   results::Dict{String, Any};
                   output_file::String = "analysis_report.html")

Generate comprehensive analysis report.
"""
function generate_report(
    manager::VisualizationManager,
    results::Dict{String,Any};
    output_file::String = "analysis_report.html",
)
    # Create HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>VMC Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin: 20px 0; }
            .plot { text-align: center; margin: 20px 0; }
            .stats { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>VMC Analysis Report</h1>
        <p>Generated on: $(now())</p>

        <div class="section">
            <h2>Summary Statistics</h2>
            <div class="stats">
    """

    # Add summary statistics
    if haskey(results, "energy_mean")
        html_content *= "<p><strong>Energy Mean:</strong> $(results["energy_mean"])</p>"
    end
    if haskey(results, "energy_std")
        html_content *= "<p><strong>Energy Std:</strong> $(results["energy_std"])</p>"
    end
    if haskey(results, "acceptance_rate")
        html_content *= "<p><strong>Acceptance Rate:</strong> $(results["acceptance_rate"])</p>"
    end
    if haskey(results, "n_samples")
        html_content *= "<p><strong>Number of Samples:</strong> $(results["n_samples"])</p>"
    end

    html_content *= """
            </div>
        </div>

        <div class="section">
            <h2>Plots</h2>
    """

    # Add plot references
    for (plot_name, plot_obj) in manager.current_plots
        plot_filename = "$(plot_name)_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).$(manager.config.output_format)"
        html_content *= """
            <div class="plot">
                <h3>$(replace(plot_name, "_" => " "))</h3>
                <img src="$(plot_filename)" alt="$(plot_name)" style="max-width: 100%;">
            </div>
        """
    end

    html_content *= """
        </div>

        <div class="section">
            <h2>Detailed Results</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
    """

    # Add detailed results table
    for (key, value) in results
        if !startswith(key, "_")
            html_content *= "<tr><td>$key</td><td>$value</td></tr>"
        end
    end

    html_content *= """
            </table>
        </div>
    </body>
    </html>
    """

    # Write HTML file
    open(output_file, "w") do file
        write(file, html_content)
    end

    println("Report generated: $output_file")
end

"""
    benchmark_visualization(n_samples::Int = 1000, n_params::Int = 50)

Benchmark visualization performance.
"""
function benchmark_visualization(n_samples::Int = 1000, n_params::Int = 50)
    println("Benchmarking visualization (n_samples=$n_samples, n_params=$n_params)...")

    # Create visualization manager
    config = PlotConfig(save_plots = false, show_plots = false)
    manager = VisualizationManager(config = config)

    # Generate test data
    energy_data = cumsum(randn(n_samples)) .+ 10.0
    parameter_data = [randn(n_params) for _ = 1:n_samples]
    correlation_data = [exp.(-collect(0:9) ./ 2.0) for _ = 1:3]
    acceptance_data = 0.3 .+ 0.2 .* rand(n_samples)
    wavefunction_data = rand(ComplexF64, n_samples)

    # Benchmark plotting functions
    @time begin
        plot_energy_convergence(manager, energy_data)
    end
    println("  Energy convergence plot")

    @time begin
        plot_parameter_evolution(manager, parameter_data)
    end
    println("  Parameter evolution plot")

    @time begin
        plot_correlation_functions(manager, correlation_data)
    end
    println("  Correlation functions plot")

    @time begin
        plot_acceptance_rates(manager, acceptance_data)
    end
    println("  Acceptance rates plot")

    @time begin
        plot_wavefunction_analysis(manager, wavefunction_data)
    end
    println("  Wavefunction analysis plot")

    @time begin
        plot_statistical_analysis(manager, energy_data)
    end
    println("  Statistical analysis plot")

    # Generate report
    results = Dict{String,Any}(
        "energy_mean" => mean(energy_data),
        "energy_std" => std(energy_data),
        "acceptance_rate" => mean(acceptance_data),
        "n_samples" => n_samples,
    )

    @time begin
        generate_report(manager, results)
    end
    println("  Report generation")

    println("Visualization benchmark completed.")
    println("  Total plots generated: $(manager.total_plots)")
end
