# Tests for advanced I/O system
#
# Tests all I/O functionality including:
# - Configuration I/O
# - Results I/O
# - Data export
# - Format validation
# - Performance benchmarks

@testitem "I/O Mode and Data Format Enums" begin
    using ManyVariableVariationalMonteCarlo

    # Test I/O mode enum
    @test READ_ONLY isa IOMode
    @test WRITE_ONLY isa IOMode
    @test READ_WRITE isa IOMode
    @test APPEND isa IOMode

    # Test data format enum
    @test HDF5_FORMAT isa DataFormat
    @test JSON_FORMAT isa DataFormat
    @test BINARY_FORMAT isa DataFormat
    @test TEXT_FORMAT isa DataFormat
end

@testitem "Configuration IO Basic Functionality" begin
    using ManyVariableVariationalMonteCarlo

    # Test configuration IO creation
    io = ConfigurationIO()
    @test io.config_file == "config.json"
    @test io.output_dir == "output"
    @test JSON_FORMAT in io.supported_formats
    @test HDF5_FORMAT in io.supported_formats
    @test io.validate_on_read == true
    @test io.validate_on_write == true
    @test io.strict_mode == false
    @test io.error_on_missing == true

    # Test custom configuration IO
    io_custom = ConfigurationIO(config_file = "custom.json", output_dir = "custom_output")
    @test io_custom.config_file == "custom.json"
    @test io_custom.output_dir == "custom_output"
end

@testitem "Configuration IO Save and Load" begin
    using ManyVariableVariationalMonteCarlo

    # Test configuration save and load
    io = ConfigurationIO(config_file = "test_config.json")

    # Create test configuration
    config = Dict{String,Any}(
        "n_sites" => 10,
        "n_electrons" => 5,
        "n_samples" => 1000,
        "temperature" => 1.0,
        "parameters" => [1.0, 2.0, 3.0],
    )

    # Save configuration
    save_configuration(io, config, JSON_FORMAT)
    @test isfile("test_config.json")

    # Load configuration
    loaded_config = load_configuration(io, JSON_FORMAT)
    @test loaded_config["n_sites"] == 10
    @test loaded_config["n_electrons"] == 5
    @test loaded_config["n_samples"] == 1000
    @test loaded_config["temperature"] == 1.0
    @test loaded_config["parameters"] == [1.0, 2.0, 3.0]

    # Clean up
    rm("test_config.json", force = true)
end

@testitem "Configuration Validation" begin
    using ManyVariableVariationalMonteCarlo

    # Test valid configuration
    valid_config =
        Dict{String,Any}("n_sites" => 10, "n_electrons" => 5, "n_samples" => 1000)

    # Should not throw
    validate_configuration(valid_config)

    # Test invalid configuration - missing required key
    invalid_config = Dict{String,Any}(
        "n_sites" => 10,
        "n_electrons" => 5,
        # Missing "n_samples"
    )

    @test_throws ArgumentError validate_configuration(invalid_config)

    # Test invalid configuration - invalid values
    invalid_config2 = Dict{String,Any}(
        "n_sites" => -1,  # Negative
        "n_electrons" => 5,
        "n_samples" => 1000,
    )

    @test_throws AssertionError validate_configuration(invalid_config2)

    # Test invalid configuration - n_electrons > n_sites
    invalid_config3 = Dict{String,Any}(
        "n_sites" => 5,
        "n_electrons" => 10,  # More electrons than sites
        "n_samples" => 1000,
    )

    @test_throws AssertionError validate_configuration(invalid_config3)
end

@testitem "Results IO Basic Functionality" begin
    using ManyVariableVariationalMonteCarlo

    # Test results IO creation
    io = ResultsIO()
    @test io.results_file == "results.h5"
    @test io.backup_dir == "backup"
    @test io.primary_format == HDF5_FORMAT
    @test JSON_FORMAT in io.backup_formats
    @test BINARY_FORMAT in io.backup_formats
    @test io.use_compression == true
    @test io.compression_level == 6
    @test io.include_metadata == true
    @test io.include_timestamps == true
    @test io.buffer_size == 1024 * 1024
    @test io.async_write == false

    # Test custom results IO
    io_custom = ResultsIO(results_file = "custom.h5", primary_format = JSON_FORMAT)
    @test io_custom.results_file == "custom.h5"
    @test io_custom.primary_format == JSON_FORMAT
end

@testitem "Results IO Save and Load" begin
    using ManyVariableVariationalMonteCarlo

    # Test results save and load
    io = ResultsIO(results_file = "test_results.h5")

    # Create test results
    results = Dict{String,Any}(
        "energy_mean" => 1.5,
        "energy_std" => 0.1,
        "energy_samples" => [1.4, 1.5, 1.6],
        "gradient_samples" => [1.0 2.0; 3.0 4.0],
        "metadata" => Dict("n_sites" => 10, "n_electrons" => 5),
    )

    # Save results
    save_results(io, results, HDF5_FORMAT)
    @test isfile("test_results.h5")

    # Load results
    loaded_results = load_results(io, HDF5_FORMAT)
    @test loaded_results["energy_mean"] == 1.5
    @test loaded_results["energy_std"] == 0.1
    @test loaded_results["energy_samples"] == [1.4, 1.5, 1.6]
    @test loaded_results["gradient_samples"] == [1.0 2.0; 3.0 4.0]
    @test loaded_results["metadata"]["n_sites"] == 10
    @test loaded_results["metadata"]["n_electrons"] == 5

    # Clean up
    rm("test_results.h5", force = true)
end

@testitem "Data Exporter Basic Functionality" begin
    using ManyVariableVariationalMonteCarlo

    # Test data exporter creation
    exporter = DataExporter()
    @test exporter.output_format == HDF5_FORMAT
    @test exporter.include_metadata == true
    @test exporter.include_timestamps == true
    @test exporter.data_compression == true
    @test exporter.data_validation == true
    @test exporter.batch_size == 1000
    @test exporter.async_export == false

    # Test custom data exporter
    exporter_custom = DataExporter(output_format = JSON_FORMAT, batch_size = 500)
    @test exporter_custom.output_format == JSON_FORMAT
    @test exporter_custom.batch_size == 500
end

@testitem "Data Export" begin
    using ManyVariableVariationalMonteCarlo
    using JSON

    # Test data export
    exporter = DataExporter(output_format = JSON_FORMAT)

    # Create test data
    data = Dict{String,Any}(
        "energy_data" => [1.0, 2.0, 3.0],
        "parameter_data" => [0.1, 0.2, 0.3],
        "metadata" => Dict("n_samples" => 3, "timestamp" => "2024-01-01"),
    )

    # Export data
    export_data(exporter, data, "test_export.json")
    @test isfile("test_export.json")

    # Verify export
    exported_data = JSON.parsefile("test_export.json")
    @test exported_data["energy_data"] == [1.0, 2.0, 3.0]
    @test exported_data["parameter_data"] == [0.1, 0.2, 0.3]
    @test exported_data["metadata"]["n_samples"] == 3

    # Clean up
    rm("test_export.json", force = true)
end

@testitem "I/O Performance" begin
    using ManyVariableVariationalMonteCarlo

    # Test I/O performance
    io = ConfigurationIO()

    # Create large configuration
    large_config = Dict{String,Any}(
        "n_sites" => 100,
        "n_electrons" => 50,
        "n_samples" => 10000,
        "large_array" => rand(1000),
        "nested_dict" => Dict("level1" => Dict("level2" => rand(100))),
    )

    # Benchmark save
    @time begin
        save_configuration(io, large_config, JSON_FORMAT)
    end

    # Benchmark load
    @time begin
        loaded_config = load_configuration(io, JSON_FORMAT)
    end

    # Verify data integrity
    @test loaded_config["n_sites"] == 100
    @test loaded_config["n_electrons"] == 50
    @test loaded_config["n_samples"] == 10000
    @test length(loaded_config["large_array"]) == 1000
    @test length(loaded_config["nested_dict"]["level1"]["level2"]) == 100

    # Clean up
    rm("config.json", force = true)
end

@testitem "I/O Error Handling" begin
    using ManyVariableVariationalMonteCarlo

    # Test error handling
    io = ConfigurationIO(error_on_missing = true)

    # Test missing file
    @test_throws ArgumentError load_configuration(io, JSON_FORMAT)

    # Test unsupported format
    io_limited = ConfigurationIO(supported_formats = [HDF5_FORMAT])
    @test_throws ArgumentError save_configuration(
        io_limited,
        Dict{String,Any}(),
        JSON_FORMAT,
    )

    # Test invalid configuration
    io_strict = ConfigurationIO(strict_mode = true)
    invalid_config = Dict{String,Any}("invalid" => "config")
    @test_throws ArgumentError save_configuration(io_strict, invalid_config, JSON_FORMAT)
end

@testitem "I/O Format Compatibility" begin
    using ManyVariableVariationalMonteCarlo

    # Test format compatibility
    test_data = Dict{String,Any}(
        "number" => 42,
        "string" => "test",
        "array" => [1, 2, 3],
        "matrix" => [1 2; 3 4],
        "nested" => Dict("key" => "value"),
    )

    # Test JSON format
    io_json = ConfigurationIO(
        config_file = "test.json",
        validate_on_write = false,
        validate_on_read = false,
    )
    save_configuration(io_json, test_data, JSON_FORMAT)
    loaded_json = load_configuration(io_json, JSON_FORMAT)
    @test loaded_json["number"] == 42
    @test loaded_json["string"] == "test"
    @test loaded_json["array"] == [1, 2, 3]
    # Note: JSON doesn't preserve matrix structure, so we skip this test
    # @test loaded_json["matrix"] == [1 2; 3 4]
    @test loaded_json["nested"]["key"] == "value"

    # Test binary format
    io_binary = ConfigurationIO(
        config_file = "test.bin",
        validate_on_write = false,
        validate_on_read = false,
        supported_formats = [BINARY_FORMAT, JSON_FORMAT],
    )
    save_configuration(io_binary, test_data, BINARY_FORMAT)
    loaded_binary = load_configuration(io_binary, BINARY_FORMAT)
    @test loaded_binary["number"] == 42
    @test loaded_binary["string"] == "test"
    @test loaded_binary["array"] == [1, 2, 3]
    @test loaded_binary["matrix"] == [1 2; 3 4]
    @test loaded_binary["nested"]["key"] == "value"

    # Clean up
    rm("test.json", force = true)
    rm("test.bin", force = true)
end

@testitem "I/O Metadata Handling" begin
    using ManyVariableVariationalMonteCarlo

    # Test metadata handling
    io = ConfigurationIO()
    config = Dict{String,Any}("n_sites" => 10, "n_electrons" => 5, "n_samples" => 1000)

    # Save with metadata
    save_configuration(io, config, JSON_FORMAT)
    loaded_config = load_configuration(io, JSON_FORMAT)

    # Check metadata
    @test haskey(loaded_config, "_metadata")
    @test haskey(loaded_config["_metadata"], "created_at")
    @test haskey(loaded_config["_metadata"], "version")
    @test haskey(loaded_config["_metadata"], "format")
    @test loaded_config["_metadata"]["format"] == "JSON_FORMAT"

    # Clean up
    rm("config.json", force = true)
end
