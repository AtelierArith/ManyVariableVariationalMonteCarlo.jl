# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core Julia package (`ManyVariableVariationalMonteCarlo.jl` plus modules like `hamiltonian.jl`, `stdface.jl`, `vmcmain.jl`).
- `test/`: Tests using ReTestItems; entry at `test/runtests.jl`; files like `*_tests.jl` contain `@testitem` blocks.
- `examples/`: Runnable scripts (e.g., `03_basic_workflow.jl`, `04_stdface_lattices.jl`).
- `Project.toml` / `Manifest.toml`: Julia environment (targets Julia 1.10).
- `JuliaFormatter.toml`: Code style configuration (`style = "sciml"`).
- `mVMC/`, `mVMC-tutorial/`: Upstream C implementation and tutorial materials (not required to run package tests).

## Build, Test, and Development Commands
- Install deps: `julia --project -e 'using Pkg; Pkg.instantiate()'`.
- Precompile (faster dev): `julia --project -e 'using Pkg; Pkg.precompile()'`.
- Run all tests: `julia --project -e 'using Pkg; Pkg.test()'`.
- Run one test file: `julia --project test/stdface_tests.jl`.
- Run an example: `julia --project examples/03_basic_workflow.jl`.

## Coding Style & Naming Conventions
- Use 4‑space indentation; no tabs. Prefer concise, pure functions.
- Naming: Types/Modules CamelCase (`Hamiltonian`), functions/variables snake_case (`lattice_summary`), constants UPPER_SNAKE (`AMP_MAX`).
- File names: `lower_snake.jl`; test files end with `_tests.jl`.
- Formatting: JuliaFormatter with SciML style. Example:
  `julia --project -e 'using JuliaFormatter; format("src"); format("test")'`.

## Testing Guidelines
- Framework: `Test` + `ReTestItems` (`@testitem`). Add tests near related functionality.
- Determinism: use `StableRNGs` for seeded randomness when relevant.
- Coverage: keep or improve; include edge cases for lattice/model constructors and I/O paths.

## Commit & Pull Request Guidelines
- Commit style: imperative, scoped if useful (Conventional Commits preferred).
  Examples: `feat(stdface): add ladder lattice`, `fix(vmc): handle zero samples`, `docs: improve README examples`.
- PRs must: describe changes and rationale, link issues, include tests for behavior changes, update examples/docs when APIs change, and pass `Pkg.test()` locally. Add screenshots/log snippets for CLI or output changes when helpful.

## Tips & Notes
- Examples may write `zvo_*.dat` and other outputs; run them in a scratch directory and do not commit generated data.
- HDF5/JSON I/O is supported; large files should be ignored in VCS.
