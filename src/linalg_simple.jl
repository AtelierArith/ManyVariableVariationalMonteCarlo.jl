"""
Simplified Linear Algebra backend for ManyVariableVariationalMonteCarlo.jl

Provides basic linear algebra operations without workspace dependencies.
This file contains only the is_antisymmetric function and references to the main linalg.jl.
"""

using LinearAlgebra
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK

# is_antisymmetric is defined in linalg.jl

# All other linear algebra functions are defined in linalg.jl:
# - PfaffianLimitError, PFAFFIAN_LIMIT
# - pfaffian, _pfaffian_skew, pfaffian_and_inverse
# - pfaffian_det_relation, pfaffian_skew_symmetric
# - MatrixCalculation, update_matrix!
# - sherman_morrison_update!, woodbury_update!, matrix_ratio
# - get_matrix_calculation, clear_matrix_calculations!
# - benchmark_linalg
# - optimized BLAS/LAPACK operations
# - ThreadSafeMatrixOperations
