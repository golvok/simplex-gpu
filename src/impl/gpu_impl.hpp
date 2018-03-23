#ifndef IMLP__GPU_IMPL_H
#define IMLP__GPU_IMPL_H

// #include <datastructures/problem.hpp>
#include <datastructures/tableau.hpp>
#include <impl/impl_common.hpp>
#include <util/primitive_structures.hpp>

#include <cuda_runtime.h>

namespace simplex {
namespace gpu {

template<typename FloatType>
struct ThetaValuesAndEnteringColumn {
	util::PointerAndSize<FloatType> theta_values;
	util::PointerAndSize<FloatType> entering_column;

	ThetaValuesAndEnteringColumn(std::ptrdiff_t height)
		: theta_values(NULL, height)
		, entering_column(NULL, height)
	{
		cudaMalloc(&theta_values.data(), static_cast<std::size_t>(theta_values.size()));
		cudaMalloc(&entering_column.data(), static_cast<std::size_t>(entering_column.size()));
	}

	// ThetaValuesAndEnteringColumn(const ThetaValuesAndEnteringColumn&) = default;
	// ThetaValuesAndEnteringColumn(ThetaValuesAndEnteringColumn&&) = default;
};

// Tableau<double> create_tableau(const Problem& problem_stmt);

// find smallest also negative value in the first row
// boost::optional<VariableIndex> find_entering_variable(const Tableau<double>& tab);

ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableIndex entering);

VariableIndex find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering);

void update_leaving_row(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariablePair leaving_and_entering);

void update_rest_of_basis(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariableIndex leaving);

void update_entering_column(Tableau<double>& tab, const util::PointerAndSize<double>& entering_column, VariablePair leaving_and_entering);

} // end namespace simplex
} // end namespace gpu

#endif /* IMLP__GPU_IMPL_H */
