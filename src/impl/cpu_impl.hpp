#ifndef IMLP__CPU_IMPL_H
#define IMLP__CPU_IMPL_H

#include <datastructures/problem.hpp>
#include <datastructures/tableau.hpp>
#include <impl/impl_common.hpp>

#include <vector>

#include <boost/optional.hpp>

namespace simplex {
namespace cpu {

template<typename FloatType>
struct ThetaValuesAndEnteringColumn {
	std::vector<FloatType> theta_values;
	std::vector<FloatType> entering_column;
};

Tableau<double> create_tableau(const Problem& problem_stmt);

boost::optional<VariableID> find_entering_variable(const Tableau<double>& tab);

ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableID entering);

VariableID find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering);

Tableau<double> update_leaving_row(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering);

Tableau<double> update_rest_of_basis(Tableau<double>&& tab, const std::vector<double>& entering_column, VariableID leaving);

Tableau<double> update_entering_column(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering);

} // end namespace simplex
} // end namespace cpu

#endif /* IMLP__CPU_IMPL_H */
