#ifndef IMLP__CPU_IMPL_H
#define IMLP__CPU_IMPL_H

#include <datastructures/problem.hpp>
#include <datastructures/tableau.hpp>
#include <impl/impl_common.hpp>
#include <util/primitive_structures.hpp>
#include <util/print_printable.hpp>

#include <vector>

#include <boost/optional.hpp>

namespace simplex {
namespace cpu {

#if __cplusplus >= 201103L
template<typename FloatType>
struct ThetaValuesAndEnteringColumn : util::print_printable {
	std::vector<FloatType> theta_values;
	std::vector<FloatType> entering_column;

	ThetaValuesAndEnteringColumn(std::ptrdiff_t height)
		: theta_values((std::size_t)height)
		, entering_column((std::size_t)height)
	{ }

	ThetaValuesAndEnteringColumn(const ThetaValuesAndEnteringColumn&) = default;
	ThetaValuesAndEnteringColumn(ThetaValuesAndEnteringColumn&&) = default;
	ThetaValuesAndEnteringColumn& operator=(const ThetaValuesAndEnteringColumn&) = default;
	ThetaValuesAndEnteringColumn& operator=(ThetaValuesAndEnteringColumn&&) = default;

	template<typename STREAM>
	void print(STREAM& os) const {
		os << "{tv = {";
		for (const auto& tv : theta_values) {
			os << tv << ", ";
		}
		os << "}, ec = {";
		for (const auto& ecv : entering_column) {
			os << ecv << ", ";
		}
		os << "}}";
	}
};
#endif

Tableau<double> create_tableau(const Problem& problem_stmt);

// find smallest also negative value in the first row
boost::optional<VariableIndex> find_entering_variable(const util::PointerAndSize<double>& first_row);

ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableIndex entering);

boost::optional<VariableIndex> find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering);

#if __cplusplus >= 201103L
Tableau<double> update_leaving_row(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering);

Tableau<double> update_rest_of_basis(Tableau<double>&& tab, const std::vector<double>& entering_column, VariableIndex leaving);

Tableau<double> update_entering_column(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering);
#endif

} // end namespace simplex
} // end namespace cpu

#endif /* IMLP__CPU_IMPL_H */
