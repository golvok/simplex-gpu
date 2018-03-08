#include "cpu_impl.hpp"

#include <util/logging.hpp>

#include <exception>

namespace simplex {
namespace cpu {

Tableau<double> create_tableau(const Problem& problem_stmt) {
	(void)problem_stmt;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

boost::optional<VariableID> find_entering_variable(const Tableau<double>& tab) {
	(void)tab;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableID entering) {
	(void)tab;
	(void)entering;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

VariableID find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering) {
	(void)tvals_and_centering;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

Tableau<double> update_leaving_row(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering) {
	(void)tab;
	(void)entering_column;
	(void)leaving_and_entering;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

Tableau<double> update_rest_of_basis(Tableau<double>&& tab, const std::vector<double>& entering_column, VariableID leaving) {
	(void)tab;
	(void)entering_column;
	(void)leaving;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

Tableau<double> update_entering_column(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering) {
	(void)tab;
	(void)entering_column;
	(void)leaving_and_entering;
	util::print_and_throw<std::logic_error>([](auto&& s) { s << "unimplemented"; });
}

} // end namespace simplex
} // end namespace cpu
