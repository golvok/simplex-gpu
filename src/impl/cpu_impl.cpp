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
	const auto indent = dout(DL::DBG1).indentWithTitle("find_entering_variable");
	double lowest_value = 0.0;
	boost::optional<VariableID> result;

	for (int icol = 1; icol < tab.width(); ++icol) {
		const auto& val = tab.at(0, icol);
		if (val < lowest_value) {
			lowest_value = val;
			result = util::make_id<VariableID>(icol);
		}
	}

	if (result) {
		dout(DL::DBG1) << "found entering variable: " << *result << '\n';
	} else {
		dout(DL::DBG1) << "did not find a entering variable\n";
	}

	return result;
}

ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableID entering) {
	const auto indent = dout(DL::DBG1).indentWithTitle("get_theta_values_and_entering_column");
	ThetaValuesAndEnteringColumn<double> result (
		tab.height()
	);

	for (int irow = 0; irow < tab.height(); ++irow) {
		const auto& val_at_entering = tab.at(irow, entering);
		result.entering_column.at((std::size_t)irow) = val_at_entering;
		result.theta_values.at((std::size_t)irow) = tab.at(irow, 1)/val_at_entering;
	}

	dout(DL::DBG1) << "theta_values computed: ";
	util::print_container(dout(DL::DBG1), result.theta_values);
	dout(DL::DBG1) << "\nentering_column copied: ";
	util::print_container(dout(DL::DBG1), result.entering_column);
	dout(DL::DBG1) << '\n';

	return result;
}

VariableID find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering) {
	const auto indent = dout(DL::DBG1).indentWithTitle("find_leaving_variable");

	auto lowest_theta_value = std::numeric_limits<double>::max();
	boost::optional<VariableID> result;

	for (int irow = 1; irow < (int)tvals_and_centering.theta_values.size(); ++irow) {
		const auto& theta_val = tvals_and_centering.theta_values.at((std::size_t)irow);
		const auto& tab_val = tvals_and_centering.entering_column.at((std::size_t)irow);
		if (tab_val < 0 && (!result || theta_val < lowest_theta_value)) {
			lowest_theta_value = theta_val;
			result = util::make_id<VariableID>(irow);
		}
	}

	if (result) {
		dout(DL::DBG1) << "found leaving variable: " << *result << '\n';
	} else {
		dout(DL::DBG1) << "did not find a leaving variable\n";
	}

	return *result;
}

Tableau<double> update_leaving_row(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering) {
	const auto indent = dout(DL::DBG1).indentWithTitle("update_leaving_row");

	auto denom = entering_column.at((std::size_t)leaving_and_entering.leaving.getValue());

	for (int icol = 0; icol < tab.width(); ++icol) {
		tab.at(leaving_and_entering.leaving, icol) /= denom;
	}

	return tab;
}

Tableau<double> update_rest_of_basis(Tableau<double>&& tab, const std::vector<double>& entering_column, VariableID leaving) {
	const auto indent = dout(DL::DBG1).indentWithTitle("update_rest_of_basis");

	for (int irow = 0; irow < tab.height(); ++irow) {
		if (irow == leaving.getValue()) { continue; }
		const auto& entering_col_val = entering_column.at((std::size_t)irow);

		for (int icol = 0; icol < tab.width(); ++icol) {
			tab.at(irow, icol) -= tab.at(leaving, icol) * entering_col_val;
		}
	}

	return tab;
}

Tableau<double> update_entering_column(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering) {
	const auto indent = dout(DL::DBG1).indentWithTitle("update_entering_column");

	auto denom = entering_column.at((std::size_t)leaving_and_entering.leaving.getValue());

	for (int irow = 0; irow < tab.height(); ++irow) {
		if (irow == leaving_and_entering.leaving.getValue()) {
			tab.at(irow, leaving_and_entering.entering) = 1/denom;
		} else {
			tab.at(irow, leaving_and_entering.entering) = - entering_column.at((std::size_t)irow)/denom;
		}
	}

	return tab;
}

} // end namespace simplex
} // end namespace cpu
