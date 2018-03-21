#include "cpu_impl.hpp"

#include <util/logging.hpp>

#include <exception>

#include <Eigen/Core>
#include <Eigen/LU>

namespace simplex {
namespace cpu {

Tableau<double> create_tableau(const Problem& problem_stmt) {
	/* A' \in R^mXn, c' \in R^n
	 * max c'x' s.t. A'x' \leq b' && x' \geq 0
	 * ===
	 * A  = (A' I) \in R^mX(n+m), c = (c', 0, ...) \in R^(n+m)
	 * max cx s.t. Ax = b && x \geq 0
	 *   def: A = (B, N), B \in R^mXm, B is "basic matrix", N nonbasic
	 *  note: wikipedia's \lambda = (B^T)^-1 c_B
	 *                    and s_N = c_N - N^T \lambda = c_N - N^T (B^T)^-1 c_b
	 * steps:
	 *   use formulae from paper to compute 1st column and rest-of-tableau sections
	 *     - will have to use matrix library - involves inverting matrices.
	 *     -
	 */

	const auto indent = dout(DL::DBG1).indentWithTitle("create_tableau");
	(void)problem_stmt;

	const long num_constraints = 2;
	const long num_variables = 3;

	Eigen::VectorXd constraint_consts(2); constraint_consts <<
		10, 15
	;

	Eigen::MatrixXd basis(num_constraints,num_constraints); basis <<
		3, 2,
		2, 5
	;

	Eigen::RowVectorXd basic_coeff(num_constraints); basic_coeff <<
		-2, -3
	;

	Eigen::MatrixXd nonbasics(num_constraints,num_variables); nonbasics <<
		1, 1, 0,
		3, 0, 1
	;

	Eigen::RowVectorXd nonbasics_coeff(num_variables); nonbasics_coeff <<
		-4, 0, 0
	;

	const auto& inv_basis = basis.inverse();
	const auto& inv_basis_times_nonbasis = inv_basis*nonbasics;
	const auto& inv_basis_times_constraint_coeffs = inv_basis*constraint_consts;

	const auto& upper_right = basic_coeff*inv_basis_times_nonbasis - nonbasics_coeff;
	const auto& lower_right = inv_basis_times_nonbasis;

	const auto& upper_left = basic_coeff*inv_basis_times_constraint_coeffs;
	const auto& lower_left = inv_basis_times_constraint_coeffs;

	dout(DL::DBG3) << "upper_right:\n" << upper_right << '\n';
	dout(DL::DBG3) << "lower_right:\n" << lower_right << '\n';

	dout(DL::DBG3) << "upper_left:\n" << upper_left << '\n';
	dout(DL::DBG3) << "lower_left:\n" << lower_left << '\n';

	Eigen::MatrixXd tableau_data(num_constraints+1, num_variables+1); tableau_data <<
		upper_left, upper_right, lower_left, lower_right
	;

	dout(DL::DBG3) << "tableau_data:\n" << tableau_data << '\n';

	Tableau<double> result (
		tableau_data.rows(),
		tableau_data.cols()
	);

	for (std::ptrdiff_t irow = 0; irow < tableau_data.rows(); ++irow) {
		for (std::ptrdiff_t icol = 0; icol < tableau_data.cols(); ++icol) {
			result.at(irow, icol) = tableau_data(irow, icol);
		}
	}

	return result;
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
		result.theta_values.at((std::size_t)irow) = tab.at(irow, 0)/val_at_entering;
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
		if (tab_val > 0 && (!result || theta_val < lowest_theta_value)) {
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

	dout(DL::DBG2) << "tableau after:\n" << tab << '\n';

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

	dout(DL::DBG2) << "tableau after:\n" << tab << '\n';

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

	dout(DL::DBG2) << "tableau after:\n" << tab << '\n';

	return tab;
}

} // end namespace simplex
} // end namespace cpu
