#include "cpu_impl.hpp"

#include <util/logging.hpp>

#include <exception>

#include <Eigen/Core>
#include <Eigen/LU>

namespace simplex {
namespace cpu {

Tableau<double> create_tableau(const Problem& problem_stmt) {
	const auto indent = dout(DL::DBG1).indentWithTitle("create_tableau");

	const auto num_constraints = problem_stmt.num_constraints();
	const auto num_variables = problem_stmt.num_variables();

	dout(DL::DBG1) << "num_variables = " << num_variables << "\nnum_constraints = " << num_constraints << '\n';

	Eigen::MatrixXd constraint_matrix(num_constraints + 1, num_variables + 1);
	constraint_matrix.setZero();

	{int constr_count = 0;
	for (const auto& constr : problem_stmt.constraints()) {
		auto current_cm_row = constraint_matrix.row(constr_count + 1);

		current_cm_row(0) = constr.m_rhs;
		for (const auto& varid_and_val : constr.m_coeffs) {
			current_cm_row(varid_and_val.first.getValue() + 1) = varid_and_val.second;
		}

		constr_count += 1;
	}}

	{auto objective_coeffs = constraint_matrix.row(0);
	for (const auto& var_and_info : problem_stmt.variables()) {
		objective_coeffs(var_and_info.first.getValue() + 1) = var_and_info.second.m_coeff;
	}}

	dout(DL::DBG3) << "constraint matrix\n" << constraint_matrix << '\n';

	const auto& constraint_consts = constraint_matrix.leftCols<1>().segment(1, num_constraints);
	const auto& basis = Eigen::MatrixXd::Identity(num_constraints,num_constraints);
	const auto& basic_coeff = Eigen::RowVectorXd::Zero(num_constraints);
	const auto& nonbasics = constraint_matrix.bottomRightCorner(num_constraints, num_variables);
	const auto& nonbasics_coeff = constraint_matrix.topRows<1>().segment(1, num_variables);

	const auto& inv_basis = basis.inverse();
	const auto& inv_basis_times_nonbasis = inv_basis*nonbasics;
	const auto& inv_basis_times_constraint_coeffs = inv_basis*constraint_consts;

	const auto& upper_right = basic_coeff*inv_basis_times_nonbasis - nonbasics_coeff;
	const auto& lower_right = inv_basis_times_nonbasis;

	const auto& upper_left = basic_coeff*inv_basis_times_constraint_coeffs;
	const auto& lower_left = inv_basis_times_constraint_coeffs;

	(void)upper_right; // dout(DL::DBG3) << "upper_right:\n" << upper_right << '\n';
	(void)lower_right; // dout(DL::DBG3) << "lower_right:\n" << lower_right << '\n';

	(void)upper_left; // dout(DL::DBG3) << "upper_left:\n" << upper_left << '\n';
	(void)lower_left; // dout(DL::DBG3) << "lower_left:\n" << lower_left << '\n';

	// Eigen::MatrixXd tableau_data(num_constraints+1, num_variables+1); tableau_data <<
	// 	upper_left, upper_right, lower_left, lower_right
	// ;

	constraint_matrix.row(0) *= -1;
	constraint_matrix(0,0) = 0; // upper_left(0,0); // always zero
	const auto& tableau_data = constraint_matrix;

	dout(DL::DBG3) << "tableau_data:\n" << tableau_data << '\n';

	Tableau<double> result (
		new double[static_cast<std::size_t>(tableau_data.rows() * tableau_data.cols())],
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

boost::optional<VariableIndex> find_entering_variable(const util::PointerAndSize<double>& first_row) {
	const auto indent = dout(DL::DBG1).indentWithTitle("find_entering_variable");
	dout(DL::DBG2) << "first row given: ";
	util::print_container(dout(DL::DBG2), first_row);
	dout(DL::DBG2) << '\n';

	double lowest_value = 0.001;
	boost::optional<VariableIndex> result;

	for (int icol = 1; icol < first_row.size(); ++icol) {
		const auto& val = first_row.at(icol);
		if (val < lowest_value) {
			lowest_value = val;
			result = util::make_id<VariableIndex>(icol);
		}
	}

	if (result) {
		dout(DL::DBG1) << "found entering variable: " << *result << '\n';
	} else {
		dout(DL::DBG1) << "did not find a entering variable\n";
	}

	return result;
}

ThetaValuesAndEnteringColumn<double> get_theta_values_and_entering_column(const Tableau<double>& tab, VariableIndex entering) {
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

boost::optional<VariableIndex> find_leaving_variable(const ThetaValuesAndEnteringColumn<double>& tvals_and_centering) {
	const auto indent = dout(DL::DBG1).indentWithTitle("find_leaving_variable");
	dout(DL::DBG2) << "theta_values given: ";
	util::print_container(dout(DL::DBG2), tvals_and_centering.theta_values);
	dout(DL::DBG2) << "\nentering_column given: ";
	util::print_container(dout(DL::DBG2), tvals_and_centering.entering_column);
	dout(DL::DBG2) << '\n';

	auto lowest_theta_value = std::numeric_limits<double>::max();
	boost::optional<VariableIndex> result;

	for (int irow = 1; irow < (int)tvals_and_centering.theta_values.size(); ++irow) {
		const auto& theta_val = tvals_and_centering.theta_values.at((std::size_t)irow);
		const auto& tab_val = tvals_and_centering.entering_column.at((std::size_t)irow);
		if (tab_val > 0 && (!result || theta_val < lowest_theta_value)) {
			lowest_theta_value = theta_val;
			result = util::make_id<VariableIndex>(irow);
		}
	}

	if (result) {
		dout(DL::DBG1) << "found leaving variable: " << *result << '\n';
	} else {
		dout(DL::DBG1) << "did not find a leaving variable\n";
	}

	return result;
}

Tableau<double> update_leaving_row(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering) {

	const auto indent = dout(DL::DBG1).indentWithTitle("update_leaving_row");

	auto denom = entering_column.at((std::size_t)leaving_and_entering.leaving.getValue());
	// dout(DL::LINDA) << "index: " << (std::size_t)leaving_and_entering.leaving.getValue() << '\n';
	// dout(DL::LINDA) << "denom: " << denom << '\n';

	for (int icol = 0; icol < tab.width(); ++icol) {
		tab.at(leaving_and_entering.leaving, icol) /= denom;
	}

	dout(DL::DBG2) << "tableau after:\n" << tab << '\n';

	return tab;
}

Tableau<double> update_rest_of_basis(Tableau<double>&& tab, const std::vector<double>& entering_column, VariableIndex leaving) {
	const auto indent = dout(DL::DBG1).indentWithTitle("update_rest_of_basis");

	for (int irow = 0; irow < tab.height(); ++irow) {
		if (irow == leaving.getValue()) { continue; }
		const auto& entering_col_val = entering_column.at((std::size_t)irow);

		for (int icol = 0; icol < tab.width(); ++icol) {
			// dout(DL::LINDA) << "entering_col_val: " << entering_col_val << " tab.at(leaving, icol): " << tab.at(leaving, icol) << " leaving: " << leaving << " icol: " << icol << "\n";
			tab.at(irow, icol) -= tab.at(leaving, icol) * entering_col_val;
		}
	}

	dout(DL::DBG2) << "tableau after:\n" << tab << '\n';

	return tab;
}

Tableau<double> update_entering_column(Tableau<double>&& tab, const std::vector<double>& entering_column, VariablePair leaving_and_entering) {
	const auto indent = dout(DL::DBG1).indentWithTitle("update_entering_column");

	auto denom = entering_column.at((std::size_t)leaving_and_entering.leaving.getValue());

	// printf("index: %d denom: %f\n", leaving_and_entering.leaving.getValue(), denom);

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
