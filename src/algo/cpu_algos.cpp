#include "cpu_algos.hpp"

#include <impl/cpu_impl.hpp>

namespace simplex{
namespace cpu {

boost::variant<
	Assignments,
	TableauErrors
> algo_from_paper(const Problem& problem) {
	auto tableau = create_tableau(problem);

	while (true) {
		const auto entering_var = find_entering_variable(tableau);

		if (!entering_var) {
			return Assignments{};
		}
		
		const auto tv_and_centering = get_theta_values_and_entering_column(tableau, *entering_var);
		
		VariablePair entering_and_leaving = {
			*entering_var,
			find_leaving_variable(tv_and_centering),
		};

		tableau = update_entering_column(
			update_rest_of_basis(
				update_leaving_row(
					std::move(tableau),
					tv_and_centering.entering_column,
					entering_and_leaving
				),
				tv_and_centering.entering_column,
				entering_and_leaving.leaving
			),
			tv_and_centering.entering_column,
			entering_and_leaving
		);
	}
}

} // end namespace cpu
} // end namespace simplex
