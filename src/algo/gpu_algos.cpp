#include "gpu_algos.hpp"

#include <impl/cpu_impl.hpp>
#include <util/logging.hpp>

namespace simplex{
namespace cpu {

boost::variant<
	Assignments,
	TableauErrors
> algo_from_paper(const Problem& problem) {
	const auto indent = dout(DL::INFO).indentWithTitle("Algorithm from the Paper (CPU)");
	auto tableau = create_tableau(problem);
	int iteration_num = 1;

	while (true) {
		const auto indent = dout(DL::INFO).indentWithTitle([&](auto&& s){ s << "Iteration " << iteration_num; });
		dout(DL::DBG1) << "tableau:\n" << tableau << '\n';

		const auto entering_var = find_entering_variable(tableau);

		if (!entering_var) {
			break;
		}
		
		// k1
		const auto tv_and_centering = get_theta_values_and_entering_column(tableau, *entering_var);
		
		VariablePair entering_and_leaving = {
			*entering_var,
			find_leaving_variable(tv_and_centering),
		};

		tableau = update_entering_column( //k4
			update_rest_of_basis( // k3
				update_leaving_row( // k2
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

		iteration_num += 1;
	}

	{const auto indent = dout(DL::INFO).indentWithTitle("Result");
		dout(DL::INFO) << tableau;
	}

	return Assignments{};
}

} // end namespace cpu
} // end namespace simplex
