#include "gpu_algos.hpp"

#include <impl/cpu_impl.hpp>
#include <impl/gpu_impl.hpp>
#include <util/logging.hpp>

namespace simplex{

boost::variant<
	Assignments,
	TableauErrors
> gpu_cpu_algo_from_paper(const Problem& problem) {
	using cpu::create_tableau;
	using cpu::get_theta_values_and_entering_column;
	using cpu::find_entering_variable;
	using gpu::update_leaving_row;
	using gpu::update_rest_of_basis;
	using gpu::update_entering_column;

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
		auto tv_and_centering = get_theta_values_and_entering_column(tableau, *entering_var);
		
		VariablePair entering_and_leaving = {
			*entering_var,
			find_leaving_variable(tv_and_centering),
		};

		update_leaving_row( // k2
			tableau,
			tv_and_centering.entering_column,
			entering_and_leaving
		),
		update_rest_of_basis( // k3
			tableau,
			tv_and_centering.entering_column,
			entering_and_leaving.leaving
		),
		update_entering_column( //k4
			tableau,
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

} // end namespace simplex
