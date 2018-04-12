#include "cpu_algos.hpp"

#include <impl/cpu_impl.hpp>
#include <util/logging.hpp>
#include <chrono>

namespace simplex{

boost::variant<
	Assignments,
	TableauErrors
> cpu_only_algo_from_paper(const Problem& problem) {
	using cpu::create_tableau;
	using cpu::find_entering_variable;
	using cpu::get_theta_values_and_entering_column;
	using cpu::find_leaving_variable;
	using cpu::update_leaving_row;
	using cpu::update_rest_of_basis;
	using cpu::update_entering_column;

	const auto indent = dout(DL::INFO).indentWithTitle("Algorithm from the Paper (CPU)");
	auto tableau = create_tableau(problem);
	int iteration_num = 1;

	double total_time = 0;
	while (true) {
		auto start = std::chrono::system_clock::now();;

		const auto indent = dout(DL::INFO).indentWithTitle([&](auto&& s){ s << "Iteration " << iteration_num; });
		dout(DL::DBG1) << "tableau:\n" << tableau << '\n';

		const auto entering_var = find_entering_variable(util::PointerAndSize<double>(tableau.data(), tableau.width()));

		if (!entering_var) {
			dout(DL::INFO) << "Solution reached!\n";
			break;
		}
		
		const auto tv_and_centering = get_theta_values_and_entering_column(tableau, *entering_var);
		
		const auto leaving_var = find_leaving_variable(tv_and_centering);

		if (!leaving_var) {
			dout(DL::INFO) << "Problem is unbounded!\n";
			break;
		}

		VariablePair entering_and_leaving = {
			*entering_var,
			*leaving_var,
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

		iteration_num += 1;

		auto finish = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = finish-start;
		total_time += elapsed_seconds.count();

		if (iteration_num == 10000) break;
	}

	{const auto indent = dout(DL::INFO).indentWithTitle("Result");
		dout(DL::INFO) << tableau << '\n';
	}

	delete tableau.data();

	double average_time = total_time/(iteration_num-1);
	std::cout << "total_time: " << total_time <<  "\n";
	std::cout << "average_time: " << average_time <<  "\n";

	return Assignments{};
}

} // end namespace simplex
