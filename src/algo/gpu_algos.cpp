#include "gpu_algos.hpp"

#include <impl/cpu_impl.hpp>
#include <impl/gpu_impl.hpp>
#include <util/logging.hpp>

#include <cuda_runtime.h>

namespace simplex{

boost::variant<
	Assignments,
	TableauErrors
> gpu_cpu_algo_from_paper(const Problem& problem) {
	using cpu::create_tableau;
	using cpu::find_entering_variable;
	using cpu::get_theta_values_and_entering_column;
	using cpu::find_leaving_variable;
	using gpu::update_leaving_row;
	using gpu::update_rest_of_basis;
	using gpu::update_entering_column;

	const auto indent = dout(DL::INFO).indentWithTitle("Algorithm from the Paper (CPU)");

	auto cpu_tableau = create_tableau(problem);
	auto gpu_tableau = Tableau<double>(NULL, cpu_tableau.width(), cpu_tableau.height());

	auto gpu_tv_and_centering = gpu::ThetaValuesAndEnteringColumn<double>(cpu_tableau.height());

	auto copy_tableau_gpu_to_cpu = [&]() { cudaMemcpy(cpu_tableau.data(), gpu_tableau.data(), static_cast<std::size_t>(gpu_tableau.data_size()), cudaMemcpyDeviceToHost); };
	auto copy_tableau_cpu_to_gpu = [&]() { cudaMemcpy(gpu_tableau.data(), cpu_tableau.data(), static_cast<std::size_t>(cpu_tableau.data_size()), cudaMemcpyHostToDevice); };

	cudaMalloc(&gpu_tableau.data(), static_cast<std::size_t>(cpu_tableau.data_size()));
	copy_tableau_cpu_to_gpu();
	int iteration_num = 1;

	while (true) {
		const auto indent = dout(DL::INFO).indentWithTitle([&](auto&& s){ s << "Iteration " << iteration_num; });
		copy_tableau_gpu_to_cpu();

		if (dout(DL::DBG1).enabled()) {
			dout(DL::DBG1) << "tableau:\n" << cpu_tableau << '\n';
		}

		const auto entering_var = find_entering_variable(cpu_tableau);

		if (!entering_var) {
			break;
		}
		
		// k1
		auto cpu_tv_and_centering = get_theta_values_and_entering_column(cpu_tableau, *entering_var);
		
		VariablePair entering_and_leaving = {
			*entering_var,
			find_leaving_variable(cpu_tv_and_centering),
		};

		cudaMemcpy(gpu_tv_and_centering.entering_column.data(), cpu_tv_and_centering.entering_column.data(), (std::size_t)gpu_tv_and_centering.entering_column.data_size(), cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_tv_and_centering.theta_values.data(), cpu_tv_and_centering.theta_values.data(), (std::size_t)gpu_tv_and_centering.theta_values.data_size(), cudaMemcpyHostToDevice);

		update_leaving_row( // k2
			gpu_tableau,
			gpu_tv_and_centering.entering_column,
			entering_and_leaving
		),
		update_rest_of_basis( // k3
			gpu_tableau,
			gpu_tv_and_centering.entering_column,
			entering_and_leaving.leaving
		),
		update_entering_column( //k4
			gpu_tableau,
			gpu_tv_and_centering.entering_column,
			entering_and_leaving
		);

		iteration_num += 1;
	}

	copy_tableau_gpu_to_cpu();

	{const auto indent = dout(DL::INFO).indentWithTitle("Result");
		dout(DL::INFO) << cpu_tableau << '\n';
	}

	cudaFree(gpu_tv_and_centering.theta_values.data());
	cudaFree(gpu_tv_and_centering.entering_column.data());
	cudaFree(gpu_tableau.data());
	delete cpu_tableau.data();

	return Assignments{};
}

} // end namespace simplex
