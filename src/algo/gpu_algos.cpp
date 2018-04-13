#include "gpu_algos.hpp"

#include <impl/cpu_impl.hpp>
#include <impl/gpu_impl.hpp>
#include <util/logging.hpp>

#include <cuda_runtime.h>
#include <chrono>

static const bool use_cpu_find_entering = false;
static const bool use_cpu_find_leaving = false;

namespace simplex{

boost::variant<
	Assignments,
	TableauErrors
> gpu_cpu_algo_from_paper(const Problem& problem) {
	using cpu::create_tableau;
	using gpu::update_leaving_row;
	using gpu::update_rest_of_basis;
	using gpu::update_entering_column;

	const auto indent = dout(DL::INFO).indentWithTitle("Algorithm from the Paper (GPU)");

	auto start_create = std::chrono::system_clock::now();;
	auto cpu_tableau = create_tableau(problem);
	auto end_create = std::chrono::system_clock::now();;
	double total_time_create = ((std::chrono::duration<double>)(end_create - start_create)).count();

	auto gpu_tableau = Tableau<double>(NULL, cpu_tableau.height(), cpu_tableau.width());

	auto cpu_first_row_memory_owner = std::vector<double>((std::size_t)cpu_tableau.width());
	auto cpu_first_row = util::PointerAndSize<double>(cpu_first_row_memory_owner);
	auto cpu_tv_and_centering = cpu::ThetaValuesAndEnteringColumn<double>(cpu_tableau.height());

	auto copy_tableau_gpu_to_cpu = [&]() { cudaMemcpy(cpu_tableau.data(), gpu_tableau.data(), static_cast<std::size_t>(gpu_tableau.data_size()), cudaMemcpyDeviceToHost); };
	auto copy_tableau_cpu_to_gpu = [&]() { cudaMemcpy(gpu_tableau.data(), cpu_tableau.data(), static_cast<std::size_t>(cpu_tableau.data_size()), cudaMemcpyHostToDevice); };

	cudaMalloc(&gpu_tableau.data(), static_cast<std::size_t>(cpu_tableau.data_size()));

	copy_tableau_cpu_to_gpu();


	int iteration_num = 1;

	double total_time = 0;
	double total_time_k2 = 0;
	double total_time_k3 = 0;
	double total_time_k4 = 0;
	auto start_time = std::chrono::system_clock::now();;
	while (true) {

		const auto indent = dout(DL::INFO).indentWithTitle([&](auto&& s){ s << "Iteration " << iteration_num; });

		if (dout(DL::DBG1).enabled()) {
			copy_tableau_gpu_to_cpu();
			dout(DL::DBG1) << "tableau:\n" << cpu_tableau << '\n';
		}


		boost::optional<VariableIndex> entering_var;
		if (use_cpu_find_entering) {
			using cpu::find_entering_variable;
			cudaMemcpy(cpu_first_row.data(), gpu_tableau.data(), (std::size_t)cpu_first_row.data_size(), cudaMemcpyDeviceToHost);
			entering_var = find_entering_variable(cpu_first_row);
		} else {
			const auto indent = dout(DL::DBG1).indentWithTitle("find_entering_variable");

			using gpu::find_entering_variable;
			util::PointerAndSize<double> gpu_first_row(gpu_tableau.data(), gpu_tableau.width());
			const auto entering_var_raw = find_entering_variable(gpu_first_row);

			if (entering_var_raw >= 0) {
				entering_var = util::make_id<VariableIndex>((VariableIndex::IDType)entering_var_raw);
			}
		}

		if (!entering_var) {
			dout(DL::INFO) << "Solution reached!\n";
			break;
		}
		
		// k1
		auto indent_gtvaec = dout(DL::DBG1).indentWithTitle("get_theta_values_and_entering_column");

		using gpu::get_theta_values_and_entering_column;
		auto gpu_tv_and_centering = get_theta_values_and_entering_column(gpu_tableau, *entering_var);

		indent_gtvaec.endIndent();

		boost::optional<VariableIndex> leaving_var;
		if (use_cpu_find_leaving) {
			cudaMemcpy(cpu_tv_and_centering.entering_column.data(), gpu_tv_and_centering.entering_column.data(), (std::size_t)gpu_tv_and_centering.entering_column.data_size(), cudaMemcpyDeviceToHost);
			cudaMemcpy(cpu_tv_and_centering.theta_values.data(), gpu_tv_and_centering.theta_values.data(), (std::size_t)gpu_tv_and_centering.theta_values.data_size(), cudaMemcpyDeviceToHost);
			leaving_var = find_leaving_variable(cpu_tv_and_centering);
		} else {
			const auto indent = dout(DL::DBG1).indentWithTitle("find_leaving_variable");
			const auto leaving_var_raw = find_leaving_variable(gpu_tv_and_centering);
			if (leaving_var_raw >= 0) {
				leaving_var = util::make_id<VariableIndex>((VariableIndex::IDType)leaving_var_raw);
			}
		}

		if (!leaving_var) {
			dout(DL::INFO) << "Problem is unbounded!\n";
			break;
		}

		VariablePair entering_and_leaving = {
			*entering_var,
			*leaving_var,
		};

		auto start_k2 = std::chrono::system_clock::now();
		{const auto indent = dout(DL::DBG1).indentWithTitle("update_leaving_row");
		update_leaving_row( // k2
			gpu_tableau,
			gpu_tv_and_centering.entering_column,
			entering_and_leaving
		);
			if (dout(DL::DBG2).enabled()) {
				copy_tableau_gpu_to_cpu();
				dout(DL::DBG2) << "tableau after:\n" << cpu_tableau << '\n';
			}
		}
		auto end_k2 = std::chrono::system_clock::now();

		auto start_k3 = std::chrono::system_clock::now();
		{const auto indent = dout(DL::DBG1).indentWithTitle("update_rest_of_basis");
		update_rest_of_basis( // k3
			gpu_tableau,
			gpu_tv_and_centering.entering_column,
			entering_and_leaving.leaving
		);
			if (dout(DL::DBG2).enabled()) {
				copy_tableau_gpu_to_cpu();
				dout(DL::DBG2) << "tableau after:\n" << cpu_tableau << '\n';
			}
		}
		auto end_k3 = std::chrono::system_clock::now();


		auto start_k4 = std::chrono::system_clock::now();
		{const auto indent = dout(DL::DBG1).indentWithTitle("update_entering_column");
		update_entering_column( //k4
			gpu_tableau,
			gpu_tv_and_centering.entering_column,
			entering_and_leaving
		);
			if (dout(DL::DBG2).enabled()) {
				copy_tableau_gpu_to_cpu();
				dout(DL::DBG2) << "tableau after:\n" << cpu_tableau << '\n';
			}
		}
		auto end_k4 = std::chrono::system_clock::now();

		cudaFree(gpu_tv_and_centering.theta_values.data());
		cudaFree(gpu_tv_and_centering.entering_column.data());
		iteration_num += 1;

		total_time_k2 += ((std::chrono::duration<double>)(end_k2 - start_k2)).count();
		total_time_k3 += ((std::chrono::duration<double>)(end_k3 - start_k3)).count();
		total_time_k4 += ((std::chrono::duration<double>)(end_k4 - start_k4)).count();

		if (iteration_num == 10000) break;
	}
	auto finish_time = std::chrono::system_clock::now();

	copy_tableau_gpu_to_cpu();

	{const auto indent = dout(DL::INFO).indentWithTitle("Result");
		dout(DL::INFO) << cpu_tableau << '\n';
	}

	cudaFree(gpu_tableau.data());
	delete cpu_tableau.data();

	total_time += ((std::chrono::duration<double>)(finish_time - start_time)).count();
	double average_time = total_time/(iteration_num-1);
	// double average_k2 = total_time_k2/(iteration_num-1);
	std::cout << "total_time: " << total_time <<  "\n";
	std::cout << "average_time: " << average_time <<  "\n";
	std::cout << "k2 total time: " << total_time_k2 <<  "\n";
	std::cout << "k3 total time: " << total_time_k3 <<  "\n";
	std::cout << "k4 total time: " << total_time_k4 <<  "\n";
	std::cout << "total_time_create: " << total_time_create <<  "\n";


	return Assignments{};
}

} // end namespace simplex
