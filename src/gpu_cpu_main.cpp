
#include <algo/gpu_algos.hpp>
#include <impl/gpu_impl.hpp>
#include <datastructures/tableau.hpp>
#include <parsing/cmdargs.hpp>
#include <util/logging.hpp>

using simplex::cmdargs::ProgramConfig;

int program_main(const ProgramConfig& config);

int main(int argc, char const** argv) {

	dout.setHighestTitleRank(7);

	const auto parsed_args = simplex::cmdargs::parse(argc,argv);
	// enable logging levels
	for (auto& l : parsed_args.meta().getDebugLevelsToEnable()) {
		dout.enable_level(l);
	}

	const auto result = program_main(parsed_args.programConfig());

	return result;
}

int program_main(const ProgramConfig& config) {

	const auto& problem = [&]() {
		if (config.use_random_problem) {
			const auto problem_constraints = simplex::gpu::problem_constraints();

			simplex::RandomProblemSpecification rps(*config.num_variables, *config.num_constraints);
			rps.density = *config.constraint_density;
			rps.random_seed = config.random_problem_seed;

			return pad_with_zeroes_modulo(
				generate_random_problem(rps),
				problem_constraints.height_modulus,
				problem_constraints.width_modulus
			);
		} else {
			return simplex::make_small_sample_problem();
		}
	}();

	auto result = gpu_cpu_algo_from_paper(problem);
	(void) result;

	return 0;
}
