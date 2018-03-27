
#include <algo/cpu_algos.hpp>
#include <datastructures/tableau.hpp>
#include <impl/gpu_impl.hpp>
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
			simplex::RandomProblemSpecification rps(*config.num_variables, *config.num_constraints);
			rps.density = *config.constraint_density;
			rps.random_seed = config.random_problem_seed;
			if (config.force_problem_padding) {
				const auto problem_constraints = simplex::gpu::problem_constraints();
				return pad_with_zeroes_modulo(
					generate_random_problem(rps),
					problem_constraints.height_modulus,
					problem_constraints.width_modulus
				);
			} else {
				return generate_random_problem(rps);
			}
		} else {
			return simplex::make_small_sample_problem();
		}
	}();

	auto result = cpu_only_algo_from_paper(problem);
	(void) result;

	return 0;
}