
#include <algo/gpu_algos.hpp>
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
	using simplex::make_small_sample_problem;

	const auto& problem = make_small_sample_problem();
	auto result = gpu_cpu_algo_from_paper(problem);
	(void) result;

	return 0;
}
