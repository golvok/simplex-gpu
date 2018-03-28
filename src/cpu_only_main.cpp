
#include <algo/cpu_algos.hpp>
#include <datastructures/tableau.hpp>
#include <run/common_ui.hpp>
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

	const auto& common_data = simplex::common_cmdline_ui(config);
	const auto& problem = common_data.problem;

	auto result = cpu_only_algo_from_paper(problem);
	(void) result;

	return 0;
}