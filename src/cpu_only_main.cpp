
#include <datastructures/tableau.hpp>
#include <algo/cpu_algos.hpp>
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
	using namespace simplex;
	using namespace simplex::cpu;

	dout(DL::INFO) << config.getWidth();

	Problem problem;
	auto result = algo_from_paper(problem);
	(void) result;

	return 0;
}