
#include "common_ui.hpp"

#include <impl/gpu_impl.hpp>

#include <fstream>
#include <unordered_set>

#include <boost/program_options.hpp>

namespace boost {
	template<class T>
	void validate(boost::any& v, std::vector<std::string> const& values, boost::optional<T>*, int) {
	    if (!values.empty()) {
	        boost::any a;
	        using namespace boost::program_options;
	        validate(a, values, (T*)0, 0);
	        v = boost::any(boost::optional<T>(boost::any_cast<T>(a)));
	    }
	}
}

namespace simplex {
namespace cmdargs {

MetaConfig::MetaConfig()
	: levels_to_enable(DebugLevel::getDefaultSet())
{ }

ProgramConfig::ProgramConfig()
	: m_dataFileName()
	, m_nThreads(1)
	, use_random_problem(false)
	, random_problem_seed()
	, num_variables()
	, num_constraints()
	, constraint_density()
	, force_problem_padding(false)
	, problem_dump_filename()
{ }


ParsedArguments::ParsedArguments(int argc_int, char const** argv)
	: m_meta()
	, m_programConfig()
{
	namespace po = boost::program_options;

	po::options_description metaopts("Meta Options");
	po::options_description progopts("Program Options");

	metaopts.add_options()
		("debug",    "Turn on the most common debugging options")
	;
	DebugLevel::forEachLevel([&](DebugLevel::Level l) {
		metaopts.add_options()(("DL::" + DebugLevel::getAsString(l)).c_str(), "debug flag");
	});

	progopts.add_options()
		("help,h", "print help message")
		// ("file", po::value(&m_programConfig.m_dataFileName)->required(), "The file to use")
		// ("num-threads", po::value(&m_programConfig.m_nThreads), "The maximum nuber of simultaneous threads to use")
		("random-problem,r", po::bool_switch(&m_programConfig.use_random_problem), "Generate a random problem")
		("random-problem-seed,s", po::value(&m_programConfig.random_problem_seed), "(Optional) random seed for random problem generation")
		("num-variables", po::value(&m_programConfig.num_variables), "Number of variables in random problem")
		("num-constraints", po::value(&m_programConfig.num_constraints), "Number of constraints in random problem")
		("constraint-density", po::value(&m_programConfig.constraint_density), "Chance that a given variable in included in a constraint")
		("force-padding", po::bool_switch(&m_programConfig.force_problem_padding), "Force padding the problem with zeroes to match GPU implementation constraints")
		("dump-problem", po::value(&m_programConfig.problem_dump_filename), "Before starting the solve, save the problem in this file")
	;

	po::options_description allopts;
	allopts.add(progopts).add(metaopts);

	po::variables_map vm;
	po::store(po::parse_command_line(argc_int, argv, allopts), vm);

	// check for help flag before call to notify - don't care about required arguments in this case.
	if (vm.count("help")) {
		std::cerr << allopts << std::endl;
		exit(0);
	}

	po::notify(vm);

	if (vm.count("debug")) {
		auto debug_levels = DebugLevel::getStandardDebug();
		m_meta.levels_to_enable.insert(end(m_meta.levels_to_enable),begin(debug_levels),end(debug_levels));
	}

	DebugLevel::forEachLevel([&](DebugLevel::Level l) {
		if (vm.count("DL::" + DebugLevel::getAsString(l))) {
			auto levels_in_chain = DebugLevel::getAllShouldBeEnabled(l);
			m_meta.levels_to_enable.insert(end(m_meta.levels_to_enable),begin(levels_in_chain),end(levels_in_chain));
		}
	});
}

ParsedArguments parse(int arc_int, char const** argv) {
	return ParsedArguments(arc_int, argv);
}

} // end namespace cmdargs

CommonCmdlineData common_cmdline_ui(const cmdargs::ProgramConfig& config) {
	auto problem = [&]() {
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

	if (config.problem_dump_filename) {
		std::ofstream prob_dumpfile(*config.problem_dump_filename);
		prob_dumpfile << problem << '\n';
	}

	return {
		problem,
	};
}

} // end namespace simplex
