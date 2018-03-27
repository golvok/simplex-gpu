#ifndef PARSING__CMDARGS_H
#define PARSING__CMDARGS_H

#include <util/logging.hpp>

#include <string>

#include <boost/optional.hpp>

namespace simplex {
namespace cmdargs {

struct ParsedArguments;

struct MetaConfig {
	/**
	 * Return the list of debug levels that shold be enabled given the command line options
	 * NOTE: May contain duplicates.
	 */
	const std::vector<DebugLevel::Level> getDebugLevelsToEnable() const {
		return levels_to_enable;
	}

private:
	friend struct ParsedArguments;
	friend ParsedArguments parse(int arc_int, char const** argv);
	
	MetaConfig();

	/// The printing levels that should be enabled. Duplicate entries are possible & allowed
	std::vector<DebugLevel::Level> levels_to_enable;
};

struct ProgramConfig {
	friend struct ParsedArguments;
	friend ParsedArguments parse(int arc_int, char const** argv);

	ProgramConfig();

	std::string m_dataFileName;
	int m_nThreads;

	bool use_random_problem;
	boost::optional<unsigned long> random_problem_seed;
	boost::optional<int> num_variables;
	boost::optional<int> num_constraints;
	boost::optional<double> constraint_density;
	bool force_problem_padding;
};

struct ParsedArguments {
	const MetaConfig& meta() const { return m_meta; }
	const ProgramConfig& programConfig() const { return m_programConfig; }

private:
	friend ParsedArguments parse(int arc_int, char const** argv);

	MetaConfig m_meta;
	ProgramConfig m_programConfig;

	ParsedArguments(int arc_int, char const** argv);
};

/**
 * call this function to parse and return a ParsedArguments object from the
 * arguments, which then can be queried about the various options.
 */
ParsedArguments parse(int arc_int, char const** argv);

} // end namespace cmdargs
} // end namespace simplex

#endif /* PARSING__CMDARGS_H */
