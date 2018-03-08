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
	const std::string& dataFileName() const { return m_dataFileName; }
	int nThreads() const { return m_nThreads; }

	int getWidth() const { return *width; }
	int getHeight() const {return *height; }

private:
	friend struct ParsedArguments;
	friend ParsedArguments parse(int arc_int, char const** argv);

	ProgramConfig();

	std::string m_dataFileName;
	int m_nThreads;

	boost::optional<int> width;
	boost::optional<int> height;
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
