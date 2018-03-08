
#ifndef UTIL__LOGGING_H
#define UTIL__LOGGING_H

#include <util/utils.hpp>

#include <bitset>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/iostreams/filtering_stream.hpp>

class IndentingLeveledDebugPrinter;

namespace DebugLevel {
	/**
	 * The Enum of the levels usable with IndentingLeveledDebugPrinter
	 * If adding a level make sure to update the various functions in this namespace.
	 */
	enum Level : uint {
		LEVEL_START = 0,
		INFO = LEVEL_START,  // probably always going to have this on?
		WARN,  // same as ^ ?
		ERROR, // always on

		ROUTE_D1, // Passenger Routing debug
		ROUTE_D2, // Passenger Routing debug level 2
		ROUTE_D3, // Passenger Routing debug level 3

		PIN_BY_PIN_STEP, // Stepping though the results of routing each pin
		MAZE_ROUTE_STEP, // Stepping though the algorithm
		ROUTE_TIME, // route time measurement
		ROUTE_D4, // Passenger Routing lowest level debug

		APL_D1, // Analytic PLacement debug
		APL_D2, // Analytic PLacement debug level 2
		APL_D3, // Analytic PLacement debug level 3
		APL_D4, // Analytic PLacement lowest level debug

		DATA_READ1, // reading of data

		LEVEL_COUNT, // please make sure this is at the end
	};

	/**
	 * Get the default set of print levels that should probably always be enabled
	 * Most code assumes these are already on.
	 */
	std::vector<Level> getDefaultSet();

	/**
	 * Get all the levels that you might ever want if debugging was your goal
	 */
	std::vector<Level> getStandardDebug();

	/**
	 * If you feel like enabling a particular level, then
	 * call this function to get all the levels that probably
	 * should also be enabled
	 */
	std::vector<Level> getAllShouldBeEnabled(Level l);

	std::pair<Level,bool> getFromString(std::string str);
	std::string getAsString(Level l);

	template<typename Func>
	void forEachLevel(Func&& f) {
		for (uint i = Level::LEVEL_START; i < Level::LEVEL_COUNT; ++i) {
			f(Level(i));
		}
	}

}

using DL = DebugLevel::Level;

/**
 * This class supplies REDIRECT_TYPE operator()(LEVEL_TYPE) and
 * methods for manipulating the enable/disable state of each level
 *
 * Classes that use this are expected to inherit from this class,
 * so that the level manipulation & operator() usage is seamless.
 * It is not necessary, however.
 *
 * Template Arguments:
 * STREAM_GET_TYPE - something that provides operator()(LevelRedirecter*)
 *     which converts it's argument to something that can be passed
 *     to the constructor of REDIRECT_TYPE
 * REDIRECT_TYPE - the return value of the operator()(LEVEL_TYPE). Must
 *     be constructable from the return value of STREAM_GET_TYPE()(LevelRedirecter*)
 * LEVEL_TYPE - the type that should be passed to operator()(LEVEL_TYPE)
 * NUM_LEVELS - the total number & one plus the maximum of the levels that will be usable
 *     The enable/disable functions will throw exceptions if called with something
 *     equal or greater to this.
 */
template<
	typename STREAM_GET_TYPE,
	typename REDIRECT_TYPE,
	typename LEVEL_TYPE,
	size_t NUM_LEVELS
>
class LevelRedirecter {
private:
	std::bitset<NUM_LEVELS> enabled_levels;

public:
	LevelRedirecter()
		: enabled_levels()
	{ }

	virtual ~LevelRedirecter() = default;

	LevelRedirecter(const LevelRedirecter&) = delete;
	LevelRedirecter& operator=(const LevelRedirecter&) = delete;

	LevelRedirecter(LevelRedirecter&&) = default;
	LevelRedirecter& operator=(LevelRedirecter&&) = default;

	REDIRECT_TYPE operator()(const LEVEL_TYPE& level) {
		if (enabled_levels.test(level)) {
			return REDIRECT_TYPE(STREAM_GET_TYPE()(this));
		} else {
			return REDIRECT_TYPE(nullptr);
		}
	}

	template<typename T>
	void enable_level(const T& level) {
		set_enable_for_level(level,true);
	}

	template<typename T>
	void disable_level(const T& level) {
		set_enable_for_level(level,false);
	}

	template<typename LOCAL_LEVEL_TYPE>
	void set_enable_for_level(
		const LOCAL_LEVEL_TYPE& level,
		bool enable,
		typename EnableIfEnum<LOCAL_LEVEL_TYPE>::type** = 0,
		typename std::enable_if<std::is_same<LOCAL_LEVEL_TYPE,LEVEL_TYPE>::value>::type* = 0
	) {
		enabled_levels.set(
			static_cast<std::underlying_type_t<LOCAL_LEVEL_TYPE>>(level),
			enable
		);
	}

	template<typename LOCAL_LEVEL_TYPE>
	void set_enable_for_level(
		const LOCAL_LEVEL_TYPE& level,
		bool enable,
		typename EnableIfIntegral<LOCAL_LEVEL_TYPE>::type* = 0
	) {
		enabled_levels.set(
			level,
			enable
		);
	}
};

/**
 * A little helper class that is returned when an indent is done
 * It will either unindent when its destructor is called, or endIndent() is called
 */
class IndentLevel {
	friend class IndentingLeveledDebugPrinter;
	friend class LevelStream;
	IndentingLeveledDebugPrinter* src;
	bool ended;
	IndentLevel(IndentingLeveledDebugPrinter* src) : src(src), ended(false) { }
public:
	void endIndent();
	~IndentLevel();

	/**
	 * Move Constructor - disable the old one, but otherwise copy everything over
	 */
	IndentLevel(IndentLevel&& src_ilevel)
		: src(std::move(src_ilevel.src))
		, ended(std::move(src_ilevel.ended))
	{
		src_ilevel.ended = true;
	}

	/// copying would cause problems - who would do the unindent?
	IndentLevel(const IndentLevel&) = delete;

	/**
	 * Move Assignment - disable the old one, but otherwise copy everything over
	 */
	IndentLevel& operator=(IndentLevel&& rhs) {
		this->src = std::move(rhs.src);
		this->ended = std::move(rhs.ended);
		rhs.ended = true;
		return *this;
	}

	/// same problems as copying
	IndentLevel& operator=(const IndentLevel&) = delete;
};

/**
 * A helper class for actually printing. IndentingLeveledDebugPrinter doesn't
 * actually have operator<< defined, so that you have to use operator() to print.
 * (it returns an object of this class)
 */
class LevelStream {
private:
	friend class IndentingLeveledDebugPrinter;

	IndentingLeveledDebugPrinter* src;
	std::stringstream underlying_ss;

public:
	LevelStream(IndentingLeveledDebugPrinter* src)
		: src(src)
		, underlying_ss()
	{ }

	LevelStream(const LevelStream&) = default;
	LevelStream(LevelStream&&) = default;

	~LevelStream();

	LevelStream& operator=(const LevelStream&) = default;
	LevelStream& operator=(LevelStream&&) = default;

	template<typename T>
	IndentLevel indentWithTitle(const T& t);

	bool enabled() { return src != nullptr; }
	void flush();

	template<typename T>
	LevelStream& push_in(const T& t) {
		if (enabled()) {
			underlying_ss << t;
		}
		return *this;
	}
};

/**
 * The class that handles most of the output control & formatting - the type of dout
 * It inherits from LevelRedirecter to define operator()(Debug::Level) and the enable/disable
 * controls. It also inherits from a filtering_ostream to allow filtering of the output.
 *
 * To print something, you must do dout(DL::*) << "string", and the same must be done for
 * an indent level.
 *
 * The highest_title_rank controls how many '=' to put around the title of the indent level at
 * the default level. Each inner level will be indented one tab stop, and have one fewer '=' on
 * each side of the title's text - with a minimum of one.
 */
class IndentingLeveledDebugPrinter
	: private boost::iostreams::filtering_ostream
	, public LevelRedirecter
		< StaticCaster<IndentingLeveledDebugPrinter*>
		, LevelStream
		, DebugLevel::Level
		, DebugLevel::LEVEL_COUNT
	>
{
	int highest_title_rank;
	int indent_level;

	bool just_saw_newline;
public:

	IndentingLeveledDebugPrinter(std::ostream& os, int highest_title_rank)
		: boost::iostreams::filtering_ostream()
		, LevelRedirecter()
		, highest_title_rank(highest_title_rank)
		, indent_level(0)
		, just_saw_newline(false)
	{
		push(os);
	}

	IndentingLeveledDebugPrinter(const IndentingLeveledDebugPrinter&) = delete;
	IndentingLeveledDebugPrinter& operator=(const IndentingLeveledDebugPrinter&) = delete;

	void print(std::istream& ss) {
		while (true) {
			auto c = ss.get();
			if (ss.eof()) { break; }

			// if we see a newline, remember it, and output spaces next time.
			if (c == '\n') {
				just_saw_newline = true;
			} else if (just_saw_newline) {
				util::repeat(getNumSpacesToIndent(),[&](){
					this->put(' ');
				});
				just_saw_newline = false;
			}

			this->put((char_type)c);
		}
		flush(); // ensure each ss is printed immediately
	}

	int getIndentLevel() {
		return indent_level;
	}

	int getNumSpacesToIndent() {
		return indent_level * 2;
	}

	int getTitleLevel() {
		if (indent_level >= highest_title_rank) {
			return 1;
		} else {
			return highest_title_rank-indent_level;
		}
	}

	void setHighestTitleRank(int level) { highest_title_rank = level; }

private:
	friend class IndentLevel;
	friend class LevelStream;

	template<typename FUNC>
	auto indentWithTitle(const FUNC& f) -> decltype(f(std::stringstream()),IndentLevel(this)) {
		// the weird return value is so the compiler SFINAE's away this
		// overload if FUNC is not a lambda style type
		std::stringstream local_ss;

		util::repeat(getTitleLevel(),[&](){
			local_ss << '=';
		});
		local_ss << ' ';

		f(local_ss);

		local_ss << ' ';
		util::repeat(getTitleLevel(),[&](){
			local_ss << '=';
		});
		local_ss << '\n';

		print(local_ss);
		indent_level++;
		return IndentLevel(this);
	}

	IndentLevel indentWithTitle(const std::string& title) {
		return indentWithTitle([&](auto&& s){ s << title; });
	}

	IndentLevel indentWithTitle(const char*& title) {
		return indentWithTitle([&](auto&& s){ s << title; });
	}

	void endIndent() {
		if (indent_level > 0) {
			indent_level = indent_level - 1;
		}
	}
};

extern IndentingLeveledDebugPrinter dout;

template<typename T>
LevelStream& operator<<(LevelStream&& lhs, const T& t) {
	return lhs << t;
}

template<typename T>
LevelStream& operator<<(LevelStream& lhs, const T& t) {
	return lhs.push_in(t);
}

// has to be down here because the type of src (IndentingLeveledDebugPrinter*)
// hadn't been defined yet, so it needed be be after it to call it's methods
template<typename T>
IndentLevel LevelStream::indentWithTitle(const T& t) {
	if (enabled()) {
		return src->indentWithTitle(t);
	} else {
		return IndentLevel(nullptr);
	}
}

namespace util {
	template<typename EXCEPTION, typename FUNC>
	[[noreturn]] void print_and_throw(FUNC func, ::DebugLevel::Level level = ::DebugLevel::Level::ERROR) {
		std::ostringstream os;
		func(os);
		dout(level) << '\n' << os.str() << '\n';
		throw EXCEPTION(os.str());
	}
}

#endif /* UTIL__LOGGING_H */
