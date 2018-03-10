#ifndef IMPL__COMMON_INTERFACE
#define IMPL__COMMON_INTERFACE

#include <util/id.hpp>

#include <cstdint>
#include <iostream>

namespace simplex {

struct VariableIDTag { static const std::int32_t DEFAULT_VALUE = INT32_MIN; };
using VariableID = util::ID<std::int32_t, VariableIDTag>;

inline std::ostream& operator<<(std::ostream& os, const VariableID& id) {
	os << "var" << id.getValue();
	return os;
}

struct VariablePair {
	VariableID entering;
	VariableID leaving;
};

} // end namespace simplex

#endif /* IMPL__COMMON_INTERFACE */
