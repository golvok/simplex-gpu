#ifndef IMPL__COMMON_INTERFACE
#define IMPL__COMMON_INTERFACE

#include <util/id.hpp>

#include <cstdint>

namespace simplex {

struct VariableIDTag { static const std::int32_t DEFAULT_VALUE = INT32_MIN; };
using VariableID = util::ID<std::int32_t, VariableIDTag>;

struct VariablePair {
	VariableID entering;
	VariableID leaving;
};

} // end namespace simplex

#endif /* IMPL__COMMON_INTERFACE */
