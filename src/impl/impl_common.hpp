#ifndef IMPL__COMMON_INTERFACE
#define IMPL__COMMON_INTERFACE

#include <util/id.hpp>
// #include <cstdint>

#ifndef __CUDACC__
#include <iostream>
#endif

namespace simplex {

struct VariableIndexTag { static const int DEFAULT_VALUE = -1; };
typedef util::ID<int, VariableIndexTag> VariableIndex;

#ifndef __CUDACC__
inline std::ostream& operator<<(std::ostream& os, const VariableIndex& id) {
	os << "var" << id.getValue();
	return os;
}
#endif

struct VariablePair {
	VariableIndex entering;
	VariableIndex leaving;
};

} // end namespace simplex

#endif /* IMPL__COMMON_INTERFACE */
