#ifndef IMPL__COMMON_INTERFACE
#define IMPL__COMMON_INTERFACE

#include <util/id.hpp>
// #include <cstdint>

#ifndef __CUDACC__
#include <iostream>
#endif

namespace simplex {

struct VariableIDTag { static const int DEFAULT_VALUE = -1; };
typedef util::ID<int, VariableIDTag> VariableID;

#ifndef __CUDACC__
inline std::ostream& operator<<(std::ostream& os, const VariableID& id) {
	os << "var" << id.getValue();
	return os;
}
#endif

struct VariablePair {
	VariableID entering;
	VariableID leaving;
};

} // end namespace simplex

#endif /* IMPL__COMMON_INTERFACE */
