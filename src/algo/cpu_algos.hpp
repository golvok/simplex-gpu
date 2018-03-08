#ifndef ALGO__CPU_ARGOS
#define ALGO__CPU_ARGOS

#include <datastructures/problem.hpp>
#include <algo/algo_common.hpp>

#include <boost/variant.hpp>

namespace simplex{
namespace cpu {

boost::variant<
	Assignments,
	TableauErrors
> algo_from_paper(const Problem& problem);

} // end namespace cpu
} // end namespace simplex

#endif /* ALGO__CPU_ARGOS */
