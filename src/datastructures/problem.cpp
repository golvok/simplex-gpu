#include "problem.hpp"

namespace simplex {

Problem generate_random_problem(int num_variables, int num_constraints, double density) {
	Problem p;

	return p;
}

Problem make_small_sample_problem() {
	Problem p;

	const auto v0 = util::make_id<VariableID>(0);
	const auto v1 = util::make_id<VariableID>(1);
	const auto v2 = util::make_id<VariableID>(2);
	p.add_constraint(
		std::vector<std::pair<VariableID, Problem::FloatType>>{
			{ v0, 1 },
			{ v1, 3 },
			{ v2, 2 },
		},
		10
	);

	p.add_constraint(
		std::vector<std::pair<VariableID, Problem::FloatType>>{
			{ v0, 3 },
			{ v1, 2 },
			{ v2, 5 },
		},
		15
	);

	p.set_cost(v0, 4);
	p.set_cost(v1, 2);
	p.set_cost(v2, 3);

	return p;
}

} // end namespace simplex
