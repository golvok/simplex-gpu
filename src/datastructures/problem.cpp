#include "problem.hpp"

#include <random>

namespace simplex {

Problem generate_random_problem(const RandomProblemSpecification& prob_spec) {
	const auto seed = prob_spec.random_seed.value_or(std::random_device()());
	auto prob_spec_with_seed = prob_spec;
	prob_spec_with_seed.random_seed = seed;
	Problem p(prob_spec_with_seed);

	std::mt19937 rgen{seed};
	std::uniform_real_distribution<> random_chance_dist          (   0.0,   1.0);
	std::uniform_real_distribution<> random_constr_coeff_dist    (prob_spec.constr_coeff_range.first,     prob_spec.constr_coeff_range.second);
	std::uniform_real_distribution<> random_constr_rhs_coeff_dist(prob_spec.constr_rhs_coeff_range.first, prob_spec.constr_rhs_coeff_range.second);
	std::uniform_real_distribution<> random_objfunc_coeff_dist   (prob_spec.objfunc_coeff_range.first,    prob_spec.objfunc_coeff_range.second);
	const auto random_chance            = [&]() { return random_chance_dist          (rgen); };
	const auto random_constr_coeff      = [&]() { return random_constr_coeff_dist    (rgen); };
	const auto random_constr_rhs_coeff  = [&]() { return random_constr_rhs_coeff_dist(rgen); };
	const auto random_objfunc_coeff     = [&]() { return random_objfunc_coeff_dist   (rgen); };

	std::vector<std::pair<VariableID, Problem::FloatType>> constraint;
	for (int iconstr = 0; iconstr < prob_spec.num_constraints; ++iconstr) {
		constraint.clear();
		for (int ivar = 0; ivar < prob_spec.num_variables; ++ivar) {
			if (random_chance() < prob_spec.density) {
				constraint.emplace_back(util::make_id<VariableID>(ivar), random_constr_coeff());
			} else {
				constraint.emplace_back(util::make_id<VariableID>(ivar), 0);
			}
		}
		p.add_constraint(constraint, random_constr_rhs_coeff());
	}

	for (int ivar = 0; ivar < prob_spec.num_variables; ++ivar) {
		const auto var = util::make_id<VariableID>(ivar);
		if (p.has_variable(var)) {
			p.set_cost(var, random_objfunc_coeff());
		}
	}

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

Problem pad_with_zeroes_modulo(Problem&& p, std::ptrdiff_t height_modulus, std::ptrdiff_t width_modulus) {
	// +1 is for the special 0th row and col
	const std::ptrdiff_t rows_to_pad = height_modulus - ((1 + p.num_constraints()) % height_modulus);
	const std::ptrdiff_t cols_to_pad =  width_modulus - ((1 + p.num_variables())   %  width_modulus);

	std::vector<std::pair<VariableID, Problem::FloatType>> empty_constraint_with_new_vars;
	for (const auto vid_and_vprop : p.variables()) {
		empty_constraint_with_new_vars.emplace_back(vid_and_vprop.first, 0);
	}

	for (std::ptrdiff_t i = 0; i < cols_to_pad; ++i) {
		empty_constraint_with_new_vars.emplace_back(util::make_id<VariableID>(p.num_variables() + i), 0);
	}

	for (std::ptrdiff_t i = 0; i < rows_to_pad; ++i) {
		p.add_constraint(empty_constraint_with_new_vars, 0);
	}

	return p;
}

} // end namespace simplex
