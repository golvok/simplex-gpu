#ifndef DATASTRUCTURES__PROBLEM_H
#define DATASTRUCTURES__PROBLEM_H

#include <util/id.hpp>
#include <util/my_hash.hpp>

#include <unordered_map>
#include <vector>

#include <boost/optional.hpp>

namespace simplex {

struct VariableIDTag { static const int DEFAULT_VALUE = -1; };
typedef util::ID<int, VariableIDTag> VariableID;

class Problem {
public:
	using FloatType = double;

	std::ptrdiff_t num_variables() const { if(m_max_var_id) return m_max_var_id->getValue() + 1; else return 0; }
	std::ptrdiff_t num_constraints() const { return static_cast<std::ptrdiff_t>(m_constraints.size()); }

	auto& constraints() const { return m_constraints; }
	auto& variables()   const { return m_variables; }

	template<typename CONSTRAINTS>
	void add_constraint(CONSTRAINTS&& c, FloatType rhs) {
		m_constraints.emplace_back(rhs);
		auto& new_constr = m_constraints.back();
		for (const auto& r : c) {
			m_variables[r.first];
			m_max_var_id = m_max_var_id.value_or(r.first);
			m_max_var_id = r.first.getValue() > m_max_var_id->getValue() ? r.first : m_max_var_id;
			new_constr.m_coeffs.emplace_back(r.first, r.second);
		}
	}

	void set_cost(VariableID var, FloatType coeff) {
		m_variables.at(var).m_coeff = coeff;
	}

	bool has_variable(VariableID var) {
		return m_variables.find(var) != end(m_variables);
	}

private:
	struct VariableProperties {
		struct hasher { std::size_t operator()(const VariableProperties& vp) const { return std::hash<FloatType>()(vp.m_coeff); } };
		FloatType m_coeff = 0;
	};

	struct Constraint {
		Constraint(FloatType rhs)
			: m_coeffs()
			, m_rhs(rhs)
		{ }

		std::vector<std::pair<VariableID, FloatType>> m_coeffs;
		FloatType m_rhs;
	};

	std::unordered_map<VariableID, VariableProperties, typename util::MyHash<VariableID>::type> m_variables;
	std::vector<Constraint> m_constraints;
	boost::optional<VariableID> m_max_var_id;
};

class Assignments {

};

class RandomProblemSpecification {
public:
	RandomProblemSpecification(int num_variables, int num_constraints)
		: num_variables(num_variables)
		, num_constraints(num_constraints)
	{ }

	int num_variables;
	int num_constraints;

	Problem::FloatType density = 0.8;
	std::pair<Problem::FloatType, Problem::FloatType> constr_coeff_range     = {-100.0, 100.0};
	std::pair<Problem::FloatType, Problem::FloatType> constr_rhs_coeff_range = {-100.0, 100.0};
	std::pair<Problem::FloatType, Problem::FloatType> objfunc_coeff_range    = {-100.0, 100.0};
};

Problem generate_random_problem(const RandomProblemSpecification& prob_spec);

Problem make_small_sample_problem();

} // end namespace simplex

#endif /* DATASTRUCTURES__PROBLEM_H */
