#ifndef DATASTRUCTURES__TABLEAU_H
#define DATASTRUCTURES__TABLEAU_H

#include <vector>

namespace simplex {

template<typename FloatType>
class Tableau {
public:
	Tableau(int width, int height)
		: m_data_width(width)
		, m_data_height(height)
		, m_data(static_cast<std::size_t>(width*height))
	{ }

	const FloatType& at(int x, int y) const { return m_data[indexof(x,y)]; }
	      FloatType& at(int x, int y)       { return m_data[indexof(x,y)]; }

	int indexof(int x, int y) const { return y*m_data_width + x; }

private:
	int m_data_width;
	int m_data_height;
	std::vector<FloatType> m_data;
};

} // end namespace simplex

#endif /* DATASTRUCTURES__TABLEAU_H */
