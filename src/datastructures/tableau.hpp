#ifndef DATASTRUCTURES__TABLEAU_H
#define DATASTRUCTURES__TABLEAU_H

#include <impl/impl_common.hpp>
#include <util/print_printable.hpp>

namespace simplex {

template<typename FloatType>
class Tableau : public util::print_printable {
public:
	Tableau(FloatType* data, std::ptrdiff_t height, std::ptrdiff_t width)
		: m_data_width(width)
		, m_data_height(height)
		, m_width(width)
		, m_height(height)
		, m_data(data)
	{ }

	const FloatType& at(std::ptrdiff_t row, std::ptrdiff_t col) const { return m_data[indexof(row,col)]; }
	      FloatType& at(std::ptrdiff_t row, std::ptrdiff_t col)       { return m_data[indexof(row,col)]; }

	const FloatType& at(std::ptrdiff_t row, VariableIndex col) const { return at(row, col.getValue()); }
	      FloatType& at(std::ptrdiff_t row, VariableIndex col)       { return at(row, col.getValue()); }

	const FloatType& at(VariableIndex row, std::ptrdiff_t col) const { return at(row.getValue(), col); }
	      FloatType& at(VariableIndex row, std::ptrdiff_t col)       { return at(row.getValue(), col); }

	const FloatType& at(VariableIndex row, VariableIndex col) const { return at(row.getValue(), col.getValue()); }
	      FloatType& at(VariableIndex row, VariableIndex col)       { return at(row.getValue(), col.getValue()); }

	const FloatType& cost() const { return at(0,0); }

	std::size_t indexof(std::ptrdiff_t row, std::ptrdiff_t col) const { return static_cast<std::size_t>(row*m_data_width + col); }

	std::ptrdiff_t width() const { return m_width; }
	std::ptrdiff_t height() const { return m_height; }

	FloatType*& data() { return m_data; }
	FloatType* const& data() const { return m_data; }
	std::ptrdiff_t data_size() const { return m_data_width*m_data_height*(std::ptrdiff_t)sizeof(FloatType); }

	template<typename STREAM>
	void print(STREAM& os) const {
		for (std::ptrdiff_t irow = 0; irow < height(); ++irow) {
			for (std::ptrdiff_t icol = 0; icol < width(); ++icol) {
				os << ' ' << at(irow, icol);
			}
			if (irow != height() - 1) {
				os << '\n';
			}
		}
	}

private:
	std::ptrdiff_t m_data_width;
	std::ptrdiff_t m_data_height;
	std::ptrdiff_t m_width;
	std::ptrdiff_t m_height;
	FloatType* m_data;
};

} // end namespace simplex

#endif /* DATASTRUCTURES__TABLEAU_H */
