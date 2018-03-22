#ifndef UTIL__PRIMITIVE_DATA_H
#define UTIL__PRIMITIVE_DATA_H

namespace util {

template<typename T>
class PointerAndSize {
public:
	typedef std::ptrdiff_t Index;
	PointerAndSize(T* data, Index size)
		: m_data(data)
		, m_size(size)
	{ }

	template<typename SRC>
	PointerAndSize(SRC& src)
		: m_data(src.data())
		, m_size(static_cast<Index>(src.size()))
	{ }
	
	T& at(Index i) { return m_data[i]; }
	const T& at(Index i) const { return m_data[i]; }

	Index size() const { return m_size; }

	T* data() { return m_data; }
	const T* data() const { return m_data; }

	T* m_data;
	Index m_size;
};

} // end namespace util

#endif /* UTIL__PRIMITIVE_DATA_H */
