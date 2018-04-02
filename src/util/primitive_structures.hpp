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
	std::ptrdiff_t data_size() const { return size()*(std::ptrdiff_t)sizeof(T); }

	T*& data() { return m_data; }

	T* begin() { return m_data; }
	T* end() { return m_data + m_size; }

	T const* begin() const { return m_data; }
	T const* end() const { return m_data + m_size; }

	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wreturn-stack-address"
	// the following incorrectly gives a warning (known bug in clang 3.8)
	const T* const& data() const { return m_data; }
	#pragma clang diagnostic pop

	T* m_data;
	Index m_size;
};

} // end namespace util

#endif /* UTIL__PRIMITIVE_DATA_H */
