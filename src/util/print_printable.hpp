
#ifndef UTIL__PRINT_PRINTABLE_HPP
#define UTIL__PRINT_PRINTABLE_HPP

// #include <iosfwd>
// #include <tuple>

namespace util {

// TODO: maybe provide some sort of default? maybe one that gives a warning when compiled?
struct print_printable { };


#if __cplusplus >= 201103L

template<typename T, typename STREAM>
auto operator<<(STREAM& os, const T& t) -> decltype(static_cast<const print_printable*>(&t),os) {
	t.print(os);
	return os;
}

template<typename T, typename STREAM>
auto operator<<(STREAM& os, const T& t) -> decltype(static_cast<const print_printable*>(&t.get()),os) {
	t.get().print(os);
	return os;
}

template<typename T>
struct print_with_printable{
	using print_with_type = T;
};

#endif

// template<typename T, typename U>
// auto operator<<(std::ostream& os, const std::tuple<T&,U&>& pair) -> decltype(static_cast<const print_with_printable<U>*>(&std::get<0>(pair)),os) {
// 	std::get<0>(pair).print(os, std::get<1>(pair));
// 	return os;
// }

// template<typename T, typename U>
// auto operator<<(std::ostream& os, const std::tuple<T&,U&>& pair) -> decltype(static_cast<const print_with_printable<U>*>(&std::get<0>(pair).get()),os) {
// 	std::get<0>(pair).get().print(os, std::get<1>(pair));
// 	return os;
// }

} // end namespace util

#endif /* UTIL__PRINT_PRINTABLE_HPP */
