#ifndef UTIL_H
#define UTIL_H

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include <iostream>

template <typename...> struct all_convertible;

template <> struct all_convertible<> : std::true_type { };

template <typename T> struct all_convertible<T> : std::true_type { };

template <typename T, typename V, typename... Rest> struct all_convertible<T, V, Rest...>
: std::integral_constant<
	bool,
	std::is_convertible<V, T>::value && all_convertible<T,Rest...>::value
> { };

namespace util {
	template<typename T>
	void reverse(T& t) {
		using std::begin; using std::end;
		reverse(begin(t),end(t));
	}

	template<typename FUNC>
	void repeat(int count, const FUNC& f) {
		for (int i = 1; i <= count; ++i) {
			f();
		}
	}

	template<typename T>
	std::shared_ptr<T> make_shared(T&& t) {
		return std::make_shared<
			typename std::remove_cv<
				typename std::remove_reference<T>::type
			>::type
		>(
			std::forward<T>(t)
		);
	}

	template<typename T>
	T make_copy(const T& t) {
		return T(t);
	}

	template<typename CONTAINER, typename PRED>
	void remove_if_assoc(CONTAINER& c, PRED&& p) {
		using std::begin; using std::end;
		for(auto it = begin(c); it != end(c); ) {
			if (p(*it)) {
				it = c.erase(it);
			} else {
				++it;
			}
		}
	}

	template<typename CONTAINER>
	bool empty(CONTAINER&& c) {
		using std::begin;
		using std::end;
		return begin(c) == end(c);
	}
}

/*******
 * Begin definition of ComparesWithTag
 *******/

template<typename T, typename U>
class ComparesWithTag {
	const T& thing;
	U identifier;

public:
	ComparesWithTag(const T& thing, U identifier)
		: thing(thing)
		, identifier(identifier)
	{ }
	ComparesWithTag(const ComparesWithTag&) = default;
	ComparesWithTag& operator=(const ComparesWithTag&) = delete;
	ComparesWithTag& operator=(ComparesWithTag&&) = default;

	operator std::tuple<T,U>() const {
		return {thing,identifier};
	}

	template<typename THEIR_T, typename THEIR_U>
	bool operator<(const ComparesWithTag<THEIR_T,THEIR_U> rhs) const {
		return thing < rhs.thing;
	}

	const T& value() const { return thing; }
	U id() const { return identifier; }
};


/**
 * Allows you to tag some data along when you call std::max or something.
 * Defines operator< using the first argumen's operator<
 *
 * Example:
 *    enum class MyEnum {
 *        A,B,
 *    };
 *
 *    auto result = std::max(
 *        compare_with_tag(getA(),MyEnum::A),
 *        compare_with_tag(getB(),MyEnum::B)
 *    );
 *
 *    if (result.id() == MyEnum::A) {
 *        std::cout << "A was smaller, A = " << result << '\n';
 *    } else if (result.id() == MyEnum::B) {
 *        std::cout << "B was smaller, B = " << result << '\n';
 *    }
 */
template<typename T, typename U>
ComparesWithTag<T,U> compare_with_tag(const T& thing, U id) {
	return ComparesWithTag<T,U>(thing,id);
}

template<typename T, typename V = void>
using EnableIfEnum = typename std::enable_if<std::is_enum<T>::value,V>;

template<typename T, typename V = void>
using EnableIfIntegral = typename std::enable_if<std::is_integral<T>::value,V>;

template<typename CAST_TO>
struct StaticCaster {
	template<typename SRC>
	CAST_TO operator()(SRC src) const {
		return static_cast<CAST_TO>(src);
	}
};

namespace util {

template<typename T>
std::string stringify_through_stream(const T& t) {
	std::ostringstream stream;
	stream << t;
	return stream.str();
}

template<typename RETVAL_TYPE, typename ITER_TYPE>
class DerefAndIncrementer {
	ITER_TYPE iter;
	size_t i;
public:
	DerefAndIncrementer(ITER_TYPE beg) : iter(beg), i(0) { }
	template<typename ARG>
	RETVAL_TYPE operator()(const ARG&) {
		RETVAL_TYPE result = *iter;
		std::cout << result << " - " << i << '\n';
		++iter;
		++i;
		return result;
	}
};

/**
 * Retuns a lambda stlye object that will return the value of
 * *iter the first time it is called, then *std::next(iter), etc.
 */
template<typename RETVAL_TYPE, typename ITER_TYPE>
DerefAndIncrementer<RETVAL_TYPE,ITER_TYPE> make_deref_and_incrementer(const ITER_TYPE& iter) {
	return DerefAndIncrementer<RETVAL_TYPE,ITER_TYPE>(iter);
}

template<class InputIt, class UnaryPredicate>
std::pair<InputIt,size_t> find_by_index(InputIt first, InputIt last, UnaryPredicate p) {
	size_t index = 0;
	for (; first != last; ++first, ++index) {
		if (p(index)) {
			return { first, index };
		}
	}
	return { last, index };
}

template<class InputIt, class BinaryPredicate>
std::pair<InputIt,size_t> find_with_index(InputIt first, InputIt last, BinaryPredicate p) {
	size_t index = 0;
	for (; first != last; ++first, ++index) {
		if (p(*first,index)) {
			return { first, index };
		}
	}
	return { last, index };
}

template<class ForwardIt, class UnaryPredicate>
ForwardIt remove_by_index(ForwardIt first, ForwardIt last, UnaryPredicate p) {
	size_t index = 0;
	std::tie(first, index) = find_by_index(first, last, p);
	if (first != last) {
		for(ForwardIt i = first; i != last; ++i, ++index) {
			if (!p(index)) {
				*first = std::move(*i);
				++first;
			}
		}
	}
	return first;
}

namespace detail {
	struct printer {
	template<typename STREAM, typename T>
		void operator()(STREAM& os, const T& t) const {
			os << t;
		}
	};
}

template<typename CONTAINER, typename OSTREAM, typename FUNC = ::util::detail::printer>
void print_container(
	OSTREAM&& os,
	const CONTAINER& c,
	const std::string& sep,
	const std::string& prefix_str,
	const std::string& suffix_str,
	FUNC func = FUNC{}
) {
	using std::begin; using std::end;
	auto beg = begin(c);
	auto en = end(c);

	os << prefix_str;
	if (beg != en) {
		func(os,*beg);
		std::for_each(std::next(beg), en, [&](const decltype(*std::begin(c))& v){
			os << sep;
			func(os,v);
		});
	}
	os << suffix_str;
}

template<typename CONTAINER, typename OSTREAM, typename FUNC = ::util::detail::printer>
void print_container(
	OSTREAM&& os,
	const CONTAINER& c,
	FUNC func = FUNC{}
) {
	print_container(os, c, ", ", "{ ", " }", func);
}

template<typename CONTAINER, typename OSTREAM, typename KEY_FUNC = ::util::detail::printer, typename VALUE_FUNC = ::util::detail::printer>
void print_assoc_container(
	OSTREAM&& os,
	const CONTAINER& c,
	const std::string& sep,
	const std::string& prefix_str,
	const std::string& suffix_str,
	KEY_FUNC func_value = KEY_FUNC{},
	VALUE_FUNC func_key = VALUE_FUNC{}
) {
	using std::begin; using std::end;
	auto beg = begin(c);
	auto en = end(c);

	os << prefix_str;
	if (beg != en) {
		os << prefix_str;
		func_key(os,beg->first);
		os << " -> ";
		func_value(os,beg->second);
		os << suffix_str;
		std::for_each(std::next(beg), en, [&](const decltype(*std::begin(c))& v){
			os << sep;
			os << prefix_str;
			func_key(os,v.first);
			os << " -> ";
			func_value(os,v.second);
			os << suffix_str;
		});
	}
	os << suffix_str;
}

template<typename CONTAINER, typename OSTREAM, typename KEY_FUNC = ::util::detail::printer, typename VALUE_FUNC = ::util::detail::printer>
void print_assoc_container(
	OSTREAM&& os,
	const CONTAINER& c,
	KEY_FUNC func_value = KEY_FUNC{},
	VALUE_FUNC func_key = VALUE_FUNC{}
) {
	print_assoc_container(os, c, ", ", "{ ", " }", func_value, func_key);
}

} // end namespace util

#endif /* UTIL_H */
