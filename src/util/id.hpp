#ifndef UTIL__ID_H
#define UTIL__ID_H

#include <util/my_hash.hpp>

#include <climits>
#include <cstdint>
#include <type_traits>

namespace util {

class IDBase {
public:
	int getValue() const;
	template<typename STREAM> void print(STREAM& os);
};

template<typename id_type, typename TAG>
class ID : public IDBase {
	id_type value;
protected:
	explicit ID(const id_type& value) : value(value) { }

	template<typename ID_TYPE, typename... ARGS>
	friend auto make_id(ARGS&&... args) -> std::enable_if_t<
		std::is_base_of<typename ID_TYPE::ThisIDType,ID_TYPE>::value,
		ID_TYPE
	>;
public:
	using IDType = id_type;
	using ThisIDType = ID<id_type,TAG>;
	constexpr static id_type DEFAULT_VALUE = TAG::DEFAULT_VALUE;
	constexpr static unsigned long long BIT_SIZE = static_cast<unsigned long long>(sizeof(IDType)*CHAR_BIT);
	constexpr static unsigned long long JUST_HIGH_BIT = static_cast<unsigned long long>(static_cast<IDType>(1) << (BIT_SIZE-1));

	ID() : value(TAG::DEFAULT_VALUE) { }
	ID(const ID&) = default;
	ID(ID&&) = default;

	ID& operator=(const ID&) = default;
	ID& operator=(ID&&) = default;

	explicit operator id_type() const { return value; }
	id_type getValue() const { return value; }
	template<typename STREAM> void print(STREAM& os) { os << value; }
};

template<typename ID_TYPE, typename... ARGS>
auto make_id(ARGS&&... args) -> std::enable_if_t<
	std::is_base_of<typename ID_TYPE::ThisIDType,ID_TYPE>::value,
	ID_TYPE
> {
	return ID_TYPE(std::forward<ARGS>(args)...);
}

template<typename ID>
class IDGenerator {
	using IDType = typename ID::IDType;

	IDType next_id_value;
public:
	IDGenerator(IDType first_value)
		: next_id_value(first_value)
	{ }

	ID gen_id() {
		ID id = make_id<ID>(next_id_value);
		++next_id_value;
		return id;
	}
};

template<typename... ARGS>
auto make_id_generator(ARGS&&... args) {
	return IDGenerator<ARGS...>(std::forward<ARGS>(args)...);
}

template<typename id_type, typename TAG>
bool operator==(const ID<id_type,TAG>& lhs, const ID<id_type,TAG>& rhs) {
	return (id_type)lhs == (id_type)rhs;
}

template<typename id_type, typename TAG>
bool operator!=(const ID<id_type,TAG>& lhs, const ID<id_type,TAG>& rhs) {
	return !(lhs == rhs);
}

template<class T, typename = void>
struct IDHasher;

template<class T>
struct IDHasher<T, std::enable_if_t<std::is_base_of<IDBase, T>::value>> {
	std::size_t operator()(const T& id) const {
		return std::hash<decltype(id.getValue())>()(id.getValue());
	}
};

template<class T>
struct MyHash<T, std::enable_if_t<std::is_base_of<IDBase, T>::value>> {
	using type = IDHasher<T>;
};

} // end namespace util

#endif // UTIL__ID_H
