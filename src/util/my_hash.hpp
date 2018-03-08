#ifndef UTIL__MY_HASH_H
#define UTIL__MY_HASH_H

#include <functional>

namespace util {

template<class T, typename = void>
struct MyHash {
	using type = std::hash<T>;
};

template<class T>
using MyHash_t = typename MyHash<T>::type;

template<template <typename... ARGS> class CONTAINER, typename KEY, typename... REST>
using with_my_hash_t = CONTAINER<KEY, REST..., MyHash_t<KEY>>;

} // end namespace util

#endif // UTIL__MY_HASH_H
