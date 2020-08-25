/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <tuple>
#include <type_traits>
#include <pcl/type_traits.h>

namespace executor {

/**
 *   In accordance with equality_comparable concept in C++ 20
 **/

template <typename T1, typename T2, typename = void>
struct equality_comparable : std::false_type {};

template <typename T1, typename T2>
struct equality_comparable<
    T1, T2,
    pcl::void_t<decltype(std::declval<T1>() == std::declval<T2>(),
                              std::declval<T2>() == std::declval<T1>(),
                              std::declval<T1>() != std::declval<T2>(),
                              std::declval<T2>() != std::declval<T1>())>>
    : std::true_type {};

template <typename T1, typename T2>
constexpr bool equality_comparable_v = equality_comparable<T1, T2>::value;

/**
 *   Custom Traits
 **/

// is_instance_of_any
template <typename Executor, template <typename...> class Type, typename = void>
struct is_instance_of : std::false_type {};

template <template <typename...> class Executor,
          template <typename...>
          class Type,
          typename... Args>
struct is_instance_of<
    Executor<Args...>,
    Type,
    std::enable_if_t<std::is_base_of<Type<Args...>, Executor<Args...>>::value>>
: std::true_type {};

template <typename Executor, template <typename...> class Type>
constexpr bool is_instance_of_v = is_instance_of<Executor, Type>::value;

template <typename Executor, template <typename...> class Type>
using InstanceOf = std::enable_if_t<is_instance_of_v<Executor, Type>, int>;

template <typename Executor, template <typename...> class... Type>
using is_instance_of_any = pcl::disjunction<is_instance_of<Executor, Type>...>;

template <typename Executor, template <typename...> class... Type>
constexpr bool is_instance_of_any_v = is_instance_of_any<Executor, Type...>::value;

template <typename Executor, template <typename...> class... Type>
using InstanceOfAny = std::enable_if_t<is_instance_of_any_v<Executor, Type...>, int>;

// is_same_template
namespace detail {

template <typename T1, typename T2>
struct is_same_template_impl : std::false_type {};

template <template <typename...> class Type, typename... Args1,
          typename... Args2>
struct is_same_template_impl<Type<Args1...>, Type<Args2...>> : std::true_type {
};

}  // namespace detail

template <typename T1, typename T2>
using is_same_template =
    detail::is_same_template_impl<pcl::remove_cvref_t<T1>,
                                  pcl::remove_cvref_t<T2>>;

// for_each_tuple_until
// Iterate over tuple
// https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11
template <typename TupleType, typename FunctionType>
void for_each_until_true(
    TupleType&&, FunctionType,
    std::integral_constant<std::size_t,
                           std::tuple_size<typename std::remove_reference<
                               TupleType>::type>::value>) {}

template <std::size_t I, typename TupleType, typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<typename std::remove_reference<
                       TupleType>::type>::value>::type>
void for_each_until_true(TupleType&& t, FunctionType f,
                          std::integral_constant<size_t, I>) {
  bool exit = f(std::get<I>(std::forward<TupleType>(t)));

  if (!exit)
    for_each_until_true(std::forward<TupleType>(t), f,
                         std::integral_constant<size_t, I + 1>());
}

template <typename TupleType, typename FunctionType>
void for_each_until_true(TupleType&& t, FunctionType f) {
  for_each_until_true(std::forward<TupleType>(t), f,
                       std::integral_constant<size_t, 0>());
}


// tuple_contains_type
namespace detail {

template <typename T, typename Tuple>
struct tuple_contains_type_impl;

template <typename T, typename... Us>
struct tuple_contains_type_impl<T, std::tuple<Us...>>
    : pcl::disjunction<std::is_same<T, Us>...> {};

}  // namespace detail

template <typename T, typename Tuple>
using tuple_contains_type = typename detail::tuple_contains_type_impl<T, Tuple>::type;


// filter_tuple_values
namespace detail {

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values_impl {
  using type = decltype(
  std::tuple_cat(typename std::conditional<predicate<T>::value, std::tuple<T>,
                                           std::tuple<>>::type()...));

  auto operator()(const std::tuple<T...> &in) { return (*this)(in, type{}); }

 private:
  // neat utility function to fetch the types we're interest in outputting
  template<typename... To>
  auto operator()(const std::tuple<T...> &in, std::tuple<To...>) {
    return std::make_tuple(std::get<To>(in)...);
  }
};

}

template <template <typename...> class predicate, typename T>
struct filter_tuple_values;

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values<predicate, std::tuple<T...>>
    : detail::filter_tuple_values_impl<predicate, T...> {};

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values<predicate, const std::tuple<T...>>
    : detail::filter_tuple_values_impl<predicate, T...> {};

}  // namespace executor
