/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *
 */

#pragma once

#include <pcl/type_traits.h>

#include <type_traits>
#include <tuple>

namespace pcl {

// for_each_tuple_until
// Iterate over tuple
// https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11
template <typename TupleType, typename FunctionType>
void
for_each_until_true(
    TupleType&&,
    FunctionType,
    std::integral_constant<
        std::size_t,
        std::tuple_size<typename std::remove_reference<TupleType>::type>::value>)
{}

template <std::size_t I,
          typename TupleType,
          typename FunctionType,
          typename = typename std::enable_if<
              I != std::tuple_size<
                       typename std::remove_reference<TupleType>::type>::value>::type>
void
for_each_until_true(TupleType&& t, FunctionType f, std::integral_constant<size_t, I>)
{
  bool exit = f(std::get<I>(std::forward<TupleType>(t)));

  if (!exit)
    for_each_until_true(
        std::forward<TupleType>(t), f, std::integral_constant<size_t, I + 1>());
}

template <typename TupleType, typename FunctionType>
void
for_each_until_true(TupleType&& t, FunctionType f)
{
  for_each_until_true(
      std::forward<TupleType>(t), f, std::integral_constant<size_t, 0>());
}

// tuple_contains_type
namespace detail {

template <typename T, typename Tuple>
struct tuple_contains_type_impl;

template <typename T, typename... Us>
struct tuple_contains_type_impl<T, std::tuple<Us...>>
: pcl::disjunction<std::is_same<T, Us>...> {};

} // namespace detail

template <typename T, typename Tuple>
using tuple_contains_type = typename detail::tuple_contains_type_impl<T, Tuple>::type;

// filter_tuple_values
namespace detail {

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values_impl {
  using type = decltype(std::tuple_cat(
      typename std::conditional<predicate<T>::value, std::tuple<T>, std::tuple<>>::
          type()...));

  auto
  operator()(const std::tuple<T...>& in)
  {
    return (*this)(in, type{});
  }

private:
  // neat utility function to fetch the types we're interest in outputting
  template <typename... To>
  auto
  operator()(const std::tuple<T...>& in, std::tuple<To...>)
  {
    return std::make_tuple(std::get<To>(in)...);
  }
};

} // namespace detail

template <template <typename...> class predicate, typename T>
struct filter_tuple_values;

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values<predicate, std::tuple<T...>>
: detail::filter_tuple_values_impl<predicate, T...> {};

template <template <typename...> class predicate, typename... T>
struct filter_tuple_values<predicate, const std::tuple<T...>>
: detail::filter_tuple_values_impl<predicate, T...> {};

} // namespace pcl
