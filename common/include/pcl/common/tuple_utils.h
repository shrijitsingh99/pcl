/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *
 */

#pragma once

#include <pcl/type_traits.h>

#include <type_traits>
#include <tuple>

namespace pcl {

template <typename Tuple, typename Function>
void
for_each_until_true(
    Tuple&&,
    Function,
    std::integral_constant<
        std::size_t,
        std::tuple_size<typename std::remove_reference<Tuple>::type>::value>)
{}

template <
    std::size_t I,
    typename Tuple,
    typename Function,
    typename = typename std::enable_if<
        I != std::tuple_size<typename std::remove_reference<Tuple>::type>::value>::type>
void
for_each_until_true(Tuple&& t, Function f, std::integral_constant<size_t, I>)
{
  bool exit = f(std::get<I>(std::forward<Tuple>(t)));

  if (!exit)
    for_each_until_true(
        std::forward<Tuple>(t), f, std::integral_constant<size_t, I + 1>());
}

/**
 * \brief Iterates over all elements of tuples until the function called returns true
 *
 * \tparam Tuple The tuple to iterate through
 * \tparam Function A callable that is invoked for every tuple element and returns a
 * boolean indicating whether to coninute iteration or not
 *
 * \remark Implementation taken from
 * https://stackoverflow.com/questions/26902633/how-to-iterate-over-a-stdtuple-in-c-11
 */
template <typename Tuple, typename Function>
void
for_each_until_true(Tuple&& t, Function f)
{
  for_each_until_true(std::forward<Tuple>(t), f, std::integral_constant<size_t, 0>());
}

namespace detail {

template <typename T, typename Tuple>
struct tuple_contains_type_impl;

template <typename T, typename... Us>
struct tuple_contains_type_impl<T, std::tuple<Us...>>
: pcl::disjunction<std::is_same<T, Us>...> {};

} // namespace detail

/**
 * \brief Checks whether a tuple contains a specified type
 *
 * \tparam T a type to check for
 * \tparam Tuple a tuple in which to check for the type
 *
 */
template <typename T, typename Tuple>
using tuple_contains_type = typename detail::tuple_contains_type_impl<T, Tuple>::type;

namespace detail {

template <template <typename...> class Predicate, typename... TupleElements>
struct filter_tuple_values_impl {
  using type = decltype(std::tuple_cat(
      typename std::conditional<Predicate<TupleElements>::value, std::tuple<TupleElements>, std::tuple<>>::
          type()...));

  auto
  operator()(const std::tuple<TupleElements...>& in)
  {
    return (*this)(in, type{});
  }

private:
  // Utility function to fetch the types we're interest in outputting
  template <typename... To>
  auto
  operator()(const std::tuple<TupleElements...>& in, std::tuple<To...>)
  {
    return std::make_tuple(std::get<To>(in)...);
  }
};

} // namespace detail

/**
 * \brief Filters elements of a tuple based on the predicate/condition
 *
 * \tparam Predicate A trait which takes a tuple element as parameter and defines a
 * boolean member value which dictates whether to filter the tuple element or not
 * \tparam Tuple a tuple to filter
 *
 */
template <template <typename...> class Predicate, typename Tuple>
struct filter_tuple_values;

template <template <typename...> class Predicate, typename... TupleElements>
struct filter_tuple_values<Predicate, std::tuple<TupleElements...>>
: detail::filter_tuple_values_impl<Predicate, TupleElements...> {};

template <template <typename...> class Predicate, typename... TupleElements>
struct filter_tuple_values<Predicate, const std::tuple<TupleElements...>>
: detail::filter_tuple_values_impl<Predicate, TupleElements...> {};

} // namespace pcl
