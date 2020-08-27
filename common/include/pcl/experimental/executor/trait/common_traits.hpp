/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#include <pcl/type_traits.h>

#include <type_traits>

namespace pcl {
namespace executor {

/**
 *   In accordance with equality_comparable concept in C++ 20
 **/

template <typename T1, typename T2, typename = void>
struct equality_comparable : std::false_type {};

template <typename T1, typename T2>
struct equality_comparable<
    T1,
    T2,
    pcl::void_t<decltype(std::declval<T1>() == std::declval<T2>(),
                         std::declval<T2>() == std::declval<T1>(),
                         std::declval<T1>() != std::declval<T2>(),
                         std::declval<T2>() != std::declval<T1>())>> : std::true_type {
};

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

template <template <typename...> class Type, typename... Args1, typename... Args2>
struct is_same_template_impl<Type<Args1...>, Type<Args2...>> : std::true_type {};

} // namespace detail

template <typename T1, typename T2>
using is_same_template =
    detail::is_same_template_impl<pcl::remove_cvref_t<T1>, pcl::remove_cvref_t<T2>>;

} // namespace executor
} // namespace pcl
