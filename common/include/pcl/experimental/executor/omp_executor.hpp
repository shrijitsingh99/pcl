/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2020-, Open Perception, Inc.
 *  Author: Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <pcl/console/print.h>
#include <pcl/experimental/executor/property.h>
#include <pcl/experimental/executor/type_trait.h>
#include <pcl/types.h>

namespace pcl {
namespace executor {

template <typename Blocking, typename ProtoAllocator>
struct omp_executor;

#ifdef _OPENMP
template <>
struct is_executor_available<omp_executor> : std::true_type {};
#endif

template <typename Blocking = blocking_t::always_t,
          typename ProtoAllocator = std::allocator<void>>
struct omp_executor {
  using shape_type = uindex_t;
  using index_type = int;

  shape_type max_threads = 0;

  omp_executor() : omp_executor(0) {}

  omp_executor(shape_type threads) : max_threads(threads)
  {
    set_max_threads(threads);
  }

  bool
  set_max_threads(shape_type threads)
  {
#ifdef _OPENMP
    max_threads = threads ? threads : omp_get_max_threads();
    return true;
#endif
    return false;
  }

  template <typename Executor, InstanceOf<Executor, omp_executor> = 0>
  friend constexpr bool
  operator==(const omp_executor&, const Executor&) noexcept
  {
    return std::is_same<omp_executor, Executor>::value;
  }

  template <typename Executor, InstanceOf<Executor, omp_executor> = 0>
  friend constexpr bool
  operator!=(const omp_executor& lhs, const Executor& rhs) noexcept
  {
    return !operator==(lhs, rhs);
  }

  template <typename F>
  void
  execute(F&& f) const
  {
    static_assert(is_executor_available_v<omp_executor>, "OpenMP executor unavailable");
    f();
  }

  template <typename F>
  void
  bulk_execute(F&& f, const shape_type& n) const
  {
    static_assert(is_executor_available_v<omp_executor>, "OpenMP executor unavailable");
    pcl::utils::ignore(f, n);
#ifdef _OPENMP
#pragma omp parallel num_threads(max_threads)
    {
#pragma omp for nowait
      for (index_type index = 0; index < n; ++index)
        f(index);
    }
#endif
  }

  static constexpr decltype(auto)
  query(const blocking_t&) noexcept
  {
    return blocking_t::always;
  }

  omp_executor<blocking_t::always_t, ProtoAllocator>
  require(const blocking_t::always_t&) const
  {
    return {};
  }
};

using default_omp_executor = omp_executor<>;

} // namespace executor
} // namespace pcl
