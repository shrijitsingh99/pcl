/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2014-, Open Perception, Inc.
 *  Author(s): Shrijit Singh <shrijitsingh99@gmail.com>
 *
 */

#include <pcl/experimental/executor/executor.h>
#include <pcl/test/gtest.h>

using namespace pcl;
using namespace executor;

using ExecutorAlwaysAvailableTypes = ::testing::Types<const default_inline_executor,
                                                      default_inline_executor,
                                                      const default_sse_executor,
                                                      default_sse_executor>;

using ExecutorCondAvailableTypes = ::testing::Types<const default_inline_executor,
                                                    default_inline_executor,
                                                    const default_sse_executor,
                                                    default_sse_executor,
                                                    const default_omp_executor,
                                                    default_omp_executor>;

using ExecutorTypes = std::conditional_t<is_executor_available_v<omp_executor>,
                                       ExecutorCondAvailableTypes,
                                       ExecutorAlwaysAvailableTypes>;

template <typename Executor>
class ExecutorValidity : public ::testing::Test {};

TYPED_TEST_SUITE(ExecutorValidity, ExecutorTypes);

TYPED_TEST(ExecutorValidity, executors)
{
  EXPECT_TRUE(is_executor_v<TypeParam>);
  EXPECT_FALSE(is_executor_v<int>);
}

template <typename Executor>
class ExecutorPropertyTraits : public ::testing::Test {};

TYPED_TEST_SUITE(ExecutorPropertyTraits, ExecutorTypes);

TYPED_TEST(ExecutorPropertyTraits, executors)
{
  EXPECT_TRUE((can_require_v<TypeParam, blocking_t::always_t>));
  EXPECT_FALSE((can_require_v<TypeParam, blocking_t::never_t>));

  EXPECT_TRUE((can_prefer_v<TypeParam, blocking_t::always_t>));
  EXPECT_TRUE((can_prefer_v<TypeParam, blocking_t::never_t>));

  EXPECT_TRUE((can_query_v<TypeParam, blocking_t::always_t>));
  EXPECT_TRUE((can_query_v<TypeParam, blocking_t::never_t>));
}

template <typename Executor>
class ExecutorProperties : public ::testing::Test {};

TYPED_TEST_SUITE(ExecutorProperties, ExecutorTypes);

TYPED_TEST(ExecutorProperties, executors)
{
  TypeParam exec;

  EXPECT_EQ(exec.query(blocking), blocking_t::always);
  const auto new_exec1 = exec.require(blocking_t::always);
  EXPECT_EQ(new_exec1.query(blocking), blocking_t::always);

  EXPECT_EQ(query(exec, blocking_t::always), blocking_t::always);
  const auto new_exec2 = require(exec, blocking_t::always);
  EXPECT_EQ(query(new_exec2, blocking), blocking_t::always);
}

template <typename Executor>
class ExecutorExecute : public ::testing::Test {};

TYPED_TEST_SUITE(ExecutorExecute, ExecutorTypes);

TYPED_TEST(ExecutorExecute, executors)
{
  TypeParam exec;
  const int a = 1, b = 2;

  int c = 0;
  exec.execute([&]() { c = a + b; });
  EXPECT_EQ(c, 3);

  std::array<int, 3> c_vec = {0};
  exec.bulk_execute(
      [&c_vec](auto) {
        for (auto& val : c_vec)
          val = 1;
      },
      3);
  EXPECT_EQ(c_vec[0] + c_vec[1] + c_vec[2], c_vec.size());
}

template <typename Executor>
class ExecutorInstanceOfAny : public ::testing::Test {
protected:
  template <typename Blocking = executor::blocking_t::always_t>
  struct derived_inline : executor::inline_executor<Blocking> {};
};

TYPED_TEST_SUITE(ExecutorInstanceOfAny, ExecutorTypes);

TYPED_TEST(ExecutorInstanceOfAny, executors)
{
  using DerivedExecutor = typename TestFixture::template derived_inline<>;

  EXPECT_TRUE((is_instance_of_any_v<default_inline_executor, inline_executor>));
  EXPECT_TRUE((is_instance_of_any_v<DerivedExecutor, inline_executor>));
}

int
main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return (RUN_ALL_TESTS());
}
