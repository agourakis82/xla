/* Copyright 2026 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/pjrt/pjrt_build_stamp_registry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/pjrt_compiler_variant.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupCompilerSuccessful) {
  PjRtBuildStampRegistry registry;
  int factory_calls = 0;
  auto factory =
      [&]() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    factory_calls++;
    PjRtBuildStamp::Options opt;
    opt.build_timestamp = absl::FromUnixSeconds(1000);
    opt.vcs_revision = "rev1";
    opt.build_label = "label1";
    opt.source_hash = "source_hash_abc";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(registry.RegisterCompilerBuildStampFactory("cuda", "custom_variant",
                                                       factory));

  ASSERT_OK_AND_ASSIGN(
      auto stamp, registry.GetBuildStampForCompiler("cuda", "custom_variant"));
  EXPECT_EQ(stamp->source_hash(), "source_hash_abc");
  EXPECT_EQ(factory_calls, 1);

  // Subsequent lookups must be memoized without re-invoking the factory.
  ASSERT_OK_AND_ASSIGN(
      auto stamp2, registry.GetBuildStampForCompiler("cuda", "custom_variant"));
  EXPECT_EQ(stamp2->source_hash(), "source_hash_abc");
  EXPECT_EQ(factory_calls, 1);

  // ClearCacheForTesting drops memoized stamp while keeping factory registered.
  registry.ClearCacheForTesting();
  ASSERT_OK_AND_ASSIGN(
      auto stamp3, registry.GetBuildStampForCompiler("cuda", "custom_variant"));
  EXPECT_EQ(stamp3->source_hash(), "source_hash_abc");
  EXPECT_EQ(factory_calls, 2);
}

TEST(PjRtBuildStampRegistryTest, RegisterAndLookupPlatformSuccessful) {
  PjRtBuildStampRegistry registry;
  auto linked_factory =
      []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.source_hash = "linked_hash";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };
  auto forge_factory =
      []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.source_hash = "forge_hash";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(registry.RegisterPlatformBuildStampFactory("tpu", linked_factory));
  ASSERT_OK(registry.RegisterCompilerBuildStampFactory("tpu", "forge",
                                                       forge_factory));

  ASSERT_OK_AND_ASSIGN(auto default_stamp,
                       registry.GetBuildStampForPlatform("tpu"));
  EXPECT_EQ(default_stamp->source_hash(), "linked_hash");

  ASSERT_OK_AND_ASSIGN(auto linked_stamp, registry.GetBuildStampForCompiler(
                                              "tpu", kLinkedVariant));
  EXPECT_EQ(linked_stamp->source_hash(), "linked_hash");

  ASSERT_OK_AND_ASSIGN(auto forge_stamp,
                       registry.GetBuildStampForCompiler("tpu", "forge"));
  EXPECT_EQ(forge_stamp->source_hash(), "forge_hash");
}

TEST(PjRtBuildStampRegistryTest, DuplicateRegistrationFails) {
  PjRtBuildStampRegistry registry;
  auto factory = []() -> absl::StatusOr<std::shared_ptr<const PjRtBuildStamp>> {
    PjRtBuildStamp::Options opt;
    opt.source_hash = "h";
    ABSL_ASSIGN_OR_RETURN(PjRtBuildStamp stamp,
                          PjRtBuildStamp::Create(std::move(opt)));
    return std::make_shared<const PjRtBuildStamp>(std::move(stamp));
  };

  ASSERT_OK(
      registry.RegisterCompilerBuildStampFactory("tpu", "variant_a", factory));
  EXPECT_THAT(
      registry.RegisterCompilerBuildStampFactory("tpu", "variant_a", factory),
      StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(PjRtBuildStampRegistryTest, LookupNonExistentFails) {
  PjRtBuildStampRegistry registry;
  EXPECT_THAT(registry.GetBuildStampForCompiler("tpu", "unknown_variant"),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xla
