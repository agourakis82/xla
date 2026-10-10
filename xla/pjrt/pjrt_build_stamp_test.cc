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

#include "xla/pjrt/pjrt_build_stamp.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Optional;

TEST(PjRtBuildStampTest, BasicAccessors) {
  const absl::Time now = absl::FromUnixSeconds(1700000000);
  PjRtBuildStamp::Options options;
  options.build_timestamp = now;
  options.vcs_revision = "git_rev_123";
  options.build_label = "target_foo";
  options.source_hash = "hash_abcdef01";
  options.mosaic_version = 12;

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(options)));

  EXPECT_EQ(stamp.build_timestamp(), now);
  EXPECT_EQ(stamp.vcs_revision(), "git_rev_123");
  EXPECT_EQ(stamp.build_label(), "target_foo");
  EXPECT_EQ(stamp.source_hash(), "hash_abcdef01");
  EXPECT_THAT(stamp.mosaic_version(), Optional(int64_t{12}));
  EXPECT_TRUE(stamp.components().empty());
  EXPECT_TRUE(stamp.attributes().empty());
  EXPECT_EQ(stamp.source_hash_with_components(), "hash_abcdef01");
}

TEST(PjRtBuildStampTest, ValidationFailures) {
  // Empty identity fields must fail validation.
  EXPECT_THAT(PjRtBuildStamp::Create(PjRtBuildStamp::Options{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("at least one build identity field")));

  // Component with empty name must fail.
  PjRtBuildStamp::Options opt_empty_name;
  opt_empty_name.source_hash = "hash";
  opt_empty_name.components.push_back(PjRtComponentBuildStamp{
      /*kind=*/"", /*name=*/"", /*owner=*/"", /*hash=*/"fp"});
  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(opt_empty_name)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));

  // Component with empty hash must fail.
  PjRtBuildStamp::Options opt_empty_hash;
  opt_empty_hash.source_hash = "hash";
  opt_empty_hash.components.push_back(PjRtComponentBuildStamp{
      /*kind=*/"", /*name=*/"comp", /*owner=*/"", /*hash=*/""});
  EXPECT_THAT(
      PjRtBuildStamp::Create(std::move(opt_empty_hash)),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Component build stamp name and hash must not be empty")));
}

TEST(PjRtBuildStampTest, OptionsFromBuildData) {
  PjRtBuildStamp::Options options = PjRtBuildStamp::Options::FromBuildData();
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp,
                       PjRtBuildStamp::Create(std::move(options)));
  EXPECT_NE(stamp.build_timestamp(), absl::InfinitePast());
}

TEST(PjRtBuildStampTest, ToOptionsAndRecreation) {
  PjRtBuildStamp::Options initial_options;
  initial_options.vcs_revision = "rev1";
  initial_options.source_hash = "hash1";
  initial_options.mosaic_version = 7;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp original,
                       PjRtBuildStamp::Create(std::move(initial_options)));

  PjRtBuildStamp::Options options = original.ToOptions();
  EXPECT_EQ(options.vcs_revision, "rev1");
  EXPECT_EQ(options.source_hash, "hash1");
  EXPECT_THAT(options.mosaic_version, Optional(int64_t{7}));

  options.components.push_back(PjRtComponentBuildStamp{
      /*kind=*/"custom_call",
      /*name=*/"op_a",
      /*owner=*/"pkg:op_a",
      /*hash=*/"hash_a",
  });

  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp modified,
                       PjRtBuildStamp::Create(std::move(options)));
  EXPECT_EQ(modified.components().size(), 1);
  EXPECT_EQ(modified.components()[0].kind, "custom_call");
  EXPECT_EQ(modified.components()[0].name, "op_a");
  EXPECT_EQ(modified.components()[0].owner, "pkg:op_a");
  EXPECT_EQ(modified.components()[0].hash, "hash_a");
  EXPECT_NE(original.source_hash_with_components(),
            modified.source_hash_with_components());
}

TEST(PjRtBuildStampTest, AttributesDoNotAffectSourceHashWithComponents) {
  PjRtBuildStamp::Options base_options;
  base_options.source_hash = "hash1";
  base_options.mosaic_version = 15;
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp base_stamp,
                       PjRtBuildStamp::Create(base_options));

  PjRtBuildStamp::Options attr_options = base_options;
  attr_options.attributes["vendor.candidate"] =
      std::string("xla_tpu_candidate_1");
  attr_options.attributes["is_experimental"] = true;
  attr_options.attributes["int_attr"] = int64_t{42};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp attr_stamp,
                       PjRtBuildStamp::Create(std::move(attr_options)));

  EXPECT_THAT(attr_stamp.mosaic_version(), Optional(int64_t{15}));
  EXPECT_THAT(attr_stamp.GetStringAttribute("vendor.candidate"),
              Optional(absl::string_view("xla_tpu_candidate_1")));
  EXPECT_THAT(attr_stamp.GetAttribute<std::string>("vendor.candidate"),
              Optional(std::string("xla_tpu_candidate_1")));
  EXPECT_THAT(attr_stamp.GetAttribute<bool>("is_experimental"), Optional(true));
  EXPECT_THAT(attr_stamp.GetAttribute<int64_t>("int_attr"),
              Optional(int64_t{42}));
  EXPECT_EQ(attr_stamp.GetAttribute<int64_t>("non_existent"), std::nullopt);
  EXPECT_EQ(attr_stamp.GetStringAttribute("non_existent"), std::nullopt);

  // Attributes must NOT factor into source_hash_with_components.
  EXPECT_EQ(base_stamp.source_hash_with_components(),
            attr_stamp.source_hash_with_components());
}

TEST(PjRtBuildStampTest, CanonicalOrderIndependentComponentHash) {
  PjRtComponentBuildStamp c1{
      /*kind=*/"xla.gpu.native_custom_call",
      /*name=*/"alpha",
      /*owner=*/"pkg:alpha",
      /*hash=*/"1111",
  };
  PjRtComponentBuildStamp c2{
      /*kind=*/"xla.ffi.handler",
      /*name=*/"beta",
      /*owner=*/"pkg:beta",
      /*hash=*/"2222",
  };

  PjRtBuildStamp::Options opt1;
  opt1.source_hash = "base_hash";
  opt1.components = {c1, c2};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp1,
                       PjRtBuildStamp::Create(std::move(opt1)));

  PjRtBuildStamp::Options opt2;
  opt2.source_hash = "base_hash";
  opt2.components = {c2, c1};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp2,
                       PjRtBuildStamp::Create(std::move(opt2)));

  // Changing order of components MUST produce identical
  // source_hash_with_components.
  EXPECT_EQ(stamp1.source_hash_with_components(),
            stamp2.source_hash_with_components());

  // Changing a component hash MUST alter source_hash_with_components.
  PjRtComponentBuildStamp c2_modified = c2;
  c2_modified.hash = "3333";
  PjRtBuildStamp::Options opt3;
  opt3.source_hash = "base_hash";
  opt3.components = {c1, c2_modified};
  ASSERT_OK_AND_ASSIGN(PjRtBuildStamp stamp3,
                       PjRtBuildStamp::Create(std::move(opt3)));
  EXPECT_NE(stamp1.source_hash_with_components(),
            stamp3.source_hash_with_components());
}

}  // namespace
}  // namespace xla
