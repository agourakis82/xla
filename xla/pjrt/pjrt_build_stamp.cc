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

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tsl/platform/fingerprint.h"
#include "xla/tsl/builddata/builddata.h"

namespace xla {

namespace {

std::string HashComponents(
    absl::Span<const PjRtComponentBuildStamp> components) {
  if (components.empty()) {
    return "";
  }
  std::string serialized;
  for (const auto& comp : components) {
    absl::StrAppend(&serialized, comp.kind, ":", comp.name, ":", comp.owner,
                    ":", comp.hash, ";");
  }
  const tsl::Fprint128 fprint = tsl::Fingerprint128(serialized);
  return absl::StrCat(absl::Hex(fprint.high64, absl::kZeroPad16),
                      absl::Hex(fprint.low64, absl::kZeroPad16));
}

}  // namespace

PjRtBuildStamp::Options PjRtBuildStamp::Options::FromBuildData() {
  Options options;
  options.build_timestamp = absl::FromTimeT(tsl::builddata::TimestampAsInt());
  options.vcs_revision = std::string(tsl::builddata::SourceRevision());
  options.build_label = std::string(tsl::builddata::BuildLabel());
  return options;
}

PjRtBuildStamp::PjRtBuildStamp(Options options,
                               std::string source_hash_with_components)
    : options_(std::move(options)),
      source_hash_with_components_(std::move(source_hash_with_components)) {}

absl::StatusOr<PjRtBuildStamp> PjRtBuildStamp::Create(Options options) {
  if (options.source_hash.empty() && options.vcs_revision.empty() &&
      options.build_timestamp == absl::InfinitePast()) {
    return absl::InvalidArgumentError(
        "PjRtBuildStamp requires at least one build identity field "
        "(source_hash, vcs_revision, or build_timestamp)");
  }
  for (const auto& comp : options.components) {
    if (comp.name.empty() || comp.hash.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Component build stamp name and hash must not be empty. kind='",
          comp.kind, "', name='", comp.name, "', owner='", comp.owner,
          "', hash='", comp.hash, "'"));
    }
  }

  absl::c_sort(options.components);
  options.components.erase(
      std::unique(options.components.begin(), options.components.end()),
      options.components.end());

  std::string source_hash = options.source_hash;
  if (source_hash.empty()) {
    if (!options.vcs_revision.empty()) {
      source_hash = absl::StrCat("vcs:", options.vcs_revision);
    } else if (options.build_timestamp != absl::InfinitePast()) {
      source_hash =
          absl::StrCat("ts:", absl::ToUnixSeconds(options.build_timestamp));
    }
  }

  std::string source_hash_with_components = source_hash;
  if (!options.components.empty()) {
    const std::string components_hash = HashComponents(options.components);
    source_hash_with_components =
        absl::StrCat(source_hash, "+", components_hash);
  }

  return PjRtBuildStamp(std::move(options),
                        std::move(source_hash_with_components));
}

PjRtBuildStamp::Options PjRtBuildStamp::ToOptions() const { return options_; }

std::optional<absl::string_view> PjRtBuildStamp::GetStringAttribute(
    absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND {
  auto it = options_.attributes.find(key);
  if (it == options_.attributes.end()) {
    return std::nullopt;
  }
  if (const auto* val = std::get_if<std::string>(&it->second)) {
    return *val;
  }
  return std::nullopt;
}

}  // namespace xla
