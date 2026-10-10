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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_H_

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_common.h"

namespace xla {

// Represents a compile-time component (such as an XLA:GPU native custom call
// handler, an FFI compile-time Instantiate handler, or a dialect lowering
// engine) that is not linked into the compiler binary and must be included
// to get a comprehensive source hash of the compiler.
struct PjRtComponentBuildStamp {
  // Category or subsystem of the component, e.g. "xla.ffi.handler".
  std::string kind;

  // Target or symbol identifier, e.g. "custom_matmul".
  std::string name;

  // Build target or package owning the component, e.g.
  // "@openxla//xla/backends/gpu:custom_kernel".
  std::string owner;

  // Fingerprint of the build target or package.
  std::string hash;

  bool operator==(const PjRtComponentBuildStamp& other) const {
    return std::tie(kind, name, owner, hash) ==
           std::tie(other.kind, other.name, other.owner, other.hash);
  }
  bool operator!=(const PjRtComponentBuildStamp& other) const {
    return !(*this == other);
  }
  bool operator<(const PjRtComponentBuildStamp& other) const {
    return std::tie(kind, name, owner, hash) <
           std::tie(other.kind, other.name, other.owner, other.hash);
  }
};

// Immutable value type representing the build provenance, binary hash,
// compile-time components, and extensible attributes for a given PJRT plugin or
// compiler.
//
// Constructed via `PjRtBuildStamp::Create(Options)` to ensure required
// identity invariants are validated and cache key fingerprints are canonical.
class PjRtBuildStamp final {
 public:
  // Options struct used to configure and construct a PjRtBuildStamp.
  struct Options {
    // The build label with which the binary was built.
    std::string build_label;

    // The timestamp at which the binary was built.
    absl::Time build_timestamp = absl::InfinitePast();

    // Mosaic IR compatibility version supported by this compiler.
    std::optional<int64_t> mosaic_version;

    // The VCS revision (Git commit SHA or Piper source URI) from which the
    // binary was built.
    std::string vcs_revision;

    // Hash of the PjRt plugin or compiler.
    std::string source_hash;

    // Compile-time components (such as an XLA:GPU native custom call handler,
    // an FFI compile-time Instantiate handler, or a dialect lowering engine)
    // that are not linked into the compiler binary and must be included to get
    // a comprehensive source hash of the compiler.
    std::vector<PjRtComponentBuildStamp> components;

    // Environment-specific attributes using dot-separated namespaces, e.g.
    // "cuda.toolkit_version" -> "13.4".
    absl::flat_hash_map<std::string, PjRtValueType> attributes;

    // Populates build provenance, e.g. timestamp, VCS revision, etc.
    static Options FromBuildData();
  };

  // Validates invariants, canonicalizes components, precomputes cache key
  // fingerprints, and creates an immutable PjRtBuildStamp.
  static absl::StatusOr<PjRtBuildStamp> Create(Options options);

  // Converts this build stamp back into an Options struct, enabling
  // modification and re-creation (e.g. adding dynamic components).
  Options ToOptions() const;

  absl::string_view build_label() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return options_.build_label;
  }

  absl::Time build_timestamp() const { return options_.build_timestamp; }

  std::optional<int64_t> mosaic_version() const {
    return options_.mosaic_version;
  }

  absl::string_view vcs_revision() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return options_.vcs_revision;
  }

  absl::string_view source_hash() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return options_.source_hash;
  }

  // Returns a deterministic string combining `source_hash` (or a fallback
  // built from `vcs_revision` or `build_timestamp`) with sorted, deduplicated
  // `components`. Does not include `attributes`.
  absl::string_view source_hash_with_components() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return source_hash_with_components_;
  }

  absl::Span<const PjRtComponentBuildStamp> components() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return options_.components;
  }

  const absl::flat_hash_map<std::string, PjRtValueType>& attributes() const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return options_.attributes;
  }

  template <typename T>
  std::optional<T> GetAttribute(absl::string_view key) const {
    auto it = options_.attributes.find(key);
    if (it == options_.attributes.end()) {
      return std::nullopt;
    }
    if (const T* val = std::get_if<T>(&it->second)) {
      return *val;
    }
    return std::nullopt;
  }

  // Zero-copy lookup for a string attribute in `attributes()`.
  // Returns `std::nullopt` if `key` is not present or does not hold a string.
  std::optional<absl::string_view> GetStringAttribute(
      absl::string_view key) const ABSL_ATTRIBUTE_LIFETIME_BOUND;

 private:
  explicit PjRtBuildStamp(Options options,
                          std::string source_hash_with_components);

  Options options_;
  std::string source_hash_with_components_;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_H_
