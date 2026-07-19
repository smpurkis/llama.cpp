// SPDX-License-Identifier: MIT
// Copyright (c) 2026 fewtarius
//
// Cross-platform available-RAM query used by auto-sizing code paths in
// common/ and by the Vulkan FA scratch gate in ggml/src/ggml-vulkan/.
//
// Linux:   sysinfo.freeram * mem_unit. Free RAM in bytes; the kernel will
//          reclaim the page cache on demand so this is what auto-sizing
//          actually wants.
// macOS:   hw.memsize (total) as fallback, vm_statistics64 free + inactive
//          as the real answer. Inactive is Apple's reclaimable page cache
//          bucket, same semantics as Linux's free+cache.
// Other:   8 GiB conservative fallback. Lets the feature ship on Windows
//          and exotic platforms without crashing; users on those platforms
//          should configure explicitly rather than rely on auto-sizing.

#pragma once

#include <cstddef>

namespace common {

// Returns currently available RAM in bytes. On Linux/macOS this is the
// reclaimable memory (free + cache / inactive). On other platforms an 8
// GiB fallback. Safe to call frequently - the underlying syscalls are
// cheap and read-only.
std::size_t host_available_ram();

}  // namespace common