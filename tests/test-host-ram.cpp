#undef NDEBUG

#include "host-ram.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

static int tests_run    = 0;
static int tests_failed = 0;

#define RUN(name) do {                              \
    printf("  %-50s ", #name);                      \
    fflush(stdout);                                 \
    tests_run++;                                    \
    try {                                           \
        test_##name();                              \
        printf("OK\n");                             \
    } catch (const std::exception & e) {            \
        (void)e;                                    \
        printf("FAIL: %s\n", e.what());             \
        tests_failed++;                             \
    }                                               \
} while (0)

#define REQUIRE(cond, msg) do {                     \
    if (!(cond)) {                                  \
        throw std::runtime_error(msg);              \
    }                                               \
} while (0)

// host_available_ram must return a positive value (the 8 GiB fallback
// is at minimum > 0, and on any real system it should be at least a few MB).
static void test_available_ram_positive() {
    auto ram = common::host_available_ram();
    REQUIRE(ram > 0, "host_available_ram() returned 0");
}

// host_available_ram_query should return true on Linux/macOS and report
// a non-zero value. The result should match (roughly) the value from
// host_available_ram().
static void test_query_succeeds() {
    std::size_t bytes = 0;
    bool ok = common::host_available_ram_query(&bytes);
    // On Linux this should succeed
    REQUIRE(ok, "host_available_ram_query returned false on Linux");
    REQUIRE(bytes > 0, "host_available_ram_query returned 0 bytes");
}

// host_available_ram_query with null out param must return false.
static void test_query_null_param() {
    bool ok = common::host_available_ram_query(nullptr);
    REQUIRE(!ok, "host_available_ram_query(nullptr) should return false");
}

// Repeated calls must be consistent within an order of magnitude (system
// memory doesn't dramatically change between two sequential calls).
static void test_repeated_calls_consistent() {
    auto a = common::host_available_ram();
    auto b = common::host_available_ram();
    auto c = common::host_available_ram();
    REQUIRE(a > 0, "first call returned 0");
    REQUIRE(b > 0, "second call returned 0");
    REQUIRE(c > 0, "third call returned 0");
    // All values should be within 10% of each other
    auto max_val = a > b ? a : b;
    max_val = max_val > c ? max_val : c;
    auto min_val = a < b ? a : b;
    min_val = min_val < c ? min_val : c;
    REQUIRE(min_val * 10 / 9 >= max_val,
            "repeated calls differ by more than 10%");
}

int main(void) {
    printf("test-host-ram: host_available_ram unit tests\n");
    printf("============================================\n\n");

    RUN(available_ram_positive);
    RUN(query_succeeds);
    RUN(query_null_param);
    RUN(repeated_calls_consistent);

    printf("\n============================================\n");
    printf("Ran %d tests, %d failed\n", tests_run, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
