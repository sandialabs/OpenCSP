// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// Lightweight test framework for FFIAM

#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>
#include <chrono>

// Include CUDA vector types (float3, int3, etc.)
#include "util/cuda_utils.h"

namespace ffiam {
namespace test {

// Test result tracking
struct TestResult {
    std::string name;
    bool passed;
    std::string message;
};

class TestRunner {
public:
    static TestRunner& instance() {
        static TestRunner runner;
        return runner;
    }

    void addTest(const std::string& name, std::function<bool()> testFunc) {
        tests_.push_back({name, testFunc});
    }

    int runAll() {
        int passed = 0;
        int failed = 0;

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "                    FFIAM Test Suite\n";
        std::cout << "================================================================\n\n";

        auto startTime = std::chrono::high_resolution_clock::now();

        for (const auto& test : tests_) {
            std::cout << "[ RUN      ] " << test.name << std::endl;

            bool result = false;
            try {
                result = test.func();
            } catch (const std::exception& e) {
                std::cout << "  Exception: " << e.what() << std::endl;
                result = false;
            } catch (...) {
                std::cout << "  Unknown exception" << std::endl;
                result = false;
            }

            if (result) {
                std::cout << "[       OK ] " << test.name << std::endl;
                passed++;
            } else {
                std::cout << "[  FAILED  ] " << test.name << std::endl;
                failed++;
                failedTests_.push_back(test.name);
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "                    Test Summary\n";
        std::cout << "================================================================\n";
        std::cout << "  Total:  " << tests_.size() << " tests\n";
        std::cout << "  Passed: " << passed << "\n";
        std::cout << "  Failed: " << failed << "\n";
        std::cout << "  Time:   " << duration.count() << " ms\n";

        if (!failedTests_.empty()) {
            std::cout << "\nFailed tests:\n";
            for (const auto& name : failedTests_) {
                std::cout << "  - " << name << "\n";
            }
        }

        std::cout << "\n";
        return failed;
    }

private:
    struct Test {
        std::string name;
        std::function<bool()> func;
    };

    std::vector<Test> tests_;
    std::vector<std::string> failedTests_;
};

// Test registration macro
#define TEST(suite, name) \
    bool suite##_##name##_impl(); \
    namespace { \
        struct suite##_##name##_registrar { \
            suite##_##name##_registrar() { \
                ffiam::test::TestRunner::instance().addTest(#suite "." #name, suite##_##name##_impl); \
            } \
        } suite##_##name##_instance; \
    } \
    bool suite##_##name##_impl()

// Assertion macros
#define EXPECT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::cout << "  EXPECT_TRUE failed: " << #expr << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define EXPECT_FALSE(expr) \
    do { \
        if (expr) { \
            std::cout << "  EXPECT_FALSE failed: " << #expr << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define EXPECT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            std::cout << "  EXPECT_EQ failed: " << #expected << " != " << #actual << std::endl; \
            std::cout << "    Expected: " << (expected) << std::endl; \
            std::cout << "    Actual:   " << (actual) << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define EXPECT_NE(val1, val2) \
    do { \
        if ((val1) == (val2)) { \
            std::cout << "  EXPECT_NE failed: " << #val1 << " == " << #val2 << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define EXPECT_NEAR(expected, actual, tolerance) \
    do { \
        if (std::abs((expected) - (actual)) > (tolerance)) { \
            std::cout << "  EXPECT_NEAR failed: |" << #expected << " - " << #actual << "| > " << #tolerance << std::endl; \
            std::cout << "    Expected: " << (expected) << std::endl; \
            std::cout << "    Actual:   " << (actual) << std::endl; \
            std::cout << "    Diff:     " << std::abs((expected) - (actual)) << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define EXPECT_NOT_NULL(ptr) \
    do { \
        if ((ptr) == nullptr) { \
            std::cout << "  EXPECT_NOT_NULL failed: " << #ptr << " is null" << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define EXPECT_NULL(ptr) \
    do { \
        if ((ptr) != nullptr) { \
            std::cout << "  EXPECT_NULL failed: " << #ptr << " is not null" << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

// Helper for float3 comparison
inline bool float3_near(const float3& a, const float3& b, float tol = 0.01f) {
    return std::abs(a.x - b.x) < tol &&
           std::abs(a.y - b.y) < tol &&
           std::abs(a.z - b.z) < tol;
}

#define EXPECT_FLOAT3_NEAR(expected, actual, tolerance) \
    do { \
        if (!ffiam::test::float3_near((expected), (actual), (tolerance))) { \
            std::cout << "  EXPECT_FLOAT3_NEAR failed" << std::endl; \
            std::cout << "    Expected: (" << (expected).x << ", " << (expected).y << ", " << (expected).z << ")" << std::endl; \
            std::cout << "    Actual:   (" << (actual).x << ", " << (actual).y << ", " << (actual).z << ")" << std::endl; \
            std::cout << "    at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

}  // namespace test
}  // namespace ffiam

// Main function macro
#define RUN_ALL_TESTS() ffiam::test::TestRunner::instance().runAll()
