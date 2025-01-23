#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <random>
#include <numeric>
#include <cmath>

// Forward declarations (implement these in your ring_reduce_impl)
std::vector<double> ring_allreduce(int rank, const std::vector<double> &local_data);

bool init_mailboxes(int new_world_size);

extern std::atomic<size_t> bytes_sent;
extern std::atomic<size_t> bytes_received;

// Utility function to run allreduce across threads (similar to Python implementation)
std::vector<std::vector<double>> run_allreduce_across_threads(const int world_size,
                                                              const std::vector<std::vector<double>> &local_data) {
    // make sure local data arrays have exactly world_size elements
    if (local_data.size() != world_size) {
        throw std::runtime_error("local_data must have exactly world_size elements");
    }

    // make sure each local data array has the same size
    const size_t expected_size = local_data[0].size();
    for (const auto &data: local_data) {
        if (data.size() != expected_size) {
            throw std::runtime_error("all local_data arrays must have the same size");
        }
    }

    // storage for results
    std::vector results(world_size, std::vector<double>(expected_size));

    const auto worker = [&results, &local_data](const int rank) {
        const auto arr = ring_allreduce(rank, local_data[rank]);
        results[rank] = arr;
    };

    // Launch threads
    std::vector<std::thread> threads{};
    threads.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        threads.emplace_back([r, &worker] {
            worker(r);
        });
    }

    // Wait for all threads to finish
    for (auto &t: threads) {
        t.join();
    }

    return results;
}

// Utility function to generate test world sizes
std::vector<int> get_world_sizes_to_test() {
    return {2, 3, 8, 13, 16, 32, 64};
}

// Utility function to generate array lengths to test
std::vector<int> get_array_lengths_to_test() {
    return {100, 128, 131, 213, 256, 512, 513};
}

// Helper function to check if two vectors are close
bool allclose(const std::vector<double> &a, const std::vector<double> &b, double rtol = 1e-5, double atol = 1e-8) {
    if (a.size() != b.size())
        return false;

    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(a[i] - b[i]);
        const double tolerance = atol + rtol * std::abs(b[i]);
        if (diff > tolerance)
            return false;
    }
    return true;
}

// Test fixture for parameterized tests
class RingAllReduceTest : public ::testing::TestWithParam<int> {
};

// Small fixed data test
TEST_P(RingAllReduceTest, SmallFixedData) {
    const int world_size = GetParam();

    // Prepare deterministic data for each rank
    std::vector<std::vector<double>> local_data_arrays;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data;
        rank_data.reserve(3);
        for (int j = 0; j < 3; ++j) {
            rank_data.push_back(i + j);
        }
        local_data_arrays.push_back(rank_data);
    }

    // Calculate expected sum
    std::vector expected(3, 0.0);
    for (const auto &arr: local_data_arrays) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected[j] += arr[j];
        }
    }

    // Run allreduce
    init_mailboxes(world_size);
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // Check results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results[r], expected))
            << "Rank " << r << " output mismatch";
    }
}

// Random data test
TEST_P(RingAllReduceTest, RandomDataLength5) {
    const int world_size = GetParam();

    // Set random seed for reproducibility
    std::mt19937 gen(0);
    std::normal_distribution<> dis(0, 1);

    // Generate random data
    std::vector<std::vector<double>> local_data_arrays;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data;
        rank_data.reserve(5);
        for (int j = 0; j < 5; ++j) {
            rank_data.push_back(dis(gen));
        }
        local_data_arrays.push_back(rank_data);
    }

    // Calculate expected sum
    std::vector expected(5, 0.0);
    for (const auto &arr: local_data_arrays) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected[j] += arr[j];
        }
    }

    // Run allreduce
    init_mailboxes(world_size);
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // Check results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results[r], expected))
            << "Rank " << r << " output mismatch";
    }
}

// Zero length array test
TEST_P(RingAllReduceTest, ZeroLengthArray) {
    const int world_size = GetParam();

    // Prepare zero-length arrays
    const std::vector<std::vector<double>> local_data_arrays(world_size);

    // Run allreduce
    init_mailboxes(world_size);
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // Check results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_EQ(results[r].size(), 0);
    }
}

// Parameterized test for larger random arrays with different lengths
class RingAllReduceParamTest : public ::testing::TestWithParam<std::tuple<int, int>> {
};

TEST_P(RingAllReduceParamTest, LargerRandomArray) {
    const int world_size = std::get<0>(GetParam());
    const int length = std::get<1>(GetParam());

    // Set random seed for reproducibility
    std::mt19937 gen(123);
    std::normal_distribution<> dis(0, 1);

    // Generate random data
    std::vector<std::vector<double>> local_data_arrays;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data;
        rank_data.reserve(length);
        for (int j = 0; j < length; ++j) {
            rank_data.push_back(dis(gen));
        }
        local_data_arrays.push_back(rank_data);
    }

    // Calculate expected sum
    std::vector expected(length, 0.0);
    for (const auto &arr: local_data_arrays) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected[j] += arr[j];
        }
    }

    // Run allreduce
    init_mailboxes(world_size);
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // Check results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results[r], expected))
            << "Rank " << r << " output mismatch";
    }
}

// Uneven chunk distribution test
TEST_P(RingAllReduceTest, UnevenChunkDistribution) {
    const int world_size = GetParam();
    constexpr int array_length = 10;

    // Set random seed for reproducibility
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0, 1);

    // Generate random data
    std::vector<std::vector<double>> local_data_arrays;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data;
        rank_data.reserve(array_length);
        for (int j = 0; j < array_length; ++j) {
            rank_data.push_back(dis(gen));
        }
        local_data_arrays.push_back(rank_data);
    }

    // Calculate expected sum
    std::vector expected(array_length, 0.0);
    for (const auto &arr: local_data_arrays) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected[j] += arr[j];
        }
    }

    // Run allreduce
    init_mailboxes(world_size);
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // Check results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results[r], expected))
            << "Rank " << r << " output mismatch";
    }
}

// Insufficient chunks test
TEST_P(RingAllReduceTest, InsufficientChunks) {
    const int world_size = GetParam();
    const int array_length = world_size - 1;

    // Set random seed for reproducibility
    std::mt19937 gen(123);
    std::normal_distribution<> dis(0, 1);

    // Generate random data
    std::vector<std::vector<double>> local_data_arrays;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data;
        rank_data.reserve(array_length);
        for (int j = 0; j < array_length; ++j) {
            rank_data.push_back(dis(gen));
        }
        local_data_arrays.push_back(rank_data);
    }

    // Calculate expected sum
    std::vector expected(array_length, 0.0);
    for (const auto &arr: local_data_arrays) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected[j] += arr[j];
        }
    }

    // Run allreduce
    init_mailboxes(world_size);
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // Check results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results[r], expected))
            << "Rank " << r << " output mismatch";
    }
}

// Multiple runs test
TEST_P(RingAllReduceTest, MultipleRunsInOneTest) {
    const int world_size = GetParam();

    // First run
    std::vector<std::vector<double>> arrs_1;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data(3);
        std::fill(rank_data.begin(), rank_data.end(), static_cast<double>(i));
        arrs_1.push_back(rank_data);
    }

    // Calculate expected sum for first run
    std::vector expected_1(3, 0.0);
    for (const auto &arr: arrs_1) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected_1[j] += arr[j];
        }
    }

    // Run allreduce for first time
    init_mailboxes(world_size);
    const auto results_1 = run_allreduce_across_threads(world_size, arrs_1);

    // Check first run results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results_1[r], expected_1))
            << "First run: Rank " << r << " output mismatch";
    }

    // Second run
    std::vector<std::vector<double>> arrs_2;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data(3);
        for (int j = 0; j < 3; ++j) {
            rank_data[j] = j * (i + 1);
        }
        arrs_2.push_back(rank_data);
    }

    // Calculate expected sum for second run
    std::vector expected_2(3, 0.0);
    for (const auto &arr: arrs_2) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected_2[j] += arr[j];
        }
    }

    // Run allreduce for second time
    init_mailboxes(world_size);
    const auto results_2 = run_allreduce_across_threads(world_size, arrs_2);

    // Check second run results
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results_2[r], expected_2))
            << "Second run: Rank " << r << " output mismatch";
    }
}

// Byte counters test
TEST_P(RingAllReduceTest, ByteCounters) {
    const int world_size = GetParam();

    // Set random seed for reproducibility
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0, 1);

    // Generate random data
    std::vector<std::vector<double>> local_data_arrays;
    for (int i = 0; i < world_size; ++i) {
        std::vector<double> rank_data;
        rank_data.reserve(5);
        for (int j = 0; j < 5; ++j) {
            rank_data.push_back(dis(gen));
        }
        local_data_arrays.push_back(rank_data);
    }

    // Reset mailboxes
    init_mailboxes(world_size);

    // Before running check bytes
    EXPECT_EQ(bytes_sent, 0);
    EXPECT_EQ(bytes_received, 0);

    // Run allreduce
    const auto results = run_allreduce_across_threads(world_size, local_data_arrays);

    // After running check bytes
    EXPECT_GT(bytes_sent, 0);
    EXPECT_GT(bytes_received, 0);

    // Calculate expected sum
    std::vector expected(5, 0.0);
    for (const auto &arr: local_data_arrays) {
        for (size_t j = 0; j < arr.size(); ++j) {
            expected[j] += arr[j];
        }
    }

    // Check results correctness
    for (int r = 0; r < world_size; ++r) {
        EXPECT_TRUE(allclose(results[r], expected))
            << "Rank " << r << " output mismatch";
    }
}

// Instantiate parameterized tests for world sizes
INSTANTIATE_TEST_SUITE_P(
    RingAllReduceWorldSizes,
    RingAllReduceTest,
    ::testing::ValuesIn(get_world_sizes_to_test())
);

// Instantiate parameterized tests for world sizes and array lengths
INSTANTIATE_TEST_SUITE_P(
    RingAllReduceArrayLengths,
    RingAllReduceParamTest,
    ::testing::Combine(
        ::testing::ValuesIn(get_world_sizes_to_test()),
        ::testing::ValuesIn(get_array_lengths_to_test())
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}