#include <gtest/gtest.h>
#include <hash_utils.hpp>
#include <random>
#include <chrono>

#ifdef _MSC_VER
#include <intrin.h>
#else
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#endif
#endif

uint64_t FVN1a_64Hash(const void *data, const size_t size) {
    const auto bytes = static_cast<const uint8_t *>(data);
    uint64_t hash = 0xcbf29ce484222325;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= 0x100000001b3;
    }
    return hash;
}

extern "C" uint32_t mag_crc32c(const void* buffer, size_t size);

TEST(FVN1aTest, BenchmarkAgainstBaseline) {
    std::vector<uint8_t> data(1024 * 1024 * 10, 0); // 10 MB

    // init random
    {
        std::random_device rd;
        std::mt19937_64 generator(rd());
        std::uniform_int_distribution<uint32_t> dist(0, 255);

        for (size_t i = 0; i < data.size(); ++i) { // NOLINT(*-loop-convert)
            data[i] = dist(generator);
        }
    }

    // baseline (FVN1a_64Hash, 100 repeats)
    constexpr int n_repeat = 100; {
        const auto start = std::chrono::high_resolution_clock::now();
        uint64_t hashes[n_repeat]{};

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            hashes[i] = FVN1a_64Hash(data.data(), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Baseline: " << duration.count() << " ms" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(hashes[i], hashes[0]);
        }
        // std::cout << "FVN1a_64Hash: " << std::hex << hashes[0] << std::dec << std::endl;
    }

    // FVN1a_512Hash (100 repeats)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        uint64_t hashes[n_repeat]{};

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            hashes[i] = ccoip::hash_utils::FVN1a_512Hash(data.data(), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "FVN1a_512Hash: " << duration.count() << " ms" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(hashes[i], hashes[0]);
        }

        // std::cout << "FVN1a_512Hash: " << std::hex << hashes[0] << std::dec << std::endl;
    }

    // FVN1a_512HashAccel (100 repeats)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        uint64_t hashes[n_repeat]{};

        for (int i = 0; i < n_repeat; ++i) { // NOLINT(*-loop-convert)
            hashes[i] = ccoip::hash_utils::FVN1a_512HashAccel(data.data(), data.size());
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "FVN1a_512HashAccel: " << duration.count() << " ms" << std::endl;

        // assert all hashes are the same
        for (int i = 1; i < n_repeat; ++i) {
            ASSERT_EQ(hashes[i], hashes[0]);
        }

        // std::cout << "FVN1a_512HashAccel: " << std::hex << hashes[0] << std::dec << std::endl;
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
