#include <ccoip.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>
#include <random>
#include <typeindex>
#include <gtest/gtest.h>


// Helper function to establish p2p connection between two clients
static void establishConnections(const std::vector<const ccoip::CCoIPClient *> &clients) {
    size_t n_clients = clients.size();

    std::atomic_int clients_connected = 0;
    uint32_t target_n_clients = 0;
    std::vector<std::thread> client_threads{};
    for (const auto &client: clients) {
        std::thread client_thread([n_clients, &clients_connected, &client] {
            ASSERT_TRUE(client->connect());
            ASSERT_TRUE(client->updateTopology());
            ++clients_connected;
            while (clients_connected < n_clients) {
                ASSERT_TRUE(client->acceptNewPeers());
                ASSERT_TRUE(client->updateTopology());
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }
        });
        client_threads.push_back(std::move(client_thread));
        target_n_clients++;
        while (clients_connected < target_n_clients) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    for (auto &client_thread: client_threads) {
        client_thread.join();
    }
}

template<typename T>
class TypeAllReduceTest : public testing::Test {
};

typedef testing::Types<uint8_t, int8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t, float, double>
AllReduceTestTypes;
TYPED_TEST_SUITE(TypeAllReduceTest, AllReduceTestTypes);

template<typename T>
class QuantizeTypedAllReduceTest : public testing::Test {
};

typedef testing::Types<float, double> AllReduceQuantizeTestTypes;
TYPED_TEST_SUITE(QuantizeTypedAllReduceTest, AllReduceQuantizeTestTypes);

inline ccoip::ccoip_data_type_t getCcoipDataType(const std::type_info &ti) {
    static const std::unordered_map<std::type_index, ccoip::ccoip_data_type_t> type_map{
            {typeid(uint8_t), ccoip::ccoipUint8},
            {typeid(int8_t), ccoip::ccoipInt8},
            {typeid(uint16_t), ccoip::ccoipUint16},
            {typeid(int16_t), ccoip::ccoipInt16},
            {typeid(uint32_t), ccoip::ccoipUint32},
            {typeid(int32_t), ccoip::ccoipInt32},
            {typeid(uint64_t), ccoip::ccoipUint64},
            {typeid(int64_t), ccoip::ccoipInt64},
            {typeid(float), ccoip::ccoipFloat},
            {typeid(double), ccoip::ccoipDouble},
    };
    const auto it = type_map.find(std::type_index(ti));
    if (it == type_map.end()) {
        throw std::runtime_error("Unsupported type");
    }
    return it->second;
}


template<typename ValueType>
void reduceTest(const ccoip::ccoip_reduce_op_t reduce_op,
                const ccoip::ccoip_quantization_algorithm_t quant_algo,
                size_t n_elements,
                const uint64_t seed,
                const std::function<ValueType(ValueType, ValueType)> &op,
                size_t num_clients, const bool init_random = true) {
    // Launch master
    const auto ccoip_type = getCcoipDataType(typeid(ValueType));
    ccoip::CCoIPMaster master({
            .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
            .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Create clients
    std::vector<std::unique_ptr<ccoip::CCoIPClient>> clients;
    clients.reserve(num_clients);
    for (size_t i = 0; i < num_clients; ++i) {
        clients.emplace_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                          .inet = {.protocol = inetIPv4,
                                                                              .ipv4 = {.data = {127, 0, 0, 1}}},
                                                                          .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                  }, 0));
    }

    // Establish connections (collect raw pointers for the helper function)
    std::vector<const ccoip::CCoIPClient *> client_ptrs;
    client_ptrs.reserve(num_clients);
    for (auto &client: clients) {
        client_ptrs.push_back(client.get());
    }
    establishConnections(client_ptrs);

    // Allocate data buffers for each client
    std::vector<std::unique_ptr<ValueType[]>> client_values(num_clients);
    for (size_t i = 0; i < num_clients; ++i) {
        client_values[i] = std::make_unique<ValueType[]>(n_elements);
    }

    // Fill each client's buffer with random data
    if (init_random) {
        std::mt19937 gen(seed);
        if constexpr (std::is_integral_v<ValueType>) {
            ValueType lower_bound = 0;
            if constexpr (std::is_signed_v<ValueType>) {
                lower_bound = static_cast<ValueType>(-10);
            }
            auto upper_bound = static_cast<ValueType>(10);
            using DistType = std::conditional_t<sizeof(ValueType) < sizeof(int), int, ValueType>;
            std::uniform_int_distribution<DistType> dist(lower_bound, upper_bound);

            for (size_t i = 0; i < n_elements; ++i) {
                for (size_t c = 0; c < num_clients; ++c) {
                    client_values[c][i] = static_cast<ValueType>(dist(gen));
                }
            }
        } else {
            std::uniform_real_distribution<ValueType> dist(static_cast<ValueType>(0.0),
                                                           static_cast<ValueType>(1.0));
            for (size_t i = 0; i < n_elements; ++i) {
                for (size_t c = 0; c < num_clients; ++c) {
                    client_values[c][i] = dist(gen);
                }
            }
        }
    } else {
        // fill with 1 values
        for (size_t i = 0; i < n_elements; ++i) {
            for (size_t c = 0; c < num_clients; ++c) {
                client_values[c][i] = static_cast<ValueType>(1);
            }
        }
    }

    // Compute expected result by folding (reducing) all client inputs
    auto expected_result = std::make_unique<ValueType[]>(n_elements);
    for (size_t i = 0; i < n_elements; ++i) {
        ValueType aggregate = client_values[0][i];
        for (size_t c = 1; c < num_clients; ++c) {
            aggregate = op(aggregate, client_values[c][i]);
        }
        expected_result[i] = aggregate;
    }

    std::vector<std::unique_ptr<ValueType[]>> results(num_clients);
    for (size_t c = 0; c < num_clients; ++c) {
        const auto &client = clients[c];
        auto &result = results[c] = std::make_unique<ValueType[]>(n_elements);
        std::fill_n(result.get(), n_elements, ValueType{});

        // Perform the asynchronous all-reduce
        ASSERT_TRUE(client->allReduceAsync(client_values[c].get(),
            result.get(),
            n_elements,
            ccoip_type,
            ccoip_type,
            quant_algo,
            reduce_op,
            1));
    }

    // Wait for all clients to finish
    for (size_t c = 0; c < num_clients; ++c) {
        const auto &client = clients[c];
        ASSERT_TRUE(client->joinAsyncReduce(1));
    }

    // Check the results
    for (size_t c = 0; c < num_clients; ++c) {
        const auto &result = results[c];
        for (size_t i = 0; i < n_elements; ++i) {
            ASSERT_EQ(result[i], expected_result[i]);
        }
    }

    // Clean shutdown
    for (const auto &client: clients) {
        ASSERT_TRUE(client->interrupt());
    }
    for (const auto &client: clients) {
        ASSERT_TRUE(client->join());
    }
    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
}

TYPED_TEST(TypeAllReduceTest, TestSumWorldSize2LargeArray) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpSum, ccoip::ccoipQuantizationNone, 203530, 42,
                          [](ValueType a, ValueType b) { return a + b; }, 2);
}

TYPED_TEST(TypeAllReduceTest, TestSumWorldSize3NumElements2) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpSum, ccoip::ccoipQuantizationNone, 2, 42,
                          [](const ValueType a, const ValueType b) { return a + b; }, 3, false);
}

TYPED_TEST(TypeAllReduceTest, TestSumWorldSize3NumElements3) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpSum, ccoip::ccoipQuantizationNone, 3, 42,
                          [](const ValueType a, const ValueType b) { return a + b; }, 3, false);
}

TEST(TypeAllReduceTest, TestSumWorldSize4NumElements2) {
    using ValueType = float;
    reduceTest<ValueType>(ccoip::ccoipOpSum, ccoip::ccoipQuantizationNone, 2, 42,
                          [](const ValueType a, const ValueType b) { return a + b; }, 4, false);
}

TYPED_TEST(QuantizeTypedAllReduceTest, TestSumQuantizedWorldSize2) {
    using ValueType = TypeParam;
    auto ccoipType = getCcoipDataType(typeid(TypeParam));

    ccoip::CCoIPMaster master({
            .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
            .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // client 1
    const ccoip::CCoIPClient client1({
                                             .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                             .port = CCOIP_PROTOCOL_PORT_MASTER,
                                     }, 0);
    // client 2
    const ccoip::CCoIPClient client2({
                                             .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                             .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    // TODO: STRICTER ASSERTIONS FOR QUANTIZED TEST

    std::thread client1_reduce_thread([&client1, ccoipType] {
        const std::unique_ptr<ValueType[]> value1(new ValueType[1024]);
        std::fill_n(value1.get(), 1024, 42);

        const std::unique_ptr<ValueType[]> result(new ValueType[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(
                client1.allReduceAsync(value1.get(), result.get(), 1024, ccoipType, ccoip::ccoipUint8, ccoip::
                    ccoipQuantizationMinMax, ccoip::ccoipOpSum, 1));
        ASSERT_TRUE(client1.joinAsyncReduce(1));

        // check result
        for (size_t i = 0; i < 1024; i++) {
            ASSERT_EQ(result[i], 85);
        }
    });

    std::thread client2_reduce_thread([&client2, ccoipType] {
        const std::unique_ptr<ValueType[]> value2(new ValueType[1024]);
        std::fill_n(value2.get(), 1024, 43);

        const std::unique_ptr<ValueType[]> result(new ValueType[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(
                client2.allReduceAsync(value2.get(), result.get(), 1024, ccoipType, ccoip::ccoipUint8, ccoip::
                    ccoipQuantizationMinMax, ccoip::ccoipOpSum, 1));
        ASSERT_TRUE(client2.joinAsyncReduce(1));

        // check result
        for (size_t i = 0; i < 1024; i++) {
            ASSERT_EQ(result[i], 85);
        }
    });

    // wait for shared state sync to complete
    client1_reduce_thread.join();
    client2_reduce_thread.join();

    // clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
}

TYPED_TEST(TypeAllReduceTest, TestAvgWorldSize2) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpAvg, ccoip::ccoipQuantizationNone, 1024, 42,
                          [](ValueType a, ValueType b) {
                              return static_cast<ValueType>(a + b) / static_cast<ValueType>(2);
                          }, 2);
}


TYPED_TEST(TypeAllReduceTest, TestMinWorldSize2) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpMin, ccoip::ccoipQuantizationNone, 1024, 42,
                          [](ValueType a, ValueType b) { return std::min(a, b); }, 2);
}


TYPED_TEST(TypeAllReduceTest, TestMaxWorldSize2) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpMax, ccoip::ccoipQuantizationNone, 1024, 42,
                          [](ValueType a, ValueType b) { return std::max(a, b); }, 2);
}


TEST(AllReduceTest, TestNoAcceptNewPeersDuringConcurrentReduce) {
    ccoip::CCoIPMaster master({
            .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
            .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // client 1
    const ccoip::CCoIPClient client1({
                                             .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                             .port = CCOIP_PROTOCOL_PORT_MASTER,
                                     }, 0);
    // client 2
    const ccoip::CCoIPClient client2({
                                             .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                             .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    std::thread client1_reduce_thread([&client1] {
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[1024]);
        std::fill_n(value1.get(), 1024, 42);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(
                client1.allReduceAsync(value1.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipUint8, ccoip::
                    ccoipQuantizationNone, ccoip::ccoipOpSum, 1));
        ASSERT_FALSE(client1.acceptNewPeers());
        ASSERT_TRUE(client1.joinAsyncReduce(1));
    });

    std::thread client2_reduce_thread([&client2] {
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[1024]);
        std::fill_n(value2.get(), 1024, 43);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(
                client2.allReduceAsync(value2.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipUint8, ccoip::
                    ccoipQuantizationNone, ccoip::ccoipOpSum, 1));
        ASSERT_FALSE(client2.acceptNewPeers());
        ASSERT_TRUE(client2.joinAsyncReduce(1));
    });

    // wait for shared state sync to complete
    client1_reduce_thread.join();
    client2_reduce_thread.join();

    // clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
}


TEST(AllReduceTest, TestNoSharedStateSyncDuringConcurrentReduce) {
    ccoip::CCoIPMaster master({
            .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
            .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // client 1
    const ccoip::CCoIPClient client1({
                                             .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                             .port = CCOIP_PROTOCOL_PORT_MASTER,
                                     }, 0);
    // client 2
    const ccoip::CCoIPClient client2({
                                             .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                             .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    std::thread client1_reduce_thread([&client1] {
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[1024]);
        std::fill_n(value1.get(), 1024, 42);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(
                client1.allReduceAsync(value1.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipUint8, ccoip::
                    ccoipQuantizationNone, ccoip::ccoipOpSum, 1));

        // shared state sync
        {
            const std::unique_ptr<std::byte[]> shared_value(new std::byte[1024]);
            const std::span shared_value_span(shared_value.get(), 1024);
            ccoip_shared_state_t shared_state{
                    .revision = 1,
                    .entries = {
                            ccoip_shared_state_entry_t{.key = "key1", .data_type = ccoip::ccoipUint8, .device_type = ccoip::ccoipDeviceCpu, .data_ptr = shared_value_span.data(), .data_size = shared_value_span.size_bytes(), .allow_content_inequality=false},
                    }
            };
            ccoip_shared_state_sync_info_t info{};
            ASSERT_FALSE(client1.syncSharedState(shared_state, info));
        }

        ASSERT_TRUE(client1.joinAsyncReduce(1));
    });

    std::thread client2_reduce_thread([&client2] {
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[1024]);
        std::fill_n(value2.get(), 1024, 43);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(
                client2.allReduceAsync(value2.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipUint8, ccoip::
                    ccoipQuantizationNone, ccoip::ccoipOpSum, 1));

        // shared state sync
        {
            const std::unique_ptr<std::byte[]> shared_value(new std::byte[1024]);
            const std::span shared_value_span(shared_value.get(), 1024);
            ccoip_shared_state_t shared_state{
                    .revision = 1,
                    .entries = {
                            ccoip_shared_state_entry_t{.key = "key1", .data_type = ccoip::ccoipUint8, .device_type = ccoip::ccoipDeviceCpu, .data_ptr = shared_value_span.data(), .data_size = shared_value_span.size_bytes(), .allow_content_inequality = false},
                    }
            };
            ccoip_shared_state_sync_info_t info{};
            ASSERT_FALSE(client2.syncSharedState(shared_state, info));
        }

        ASSERT_TRUE(client2.joinAsyncReduce(1));
    });

    // wait for shared state sync to complete
    client1_reduce_thread.join();
    client2_reduce_thread.join();

    // clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
}
// TODO: ADD TEST WITH UNEVEN NUMBER OF ELEMENTS TO TEST TRAILING STAGE

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
