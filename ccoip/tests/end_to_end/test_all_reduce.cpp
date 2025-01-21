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
void reduceTest(ccoip::ccoip_reduce_op_t reduce_op, ccoip::ccoip_quantization_algorithm_t quant_algo, size_t n_elements,
                const uint64_t seed,
                const std::function<ValueType(ValueType, ValueType)> &op) {
    auto ccoip_type = getCcoipDataType(typeid(ValueType));
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

    const std::unique_ptr<ValueType[]> value1(new ValueType[n_elements]);
    const std::unique_ptr<ValueType[]> value2(new ValueType[n_elements]);

    // fill random
    {
        std::mt19937 gen(seed);
        if constexpr (std::is_integral_v<ValueType>) {
            ValueType lower_bound;
            if constexpr (std::is_signed_v<ValueType>) {
                lower_bound = -10;
            } else {
                lower_bound = 0;
            }
            ValueType upper_bound = 10;
            using DistType = std::conditional_t<sizeof(ValueType) < sizeof(int), int, ValueType>;

            std::uniform_int_distribution<DistType> dist(lower_bound, upper_bound);
            for (size_t i = 0; i < n_elements; i++) {
                value1[i] = static_cast<ValueType>(dist(gen));
                value2[i] = static_cast<ValueType>(dist(gen));
            }
        } else {
            std::uniform_real_distribution<ValueType> dist(0.0,
                                                           1.0);
            for (size_t i = 0; i < n_elements; i++) {
                value1[i] = dist(gen);
                value2[i] = dist(gen);
            }
        }
    }

    const std::unique_ptr<ValueType[]> expected_result(new ValueType[n_elements]);
    for (size_t i = 0; i < n_elements; i++) {
        expected_result[i] = static_cast<ValueType>(op(value1[i], value2[i]));
    }

    std::thread client1_reduce_thread(
            [&client1, ccoip_type, &value1, &expected_result, n_elements, reduce_op, quant_algo] {
                const std::unique_ptr<ValueType[]> result(new ValueType[n_elements]);
                std::fill_n(result.get(), n_elements, 0);

                ASSERT_TRUE(
                        client1.allReduceAsync(value1.get(), result.get(), n_elements, ccoip_type, ccoip_type,
                            quant_algo, reduce_op, 1));
                ASSERT_TRUE(client1.joinAsyncReduce(1));

                // check result
                for (size_t i = 0; i < n_elements; i++) {
                    EXPECT_EQ(result[i], expected_result[i]) << "Mismatch at index " << i;
                }
            });

    std::thread client2_reduce_thread(
            [&client2, ccoip_type, &value2, &expected_result, n_elements, reduce_op, quant_algo] {
                const std::unique_ptr<ValueType[]> result(new ValueType[n_elements]);
                std::fill_n(result.get(), n_elements, 0);

                ASSERT_TRUE(
                        client2.allReduceAsync(value2.get(), result.get(), n_elements, ccoip_type, ccoip_type,
                            quant_algo, reduce_op, 1));
                ASSERT_TRUE(client2.joinAsyncReduce(1));

                // check result
                for (size_t i = 0; i < n_elements; i++) {
                    EXPECT_EQ(result[i], expected_result[i]) << "Mismatch at index " << i;
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

TYPED_TEST(TypeAllReduceTest, TestSumWorldSize2) {
    using ValueType = double;
    reduceTest<ValueType>(ccoip::ccoipOpSum, ccoip::ccoipQuantizationNone, 203530, 42,
                          [](ValueType a, ValueType b) { return a + b; });
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
                          });
}


TYPED_TEST(TypeAllReduceTest, TestMinWorldSize2) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpMin, ccoip::ccoipQuantizationNone, 1024, 42,
                          [](ValueType a, ValueType b) { return std::min(a, b); });
}


TYPED_TEST(TypeAllReduceTest, TestMaxWorldSize2) {
    using ValueType = TypeParam;
    reduceTest<ValueType>(ccoip::ccoipOpMax, ccoip::ccoipQuantizationNone, 1024, 42,
                          [](ValueType a, ValueType b) { return std::max(a, b); });
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
                            ccoip_shared_state_entry_t{"key1", ccoip::ccoipUint8, shared_value_span, false},
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
                            ccoip_shared_state_entry_t{"key1", ccoip::ccoipUint8, shared_value_span, false},
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
