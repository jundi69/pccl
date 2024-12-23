#include <ccoip.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>
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

TEST(AllReduceTest, TestBasic) {
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

        ASSERT_TRUE(client1.allReduceAsync(value1.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));
        ASSERT_TRUE(client1.joinAsyncReduce(1));

        // check result
        for (size_t i = 0; i < 1024; i++) {
            ASSERT_EQ(result[i], 85);
        }
    });

    std::thread client2_reduce_thread([&client2] {
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[1024]);
        std::fill_n(value2.get(), 1024, 43);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(client2.allReduceAsync(value2.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));
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

        ASSERT_TRUE(client1.allReduceAsync(value1.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));
        ASSERT_FALSE(client1.acceptNewPeers());
        ASSERT_TRUE(client1.joinAsyncReduce(1));
    });

    std::thread client2_reduce_thread([&client2] {
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[1024]);
        std::fill_n(value2.get(), 1024, 43);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(client2.allReduceAsync(value2.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));
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

        ASSERT_TRUE(client1.allReduceAsync(value1.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));

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

        ASSERT_TRUE(client2.allReduceAsync(value2.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));

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


int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
