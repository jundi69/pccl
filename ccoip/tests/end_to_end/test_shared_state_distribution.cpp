#include <ccoip.h>
#include <gtest/gtest.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>
#include <cstring>

// Helper function to establish p2p connection between two clients
static void establishConnections(const std::vector<const ccoip::CCoIPClient *> &clients) {
    size_t n_clients = clients.size();

    std::atomic_int clients_connected = 0;
    uint32_t target_n_clients = 0;
    std::vector<std::thread> client_threads{};
    for (const auto &client: clients) {
        std::thread client_thread([n_clients, &clients_connected, &client] {
            ASSERT_TRUE(client->connect());
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

// Basic shared state distribution test
TEST(SharedStateDistribution, TestBasic) {
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

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 1;

        ccoip_shared_state_sync_info_t info{};
        ASSERT_TRUE(client1.syncSharedState(shared_state, info));

        // client2 state is outdated, so sync FROM client1 TO client2
        ASSERT_EQ(info.tx_bytes, value_size);
        ASSERT_EQ(info.rx_bytes, 0);
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 0x0);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        ccoip_shared_state_sync_info_t info{};
        ASSERT_TRUE(client2.syncSharedState(shared_state, info));

        // client2 state is outdated, so sync FROM client1 TO client2
        ASSERT_EQ(info.tx_bytes, 0);
        ASSERT_EQ(info.rx_bytes, value_size);
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // assert the shared state of client 2 to be equal to that of client 1
    ASSERT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Identical shared state should not trigger data transfer
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedState) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // client 1
    const ccoip::CCoIPClient client1({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);
    // client 2
    const ccoip::CCoIPClient client2({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 1;

        ccoip_shared_state_sync_info_t info{};
        ASSERT_TRUE(client1.syncSharedState(shared_state, info));

        // for identical shared state, no data should be transferred
        ASSERT_EQ(info.rx_bytes, 0);
        ASSERT_EQ(info.tx_bytes, 0);
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
    std::fill_n(value2.get(), value_size, 42);
    std::thread client2_sync_thread([&client2, &value2, value_size] {
        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 1;

        ccoip_shared_state_sync_info_t info{};
        ASSERT_TRUE(client2.syncSharedState(shared_state, info));

        // for identical shared state, no data should be transferred
        ASSERT_EQ(info.rx_bytes, 0);
        ASSERT_EQ(info.tx_bytes, 0);
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // assert the shared state of client 2 to be equal to that of client 1
    ASSERT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Test of shared state distribution with multiple peer groups.
// Shared state synchronization should be local to each peer group.
// We test with two peer groups, each with two clients, where each client of a given peer group
// synchronizes shared state that is identical to the other client of the same peer group, but different
// from the shared state of the clients of the other peer group.
// The asserted behavior is that no data is transferred during the two shared state synchronization processes.
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedStateMultiplePeerGroups) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // group 1, client 1
    const ccoip::CCoIPClient g1client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);
    // group 2, client 2
    const ccoip::CCoIPClient g1client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);

    // group 2, client 1
    const ccoip::CCoIPClient g2client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    // group 2, client 2
    const ccoip::CCoIPClient g2client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;

    const auto launch_sync_func = [value_size](const ccoip::CCoIPClient &client1, const ccoip::CCoIPClient &client2,
                                               const int peer_group) {
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
        std::fill_n(value1.get(), value_size, 42 + peer_group);

        std::thread client1_sync_thread([&client1, value_size, &value1] {
            // client 1 distributes shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key1",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 1;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client1.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);
        });
        std::this_thread::sleep_for(std::chrono::seconds(1));

        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
        std::fill_n(value2.get(), value_size, 42 + peer_group);
        std::thread client2_sync_thread([&client2, value_size, &value2] {
            // client 2 requests shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key1",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 1;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client2.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);
        });

        // wait for shared state sync to complete
        client1_sync_thread.join();
        client2_sync_thread.join();

        // assert the shared state of client 2 to be equal to that of client 1
        ASSERT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);
    };

    launch_sync_func(g1client1, g1client2, 1);
    launch_sync_func(g2client1, g2client2, 2);

    // clean shutdown
    ASSERT_TRUE(g1client1.interrupt());
    ASSERT_TRUE(g1client2.interrupt());
    ASSERT_TRUE(g2client1.interrupt());
    ASSERT_TRUE(g2client2.interrupt());

    ASSERT_TRUE(g1client1.join());
    ASSERT_TRUE(g1client2.join());
    ASSERT_TRUE(g2client1.join());
    ASSERT_TRUE(g2client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Same setup as TestNoSyncIdenticalSharedStateMultiplePeerGroups, but each peer group has a different shared state mask,
// meaning they shared different keys. Within each group, all peers still contain the same peers.
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedStateMultiplePeerGroupsDifferentKeys) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // group 1, client 1
    const ccoip::CCoIPClient g1client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);
    // group 2, client 2
    const ccoip::CCoIPClient g1client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);

    // group 2, client 1
    const ccoip::CCoIPClient g2client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    // group 2, client 2
    const ccoip::CCoIPClient g2client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;

    const auto launch_sync_func = [value_size](const ccoip::CCoIPClient &client1, const ccoip::CCoIPClient &client2,
                                               const int peer_group) {
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
        std::fill_n(value1.get(), value_size, 42 + peer_group);

        std::thread client1_sync_thread([&client1, value_size, &value1, peer_group] {
            // client 1 distributes shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = peer_group == 1 ? "key1" : "key2",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 1;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client1.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);
        });
        std::this_thread::sleep_for(std::chrono::seconds(1));

        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);
        std::fill_n(value2.get(), value_size, 42 + peer_group);
        std::thread client2_sync_thread([&client2, value_size, &value2, peer_group] {
            // client 2 requests shared state
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = peer_group == 1 ? "key1" : "key2",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 1;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client2.syncSharedState(shared_state, info));

            // for identical shared state, no data should be transferred
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);
        });

        // wait for shared state sync to complete
        client1_sync_thread.join();
        client2_sync_thread.join();

        // assert the shared state of client 2 to be equal to that of client 1
        ASSERT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);
    };

    launch_sync_func(g1client1, g1client2, 1);
    launch_sync_func(g2client1, g2client2, 2);

    // clean shutdown
    ASSERT_TRUE(g1client1.interrupt());
    ASSERT_TRUE(g1client2.interrupt());
    ASSERT_TRUE(g2client1.interrupt());
    ASSERT_TRUE(g2client2.interrupt());

    ASSERT_TRUE(g1client1.join());
    ASSERT_TRUE(g1client2.join());
    ASSERT_TRUE(g2client1.join());
    ASSERT_TRUE(g2client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Same setup as TestNoSyncIdenticalSharedStateMultiplePeerGroupsDifferentKeys, but both sync processes are launched concurrently
TEST(SharedStateDistribution, TestNoSyncIdenticalSharedStateMultiplePeerGroupsDifferentKeysConcurrent) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // group 1, client 1
    const ccoip::CCoIPClient g1client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);
    // group 2, client 2
    const ccoip::CCoIPClient g1client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);

    // group 2, client 1
    const ccoip::CCoIPClient g2client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    // group 2, client 2
    const ccoip::CCoIPClient g2client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;

    const auto launch_sync_func = [value_size, &g1client1, &g1client2, &g2client1, &g2client2] {
        const std::unique_ptr<uint8_t[]> value1_g1(new uint8_t[value_size]);
        std::fill_n(value1_g1.get(), value_size, 42 + 1);

        const std::unique_ptr<uint8_t[]> value1_g2(new uint8_t[value_size]);
        std::fill_n(value1_g2.get(), value_size, 42 + 2);

        std::atomic_int client1_sync_ctr = 0;

        std::thread client1_sync_thread(
            [&value1_g1, &value1_g2, value_size, &g1client1, &g2client1, &client1_sync_ctr] {
                // group 1 client 1 distributes shared state
                std::thread g1client1_sync_thread([&g1client1, value_size, &value1_g1, &client1_sync_ctr] {
                    ccoip_shared_state_t shared_state{};
                    shared_state.entries.push_back(ccoip_shared_state_entry_t{
                        .key = "key1",
                        .data_type = ccoip::ccoipUint8,
                        .value = std::span(reinterpret_cast<std::byte *>(value1_g1.get()), value_size),
                        .allow_content_inequality = false
                    });
                    shared_state.revision = 1;
                    ccoip_shared_state_sync_info_t info{};
                    ++client1_sync_ctr;
                    ASSERT_TRUE(g1client1.syncSharedState(shared_state, info));

                    // for identical shared state, no data should be transferred
                    ASSERT_EQ(info.rx_bytes, 0);
                    ASSERT_EQ(info.tx_bytes, 0);
                });

                // group 2 client 1 distributes shared state
                std::thread g2client1_sync_thread([&g2client1, value_size, &value1_g2, &client1_sync_ctr] {
                    ccoip_shared_state_t shared_state{};
                    shared_state.entries.push_back(ccoip_shared_state_entry_t{
                        .key = "key2",
                        .data_type = ccoip::ccoipUint8,
                        .value = std::span(reinterpret_cast<std::byte *>(value1_g2.get()), value_size),
                        .allow_content_inequality = false
                    });
                    shared_state.revision = 1;
                    ccoip_shared_state_sync_info_t info{};
                    ++client1_sync_ctr;
                    ASSERT_TRUE(g2client1.syncSharedState(shared_state, info));

                    // for identical shared state, no data should be transferred
                    ASSERT_EQ(info.rx_bytes, 0);
                    ASSERT_EQ(info.tx_bytes, 0);
                });

                // wait for shared state sync to complete
                g1client1_sync_thread.join();
                g2client1_sync_thread.join();
            });
        while (client1_sync_ctr < 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));

        std::thread client2_sync_thread([&value1_g1, &value1_g2, value_size, &g1client2, &g2client2] {
            // group 1 client 2 requests shared state
            std::thread g1client2_sync_thread([&value1_g1, &g1client2, value_size] {
                ccoip_shared_state_t shared_state{};
                shared_state.entries.push_back(ccoip_shared_state_entry_t{
                    .key = "key1",
                    .data_type = ccoip::ccoipUint8,
                    .value = std::span(reinterpret_cast<std::byte *>(value1_g1.get()), value_size),
                    .allow_content_inequality = false
                });
                shared_state.revision = 1;
                ccoip_shared_state_sync_info_t info{};
                ASSERT_TRUE(g1client2.syncSharedState(shared_state, info));

                // for identical shared state, no data should be transferred
                ASSERT_EQ(info.rx_bytes, 0);
                ASSERT_EQ(info.tx_bytes, 0);
            });

            // group 2 client 2 requests shared state
            std::thread g2client2_sync_thread([&value1_g2, &g2client2, value_size] {
                ccoip_shared_state_t shared_state{};
                shared_state.entries.push_back(ccoip_shared_state_entry_t{
                    .key = "key2",
                    .data_type = ccoip::ccoipUint8,
                    .value = std::span(reinterpret_cast<std::byte *>(value1_g2.get()), value_size),
                    .allow_content_inequality = false
                });
                shared_state.revision = 1;
                ccoip_shared_state_sync_info_t info{};
                ASSERT_TRUE(g2client2.syncSharedState(shared_state, info));

                // for identical shared state, no data should be transferred
                ASSERT_EQ(info.rx_bytes, 0);
                ASSERT_EQ(info.tx_bytes, 0);
            });

            // wait for shared state sync to complete
            g1client2_sync_thread.join();
            g2client2_sync_thread.join();
        });

        // wait for shared state sync to complete
        client1_sync_thread.join();
        client2_sync_thread.join();
    };

    launch_sync_func();

    // clean shutdown
    ASSERT_TRUE(g1client1.interrupt());
    ASSERT_TRUE(g1client2.interrupt());
    ASSERT_TRUE(g2client1.interrupt());
    ASSERT_TRUE(g2client2.interrupt());

    ASSERT_TRUE(g1client1.join());
    ASSERT_TRUE(g1client2.join());
    ASSERT_TRUE(g2client1.join());
    ASSERT_TRUE(g2client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Multistep advancement of shared state with identical updates
TEST(SharedStateDistribution, TestMultiStepAdvancement) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Client 1
    const ccoip::CCoIPClient client1({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);
    // Client 2
    const ccoip::CCoIPClient client2({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    // Initialize shared state values
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 0);
    std::fill_n(value2.get(), value_size, 0);

    // Main loop for clients
    auto client_main = [](const ccoip::CCoIPClient &client, uint8_t *value) {
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 1; step <= num_steps; ++step) {
            // Independently update shared state identically
            std::fill_n(value, value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client.syncSharedState(shared_state, info));

            // Since shared states are identical, no data should be transferred
            ASSERT_EQ(info.tx_bytes, 0);
            ASSERT_EQ(info.rx_bytes, 0);
        }
    };

    // Start client threads
    std::thread client1_main_thread([&client1, &value1, &client_main] {
        client_main(client1, value1.get());
    });
    std::thread client2_main_thread([&client2, &value2, &client_main] {
        client_main(client2, value2.get());
    });

    // Wait for both clients to finish
    client1_main_thread.join();
    client2_main_thread.join();

    // Assert the shared states are identical
    ASSERT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);

    // Clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// "Drag-along" client scenario
TEST(SharedStateDistribution, TestDragAlongClient) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Client 1
    const ccoip::CCoIPClient client1({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);
    // Client 2
    const ccoip::CCoIPClient client2({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 0); // Client 2 does not update value

    // Client 1 continuously updates shared state
    std::thread client1_main_thread([&client1, &value1, value_size, num_steps] {
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 1; step <= num_steps; ++step) {
            // Update value
            std::fill_n(value1.get(), value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client1.syncSharedState(shared_state, info));

            // Client 1 should send data to client 2
            ASSERT_EQ(info.tx_bytes, value_size);
            ASSERT_EQ(info.rx_bytes, 0);
        }
    });

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Client 2 does not update its own shared state but calls syncSharedState
    std::thread client2_main_thread([&client2, &value2, value_size, num_steps] {
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 0; // Client 2 does not update revision

        const std::unique_ptr<uint8_t[]> value1_inferred(new uint8_t[value_size]);
        for (int step = 1; step <= num_steps; ++step) {
            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client2.syncSharedState(shared_state, info));

            // Client 2 should receive data from client 1
            ASSERT_EQ(info.tx_bytes, 0);
            ASSERT_EQ(info.rx_bytes, value_size);

            // infer value1 from step
            std::fill_n(value1_inferred.get(), value_size, static_cast<uint8_t>(42 + step));

            // Value2 should now be updated to match value1
            ASSERT_EQ(std::memcmp(value1_inferred.get(), value2.get(), value_size), 0);
        }
    });

    // Wait for both clients to finish
    client1_main_thread.join();
    client2_main_thread.join();

    // Clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};


// Client gets kicked due to shared state mismatch
TEST(SharedStateDistribution, TestSharedStateMismatchKick) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Client 1
    const ccoip::CCoIPClient client1({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);
    // Client 2
    const ccoip::CCoIPClient client2({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    establishConnections({&client1, &client2});

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    // Initialize values with different content
    std::fill_n(value1.get(), value_size, 42);
    std::fill_n(value2.get(), value_size, 24); // Different content

    // Client 1 synchronizes shared state first
    std::thread client1_main_thread([&client1, &value1, value_size] {
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 1;

        ccoip_shared_state_sync_info_t info{};
        ASSERT_TRUE(client1.syncSharedState(shared_state, info));
    });

    std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait for client 1 to reach syncSharedState

    // Client 2 attempts to sync shared state with mismatched content
    std::thread client2_main_thread([&client2, &value2, value_size] {
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key2",
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 1; // Same revision, but keyset is different

        ccoip_shared_state_sync_info_t info{};

        // Since the content does not match the mask, client2 should be kicked
        ASSERT_FALSE(client2.syncSharedState(shared_state, info));

        // Subsequent calls should fail
        ASSERT_FALSE(client2.syncSharedState(shared_state, info));
    });

    client1_main_thread.join();
    client2_main_thread.join();

    // Clean shutdown
    ASSERT_TRUE(client1.interrupt());
    ASSERT_TRUE(client1.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Multiple peer groups advance the shared state for multiple steps; synchronization should be local to each peer group
TEST(SharedStateDistribution, TestConcurrentAdvancementWithinPeerGroups) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Peer group 1 clients
    const ccoip::CCoIPClient g1client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);
    const ccoip::CCoIPClient g1client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 0);

    // Peer group 2 clients
    const ccoip::CCoIPClient g2client1({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);
    const ccoip::CCoIPClient g2client2({
                                           .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                           .port = CCOIP_PROTOCOL_PORT_MASTER
                                       }, 1);

    establishConnections({&g1client1, &g1client2, &g2client1, &g2client2});

    constexpr size_t value_size = 1024;
    constexpr int num_steps = 5;

    // Function to run client's main loop within a peer group
    auto client_main = [](const ccoip::CCoIPClient &client, uint8_t *value, const std::string &key,
                          const int delay_multiplier) {
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = key,
            .data_type = ccoip::ccoipUint8,
            .value = std::span(reinterpret_cast<std::byte *>(value), value_size),
            .allow_content_inequality = false
        });
        shared_state.revision = 0;

        for (int step = 1; step <= num_steps; ++step) {
            // Independently update shared state identically within peer group
            std::fill_n(value, value_size, static_cast<uint8_t>(42 + step));
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client.syncSharedState(shared_state, info));

            // Since shared states are identical within group, no data should be transferred
            ASSERT_EQ(info.tx_bytes, 0);
            ASSERT_EQ(info.rx_bytes, 0);

            // Sleep to simulate computation with varying delays
            std::this_thread::sleep_for(std::chrono::milliseconds(100 * delay_multiplier));
        }
    };

    // Start clients in peer group 1
    const std::unique_ptr<uint8_t[]> g1value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> g1value2(new uint8_t[value_size]);
    std::fill_n(g1value1.get(), value_size, 0);
    std::fill_n(g1value2.get(), value_size, 0);

    std::thread g1client1_thread([&] {
        client_main(g1client1, g1value1.get(), "key1", 1);
    });
    std::thread g1client2_thread([&] {
        client_main(g1client2, g1value2.get(), "key1", 2); // Different delay
    });

    // Start clients in peer group 2
    const std::unique_ptr<uint8_t[]> g2value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> g2value2(new uint8_t[value_size]);
    std::fill_n(g2value1.get(), value_size, 0);
    std::fill_n(g2value2.get(), value_size, 0);

    std::thread g2client1_thread([&] {
        client_main(g2client1, g2value1.get(), "key2", 2); // Different delay
    });
    std::thread g2client2_thread([&] {
        client_main(g2client2, g2value2.get(), "key2", 1);
    });

    // Wait for all clients to finish
    g1client1_thread.join();
    g1client2_thread.join();
    g2client1_thread.join();
    g2client2_thread.join();

    // Assert the shared states are identical within each peer group
    ASSERT_EQ(std::memcmp(g1value1.get(), g1value2.get(), value_size), 0);
    ASSERT_EQ(std::memcmp(g2value1.get(), g2value2.get(), value_size), 0);

    // Clean shutdown
    ASSERT_TRUE(g1client1.interrupt());
    ASSERT_TRUE(g1client2.interrupt());
    ASSERT_TRUE(g2client1.interrupt());
    ASSERT_TRUE(g2client2.interrupt());

    ASSERT_TRUE(g1client1.join());
    ASSERT_TRUE(g1client2.join());
    ASSERT_TRUE(g2client1.join());
    ASSERT_TRUE(g2client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Concurrent "Drag-Along" operations in multiple peer groups
TEST(SharedStateDistribution, TestConcurrentDragAlongAcrossPeerGroups) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Create multiple peer groups
    constexpr int num_peer_groups = 2;
    constexpr int clients_per_group = 2;
    constexpr size_t value_size = 512;
    constexpr int num_steps = 4;

    // Initialize clients for each peer group
    std::vector<std::vector<std::unique_ptr<ccoip::CCoIPClient> > > peer_groups;
    for (int group = 0; group < num_peer_groups; ++group) {
        std::vector<std::unique_ptr<ccoip::CCoIPClient> > group_clients;
        group_clients.reserve(clients_per_group);
        for (int client_id = 0; client_id < clients_per_group; ++client_id) {
            group_clients.emplace_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                                .inet = {
                                                                                    .protocol = inetIPv4,
                                                                                    .ipv4 = {.data = {127, 0, 0, 1}}
                                                                                },
                                                                                .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                            }, group));
        }
        peer_groups.emplace_back(std::move(group_clients));
    }

    // Establish connections within each peer group
    std::vector<const ccoip::CCoIPClient *> clients_ptrs;
    for (int group = 0; group < num_peer_groups; ++group) {
        for (auto &client: peer_groups[group]) {
            clients_ptrs.push_back(client.get());
        }
    }
    establishConnections(clients_ptrs);

    // Function to perform "drag-along" synchronization within a peer group
    auto drag_along_sync = [&](int group) {
        const auto &clients = peer_groups[group];
        const std::unique_ptr<ccoip::CCoIPClient> &leader = clients[0]; // Leader client
        const std::unique_ptr<ccoip::CCoIPClient> &follower = clients[1]; // Follower client

        // Initialize shared state values
        const std::unique_ptr<uint8_t[]> leader_value(new uint8_t[value_size]);
        const std::unique_ptr<uint8_t[]> follower_value(new uint8_t[value_size]);

        std::fill_n(leader_value.get(), value_size, static_cast<uint8_t>(80 + group));
        std::fill_n(follower_value.get(), value_size, 0); // Follower does not update

        // Leader performs continuous updates
        std::thread leader_thread([&leader, &leader_value, value_size, num_steps, group] {
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "drag_key_" + std::to_string(group),
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(leader_value.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 0;

            for (int step = 1; step <= num_steps; ++step) {
                // Update leader's shared state
                std::fill_n(leader_value.get(), value_size, static_cast<uint8_t>(90 + group + step));
                shared_state.revision = step;

                ccoip_shared_state_sync_info_t info{};
                ASSERT_TRUE(leader->syncSharedState(shared_state, info));

                // Expect leader to send data to follower
                ASSERT_EQ(info.tx_bytes, value_size);
                ASSERT_EQ(info.rx_bytes, 0);

                // Simulate computation delay
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Follower requests synchronization without updating its own state
        std::thread follower_thread([&follower, &leader_value, &follower_value, value_size, num_steps, group] {
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "drag_key_" + std::to_string(group),
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(follower_value.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 0; // Does not update its own revision

            for (int step = 1; step <= num_steps; ++step) {
                ccoip_shared_state_sync_info_t info{};
                ASSERT_TRUE(follower->syncSharedState(shared_state, info));

                // Expect follower to receive data from leader
                ASSERT_EQ(info.tx_bytes, 0);
                ASSERT_EQ(info.rx_bytes, value_size);

                // Verify synchronization
                ASSERT_EQ(std::memcmp(leader_value.get(), follower_value.get(), value_size), 0);

                // Simulate computation delay
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });

        leader_thread.join();
        follower_thread.join();
    };

    // Launch "drag-along" synchronization for all peer groups concurrently
    std::vector<std::thread> drag_along_threads;
    drag_along_threads.reserve(num_peer_groups);
    for (int group = 0; group < num_peer_groups; ++group) {
        drag_along_threads.emplace_back(drag_along_sync, group);
    }

    for (auto &t: drag_along_threads) {
        t.join();
    }

    // Clean shutdown
    for (int group = 0; group < num_peer_groups; ++group) {
        for (const auto &client: peer_groups[group]) {
            ASSERT_TRUE(client->interrupt());
            ASSERT_TRUE(client->join());
        }
    }

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Overlapping keys across peer groups
TEST(SharedStateDistribution, TestConflictOverlappingKeysAcrossPeerGroups) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Create peer groups with overlapping keys
    constexpr int num_peer_groups = 2;
    constexpr int clients_per_group = 2;
    constexpr size_t value_size = 256;

    std::vector<std::vector<std::unique_ptr<ccoip::CCoIPClient> > > peer_groups;
    for (int group = 0; group < num_peer_groups; ++group) {
        std::vector<std::unique_ptr<ccoip::CCoIPClient> > group_clients;
        group_clients.reserve(clients_per_group);
        for (int client_id = 0; client_id < clients_per_group; ++client_id) {
            group_clients.push_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                             .inet = {
                                                                                 .protocol = inetIPv4,
                                                                                 .ipv4 = {.data = {127, 0, 0, 1}}
                                                                             },
                                                                             .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                         }, group));
        }
        peer_groups.emplace_back(std::move(group_clients));
    }

    // Establish connections within each peer group
    std::vector<const ccoip::CCoIPClient *> clients_ptrs;
    for (int group = 0; group < num_peer_groups; ++group) {
        for (auto &client: peer_groups[group]) {
            clients_ptrs.push_back(client.get());
        }
    }
    establishConnections(clients_ptrs);

    // Function to synchronize peer groups with overlapping keys
    auto sync_overlapping_keys = [&](const int group) {
        const auto &clients = peer_groups[group];
        const std::unique_ptr<ccoip::CCoIPClient> &client1 = clients[0];
        const std::unique_ptr<ccoip::CCoIPClient> &client2 = clients[1];

        // Initialize shared state values
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

        // Fill with unique data per group
        std::fill_n(value1.get(), value_size, static_cast<uint8_t>(200 + group));
        std::fill_n(value2.get(), value_size, static_cast<uint8_t>(200 + group));

        // Key is overlapping across peer groups
        std::string key = "shared_key_overlapping";

        // Client 1 syncs
        std::thread client1_sync_thread([&client1, &value1, value_size, key] {
            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = key,
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 1;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client1->syncSharedState(shared_state, info));

            // For identical shared state, no data should be transferred
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);
        });

        // Client 2 syncs
        std::thread client2_sync_thread([&client2, &value1, &value2, value_size, key] {
            // Allow some time for client1 to sync first
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = key,
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = 1;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client2->syncSharedState(shared_state, info));

            // For identical shared state, no data should be transferred
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);

            // Verify that the shared states are identical
            ASSERT_EQ(std::memcmp(value1.get(), value2.get(), value_size), 0);
        });

        client1_sync_thread.join();
        client2_sync_thread.join();
    };

    // Launch synchronization for all peer groups concurrently
    std::vector<std::thread> sync_threads;
    sync_threads.reserve(num_peer_groups);
    for (int group = 0; group < num_peer_groups; ++group) {
        sync_threads.emplace_back(sync_overlapping_keys, group);
    }

    for (auto &t: sync_threads) {
        t.join();
    }

    // Clean shutdown
    for (int group = 0; group < num_peer_groups; ++group) {
        for (const auto &client: peer_groups[group]) {
            ASSERT_TRUE(client->interrupt());
            ASSERT_TRUE(client->join());
        }
    }

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

// Changing peer group membership during synchronization
// Tests the behavior when a new client is dynamically added to a peer group during ongoing synchronization.
TEST(SharedStateDistribution, TestChangingPeerGroupMembershipBetweenSynchronizationSteps) {
    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // Create a peer group
    constexpr int group = 0;
    constexpr int initial_clients = 2;
    constexpr size_t value_size = 256;
    constexpr int num_steps = 3;

    std::vector<std::unique_ptr<ccoip::CCoIPClient> > group_clients{};
    group_clients.reserve(initial_clients + 1);
    for (int client_id = 0; client_id < initial_clients; ++client_id) {
        group_clients.push_back(std::make_unique<ccoip::CCoIPClient>(ccoip_socket_address_t{
                                                                         .inet = {
                                                                             .protocol = inetIPv4,
                                                                             .ipv4 = {.data = {127, 0, 0, 1}}
                                                                         },
                                                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                                                     }, group));
    }

    // Establish initial connections
    std::vector<const ccoip::CCoIPClient *> initial_clients_ptrs;
    initial_clients_ptrs.reserve(group_clients.size());
    for (auto &client: group_clients) {
        initial_clients_ptrs.push_back(client.get());
    }
    establishConnections(initial_clients_ptrs);

    // Initialize shared state values
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    const std::unique_ptr<uint8_t[]> value2(new uint8_t[value_size]);

    std::fill_n(value1.get(), value_size, 150);
    std::fill_n(value2.get(), value_size, 150);

    std::atomic_bool client3_joined = false;

    // Client 1 performs synchronization steps
    std::thread client1_thread([&group_clients, &value1, value_size, num_steps, group, &client3_joined] {
        const std::unique_ptr<ccoip::CCoIPClient> &client1 = group_clients[0];

        for (int step = 1; step <= num_steps; ++step) {
            // Update shared state
            std::fill_n(value1.get(), value_size, static_cast<uint8_t>(160 + step));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client1->syncSharedState(shared_state, info));

            // Expect no data transfer for identical shared state
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);

            // Simulate computation delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // At step 2, dynamically add a new client to the peer group
            if (step == 2) {
                // Create and connect new client
                auto new_client = std::make_unique<ccoip::CCoIPClient>(
                    ccoip_socket_address_t{
                        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                        .port = CCOIP_PROTOCOL_PORT_MASTER
                    }, group);
                const std::unique_ptr<ccoip::CCoIPClient> &added_client = group_clients.emplace_back(
                    std::move(new_client));
                std::thread added_client_thread([&added_client, &client3_joined] {
                    std::cout << "Added client connecting" << std::endl;
                    ASSERT_TRUE(added_client->connect());
                    client3_joined = true;
                });
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                ASSERT_TRUE(client1->acceptNewPeers());
                added_client_thread.join();
            }
        }
    });

    // Client 2 performs synchronization steps
    std::thread client2_thread([&group_clients, &value2, value_size, num_steps] {
        for (int step = 1; step <= num_steps; ++step) {
            const std::unique_ptr<ccoip::CCoIPClient> &client2 = group_clients[1];

            // Update local shared state
            std::fill_n(value2.get(), value_size, static_cast<uint8_t>(160 + step));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client2->syncSharedState(shared_state, info));

            // Expect no data transfer for identical shared state
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);

            // Simulate computation delay
            std::this_thread::sleep_for(std::chrono::milliseconds(150));

            if (step == 2) {
                ASSERT_TRUE(client2->acceptNewPeers());
            }
        }
    });
    while (!client3_joined) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Client 3 performs last synchronization step (3)
    std::thread client3_thread([&group_clients, &value2, value_size, num_steps] {
        const std::unique_ptr<ccoip::CCoIPClient> &client3 = group_clients[2];

        for (int step = 3; step <= num_steps; ++step) {
            // Update local shared state
            std::fill_n(value2.get(), value_size, static_cast<uint8_t>(160 + step));

            ccoip_shared_state_t shared_state{};
            shared_state.entries.push_back(ccoip_shared_state_entry_t{
                .key = "key",
                .data_type = ccoip::ccoipUint8,
                .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value_size),
                .allow_content_inequality = false
            });
            shared_state.revision = step;

            ccoip_shared_state_sync_info_t info{};
            ASSERT_TRUE(client3->syncSharedState(shared_state, info));

            // Expect no data transfer for identical shared state
            ASSERT_EQ(info.rx_bytes, 0);
            ASSERT_EQ(info.tx_bytes, 0);

            // Simulate computation delay
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    client1_thread.join();
    client2_thread.join();
    client3_thread.join();

    // Clean shutdown
    for (const auto &client: group_clients) {
        ASSERT_TRUE(client->interrupt());
        ASSERT_TRUE(client->join());
    }

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
