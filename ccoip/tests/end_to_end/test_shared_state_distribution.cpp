#include <ccoip.h>
#include <gtest/gtest.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>
#include <cstring>

// Helper function to establish p2p connection between two clients
static void establishP2P(const ccoip::CCoIPClient &client1, const ccoip::CCoIPClient &client2) {
    std::atomic_bool client2_connected = false;
    std::thread client1_thread([&client1, &client2_connected] {
        while (!client2_connected) {
            ASSERT_TRUE(client1.acceptNewPeers());
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    });

    std::thread client2_thread([&client2, &client2_connected] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        ASSERT_TRUE(client2.connect());
        client2_connected = true;
    });

    client2_thread.join();
    client1_thread.join();
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
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(client1.connect());

    // client 2
    const ccoip::CCoIPClient client2({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });

    establishP2P(client1, client2);

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoipUint8,
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
            .data_type = ccoipUint8,
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
    });
    ASSERT_TRUE(client1.connect());

    // client 2
    const ccoip::CCoIPClient client2({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });

    establishP2P(client1, client2);

    constexpr size_t value_size = 1024;
    const std::unique_ptr<uint8_t[]> value1(new uint8_t[value_size]);
    std::fill_n(value1.get(), value_size, 42);
    std::thread client1_sync_thread([&client1, &value1, value_size] {
        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .data_type = ccoipUint8,
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
            .data_type = ccoipUint8,
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
    });
    ASSERT_TRUE(client1.connect());

    // Client 2
    const ccoip::CCoIPClient client2({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });

    establishP2P(client1, client2);

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
            .data_type = ccoipUint8,
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
    });
    ASSERT_TRUE(client1.connect());

    // Client 2
    const ccoip::CCoIPClient client2({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });

    establishP2P(client1, client2);

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
            .data_type = ccoipUint8,
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
            .data_type = ccoipUint8,
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
    });
    ASSERT_TRUE(client1.connect());

    // Client 2
    const ccoip::CCoIPClient client2({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });

    // P2P connection establishment
    std::atomic_bool client2_connected = false;
    std::thread client1_thread([&client1, &client2_connected] {
        while (!client2_connected.load()) {
            ASSERT_TRUE(client1.acceptNewPeers());
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    std::thread client2_thread([&client2, &client2_connected] {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        ASSERT_TRUE(client2.connect());
        client2_connected = true;
    });

    // Wait for P2P connections to be established
    client2_thread.join();
    client1_thread.join();

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
            .data_type = ccoipUint8,
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
            .data_type = ccoipUint8,
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

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
