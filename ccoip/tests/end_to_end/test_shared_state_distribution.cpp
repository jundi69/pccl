#include <ccoip.h>
#include <gtest/gtest.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>

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

    // p2p connection establishment
    std::atomic_bool client2_connected = false;
    std::thread client1_thread([&client1, &client2_connected] {
        while (!client2_connected) {
            ASSERT_TRUE(client1.acceptNewPeers());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    std::thread client2_thread([&client2, &client2_connected] {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ASSERT_TRUE(client2.connect());
        client2_connected = true;
    });

    // wait for p2p connections to be established
    client2_thread.join();
    client1_thread.join();

    std::thread client1_sync_thread([&client1] {
        constexpr size_t value1_size = 1024;
        const std::unique_ptr<uint8_t[]> value1(new uint8_t[value1_size]);
        std::fill_n(value1.get(), value1_size, 0x42);

        // client 1 distributes shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .value = std::span(reinterpret_cast<std::byte *>(value1.get()), value1_size),
            .data_type = ccoipUint8
        });
        shared_state.revision = 2;
        ASSERT_TRUE(client1.syncSharedState(shared_state));
    });

    std::thread client2_sync_thread([&client2] {
        constexpr size_t value2_size = 1024;
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[value2_size]);
        std::fill_n(value2.get(), value2_size, 0x0);

        // client 2 requests shared state
        ccoip_shared_state_t shared_state{};
        shared_state.entries.push_back(ccoip_shared_state_entry_t{
            .key = "key1",
            .value = std::span(reinterpret_cast<std::byte *>(value2.get()), value2_size),
            .data_type = ccoipUint8
        });
        shared_state.revision = 1;
        ASSERT_TRUE(client2.syncSharedState(shared_state));
    });

    // wait for shared state sync to complete
    client1_sync_thread.join();
    client2_sync_thread.join();

    // clean shutdown
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
};

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
