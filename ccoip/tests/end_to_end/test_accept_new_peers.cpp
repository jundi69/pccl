#include <ccoip.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>
#include <atomic>
#include <port_guard.h>
#include <gtest/gtest.h>

TEST(AcceptNewPeers, TestBasic) {
    GUARD_PORT(CCOIP_PROTOCOL_PORT_MASTER);

    ccoip::CCoIPMaster master({
        .inet = {.protocol = inetIPv4, .ipv4 = {.data = {0, 0, 0, 0}}},
        .port = CCOIP_PROTOCOL_PORT_MASTER
    });
    ASSERT_TRUE(master.launch());

    // client 1
    ccoip::CCoIPClient client1({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);
    ASSERT_TRUE(client1.connect());

    // client 2
    ccoip::CCoIPClient client2({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    std::thread client1_thread([&client1] {
        client1.setMainThread(std::this_thread::get_id());

        while (!client1.isInterrupted()) {
            ASSERT_TRUE(client1.acceptNewPeers());
            const auto world_size = client1.getWorldSize();
            ASSERT_LT(world_size, 3);
            if (world_size == 2) {
                break;
            }
        }
    });
    std::thread client2_thread([&client2] {
        client2.setMainThread(std::this_thread::get_id());
        ASSERT_TRUE(client2.connect());
    });

    client1_thread.join();
    client2_thread.join();
    ASSERT_TRUE(client2.interrupt());
    ASSERT_TRUE(client1.interrupt());

    ASSERT_TRUE(client1.join());
    ASSERT_TRUE(client2.join());

    ASSERT_TRUE(master.interrupt());
    ASSERT_TRUE(master.join());
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
