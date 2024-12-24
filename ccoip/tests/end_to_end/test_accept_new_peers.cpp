#include <ccoip.h>
#include <ccoip_client.hpp>
#include <ccoip_master.hpp>
#include <thread>
#include <atomic>
#include <gtest/gtest.h>

TEST(AcceptNewPeers, TestBasic) {
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
    ASSERT_TRUE(client1.connect());

    // client 2
    const ccoip::CCoIPClient client2({
                                         .inet = {.protocol = inetIPv4, .ipv4 = {.data = {127, 0, 0, 1}}},
                                         .port = CCOIP_PROTOCOL_PORT_MASTER
                                     }, 0);

    std::atomic_bool client2_done = false;
    std::thread client1_thread([&client1, &client2_done] {
        while (!client1.isInterrupted()) {
            ASSERT_TRUE(client1.acceptNewPeers());
            if (client2_done) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // TODO: Investigate why changing this to 10 causes test to hang forever sometimes
        }
    });
    std::thread client2_thread([&client2, &client2_done] {
        ASSERT_TRUE(client2.connect());
        client2_done = true;
    });

    client2_thread.join();
    client1_thread.join();
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
