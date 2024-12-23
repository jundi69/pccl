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
            ++clients_connected;
            while (clients_connected < n_clients) {
                ASSERT_TRUE(client->acceptNewPeers());
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
    });

    std::thread client2_reduce_thread([&client2] {
        const std::unique_ptr<uint8_t[]> value2(new uint8_t[1024]);
        std::fill_n(value2.get(), 1024, 43);

        const std::unique_ptr<uint8_t[]> result(new uint8_t[1024]);
        std::fill_n(result.get(), 1024, 0);

        ASSERT_TRUE(client2.allReduceAsync(value2.get(), result.get(), 1024, ccoip::ccoipUint8, ccoip::ccoipOpSum, 1));
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
