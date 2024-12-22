#include <gtest/gtest.h>
#include <tinysockets.hpp>

#include <mutex>

inline ccoip_socket_address_t create_ipv4_address(const uint8_t a, const uint8_t b, const uint8_t c, const uint8_t d,
                                                  const uint16_t port) {
    return ccoip_socket_address_t{
        .inet = {
            .protocol = inetIPv4,
            .ipv4 = {
                .data = {a, b, c, d}
            }
        },
        .port = port
    };
}


struct DummyPacket final : ccoip::Packet {
    std::vector<std::uint8_t> payload;
    static constexpr ccoip::packetId_t packet_id = 0x1234;

    void serialize(PacketWriteBuffer &buffer) const override {
        buffer.writeContents(payload.data(), payload.size());
    }

    bool deserialize(PacketReadBuffer &buffer) override {
        payload.resize(buffer.remaining());
        buffer.readContents(payload.data(), payload.size());
        return true;
    }
};

TEST(TestBlockingIOSocket, test_basic_send_to_server) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    std::condition_variable cv;
    std::mutex m;

    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload.size(), 1);
        EXPECT_EQ(packet->payload[0], 42);

        std::unique_lock lock(m);
        cv.notify_one();
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacket packet{};
    packet.payload = {42};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacket>(packet));

    // wait for server to receive packet
    {
        std::unique_lock lock(m);
        cv.wait(lock);
    }

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_send_and_receive) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload.size(), 1);
        EXPECT_EQ(packet->payload[0], 42);

        EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, *packet)); // send the packet back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacket packet{};
    packet.payload = {42};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacket>(packet));

    auto received = client_socket.receivePacket<DummyPacket>();
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->payload.size(), 1);
    EXPECT_EQ(received->payload[0], 42);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_send_and_receive_large_packet) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload.size(), 1024 * 1024);
        EXPECT_EQ(packet->payload[0], 0xAB);

        EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, *packet)); // send the packet back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacket packet{};
    packet.payload = std::vector<std::uint8_t>(1024 * 1024, 0xAB);
    EXPECT_TRUE(client_socket.sendPacket<DummyPacket>(packet));

    auto received = client_socket.receivePacket<DummyPacket>();
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->payload.size(), 1024 * 1024);
    EXPECT_EQ(received->payload[0], 0xAB);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_send_and_receive_multiple_clients) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    std::atomic packets_received(0);
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload.size(), 1);
        EXPECT_EQ(packet->payload[0], 42);

        packets_received.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_TRUE(server_socket.runAsync());

    constexpr int num_clients = 2;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    clients.reserve(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        auto client = std::make_unique<tinysockets::BlockingIOSocket>(listen_address);
        EXPECT_TRUE(client->establishConnection());
        clients.emplace_back(std::move(client));
    }

    DummyPacket p;
    p.payload = {42};
    for (const auto &client: clients) {
        EXPECT_TRUE(client->sendPacket<DummyPacket>(p));
    }

    for (int i = 0; i < 50 && packets_received.load() < num_clients; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(packets_received.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_send_and_receive_large_packets_concurrently) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    constexpr size_t large_size = 5 * 1024 * 1024; // 5 MB per packet
    const std::vector<uint8_t> large_data(large_size, 0xAA); // Initialize with pattern 0xAA

    std::atomic received_packets(0);
    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet from client";
        }
        EXPECT_EQ(packet->payload.size(), large_size);
        EXPECT_EQ(packet->payload, large_data);
        received_packets.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_TRUE(server_socket.runAsync());

    constexpr int num_clients = 10;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    clients.reserve(num_clients);
    for (int i = 0; i < num_clients; ++i) {
        auto client = std::make_unique<tinysockets::BlockingIOSocket>(listen_address);
        EXPECT_TRUE(client->establishConnection());
        clients.emplace_back(std::move(client));
    }

    auto send_large_packet = [&](tinysockets::BlockingIOSocket *client) {
        DummyPacket p;
        p.payload = large_data;
        EXPECT_TRUE(client->sendPacket<DummyPacket>(p));
    };

    std::vector<std::thread> sender_threads;
    sender_threads.reserve(clients.size());
    for (auto &client: clients) {
        sender_threads.emplace_back(send_large_packet, client.get());
    }

    for (auto &t: sender_threads) {
        t.join();
    }

    for (int i = 0; i < 50 && received_packets.load() < num_clients; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(received_packets.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_concurrent_clients_send_large_packets) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    constexpr size_t large_size = 5 * 1024 * 1024; // 5 MB per packet
    const std::vector<uint8_t> large_data(large_size, 0xAA); // Initialize with pattern 0xAA

    constexpr int num_clients = 10;
    std::atomic received_packets(0);

    server_socket.addReadCallback([&](const ccoip_socket_address_t &, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet from client";
        }
        EXPECT_EQ(packet->payload.size(), large_size);
        EXPECT_EQ(packet->payload[0], 0xAA);
        received_packets.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_TRUE(server_socket.runAsync());

    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    for (int i = 0; i < num_clients; ++i) {
        clients.emplace_back(std::make_unique<tinysockets::BlockingIOSocket>(listen_address));
        EXPECT_TRUE(clients.back()->establishConnection());
    }

    DummyPacket p;
    p.payload = large_data;
    for (const auto &client: clients) {
        EXPECT_TRUE(client->sendPacket<DummyPacket>(p));
    }

    for (int i = 0; i < 50 && received_packets.load() < num_clients; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(received_packets.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_recv_matching_packet_ignore_others) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->payload.size(), 1);
        EXPECT_EQ(packet->payload[0], 42);

        DummyPacket response{};
        response.payload = {43};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, response)); // send packet with payload 43
        EXPECT_TRUE(server_socket.sendPacket<DummyPacket>(client_addr, *packet)); // send packet with same payload back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacket packet{};
    packet.payload = {42};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacket>(packet));

    auto received = client_socket.receiveMatchingPacket<DummyPacket>([](const DummyPacket &p) {
        return p.payload[0] == 42;
    });
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->payload.size(), 1);
    EXPECT_EQ(received->payload[0], 42);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

struct DummyPacketUnique final : ccoip::Packet {
    std::unique_ptr<std::uint8_t[]> payload;
    size_t size = 0;
    static constexpr ccoip::packetId_t packet_id = 0x1235;

    void serialize(PacketWriteBuffer &buffer) const override {
        buffer.write<std::uint64_t>(size);
        buffer.writeContents(payload.get(), size);
    }

    bool deserialize(PacketReadBuffer &buffer) override {
        size = buffer.read<std::uint64_t>();
        payload = std::make_unique<std::uint8_t[]>(size);
        buffer.readContents(payload.get(), size);
        return true;
    }

    DummyPacketUnique(
        const std::initializer_list<std::uint8_t> &list) : payload(std::make_unique<std::uint8_t[]>(list.size())),
                                                           size(list.size()) {
        std::copy_n(list.begin(), size, payload.get());
    }

    DummyPacketUnique(DummyPacketUnique &&other) noexcept {
        size = other.size;
        payload = std::move(other.payload);
    }

    DummyPacketUnique(const DummyPacketUnique &other) = delete;
};

TEST(TestBlockingIOSocket, test_recv_matching_packet_none_lost__inorder) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
        // send packet with same payload back
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response)); // send packet with payload 43
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 42;
    });
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->size, 1);
    EXPECT_EQ(received->payload[0], 42);

    auto received2 = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 43;
    });
    ASSERT_TRUE(received2.has_value());
    EXPECT_EQ(received2->size, 1);
    EXPECT_EQ(received2->payload[0], 43);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

TEST(TestBlockingIOSocket, test_recv_matching_packet_none_lost__outoforder) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response)); // send packet with payload 43
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
        // send packet with same payload back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 42;
    });
    ASSERT_TRUE(received.has_value());
    EXPECT_EQ(received->size, 1);
    EXPECT_EQ(received->payload[0], 42);

    auto received2 = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
        return p.payload[0] == 43;
    });
    ASSERT_TRUE(received2.has_value());
    EXPECT_EQ(received2->size, 1);
    EXPECT_EQ(received2->payload[0], 43);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}


TEST(TestBlockingIOSocket, test_recv_matching_packet_none_lost__multiple_threads) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response)); // send packet with payload 43
        EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
        // send packet with same payload back
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    const DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    // launch two threads to receive packets 42 and 43 respectively
    // synchronize threads precisely such that they will call receiveMatchingPacket roughly at the same time
    std::condition_variable cv{};
    std::mutex m{};
    std::atomic_int num_waiting = 0;
    std::thread t1([&client_socket, &cv, &m, &num_waiting] {
        std::unique_lock lock(m);
        ++num_waiting;
        cv.wait(lock);
        const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
            return p.payload[0] == 42;
        });
        ASSERT_TRUE(received.has_value());
        EXPECT_EQ(received->size, 1);
        EXPECT_EQ(received->payload[0], 42);
    });
    std::thread t2([&client_socket, &cv, &m, &num_waiting] {
        std::unique_lock lock(m);
        ++num_waiting;
        cv.wait(lock);

        const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>([](const DummyPacketUnique &p) {
            return p.payload[0] == 43;
        });
        ASSERT_TRUE(received.has_value());
        EXPECT_EQ(received->size, 1);
        EXPECT_EQ(received->payload[0], 43);
    });
    while (num_waiting.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cv.notify_all();

    t1.join();
    t2.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}


TEST(TestBlockingIOSocket, test_recv_matching_packet_none_lost__multiple_threads__many_packets) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.listen());

    constexpr int num_packets = 2048;

    server_socket.addReadCallback([&](const ccoip_socket_address_t &client_addr, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacketUnique>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_EQ(packet->size, 1);
        EXPECT_EQ(packet->payload[0], 42);

        for (int i = 0; i < num_packets; ++i) {
            const DummyPacketUnique response{std::initializer_list<uint8_t>{43}};
            EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, response));
            // send packet with payload 43
            EXPECT_TRUE(server_socket.sendPacket<DummyPacketUnique>(client_addr, *packet));
            // send packet with same payload back
        }
    });
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    const DummyPacketUnique packet{std::initializer_list<uint8_t>{42}};
    EXPECT_TRUE(client_socket.sendPacket<DummyPacketUnique>(packet));

    // launch two threads to receive packets 42 and 43 respectively
    // synchronize threads precisely such that they will call receiveMatchingPacket roughly at the same time
    std::condition_variable cv{};
    std::mutex m{};
    std::atomic_int num_waiting = 0;
    std::thread t1([&client_socket, &cv, &m, &num_waiting] {
        {
            std::unique_lock lock(m);
            ++num_waiting;
            cv.wait(lock);
        }
        for (int i = 0; i < num_packets; ++i) {
            const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>(
                [](const DummyPacketUnique &p) {
                    return p.payload[0] == 42;
                });
            ASSERT_TRUE(received.has_value());
            EXPECT_EQ(received->size, 1);
            EXPECT_EQ(received->payload[0], 42);
        }
    });
    std::thread t2([&client_socket, &cv, &m, &num_waiting] {
        {
            std::unique_lock lock(m);
            ++num_waiting;
            cv.wait(lock);
        }
        for (int i = 0; i < num_packets; ++i) {
            const auto received = client_socket.receiveMatchingPacket<DummyPacketUnique>(
                [](const DummyPacketUnique &p) {
                    return p.payload[0] == 43;
                });
            ASSERT_TRUE(received.has_value());
            EXPECT_EQ(received->size, 1);
            EXPECT_EQ(received->payload[0], 43);
        }
    });
    while (num_waiting.load() < 2) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    cv.notify_all();

    t1.join();
    t2.join();

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
