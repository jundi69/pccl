#include <tinysockets.hpp>
#include <gtest/gtest.h>
#include <future>

inline ccoip_socket_address_t create_ipv4_address(uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint16_t port) {
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

TEST(TestServerSocket, test_bind_valid) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(0, 0, 0, 0, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
}

TEST(TestServerSocket, test_bind_loopback) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(127, 0, 0, 1, 48148);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
}

TEST(TestServerSocket, test_bind_invalid) {
    const ccoip_socket_address_t listen_address = create_ipv4_address(0, 0, 0, 0, 0);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_FALSE(server_socket.bind());
}

TEST(TestServerSocket, test_bind_valid_ipv6) {
    const ccoip_socket_address_t listen_address = {
        .inet = {
            .protocol = inetIPv6,
            .ipv6 = {
                .data = {},
            },
        },
        .port = 48148,
    };
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
}

TEST(TestServerSocket, test_bind_invalid_ipv6) {
    constexpr ccoip_socket_address_t listen_address = {
        .inet = {
            .protocol = inetIPv6,
            .ipv6 = {
                .data = {},
            },
        },
        .port = 0,
    };
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_FALSE(server_socket.bind());
}

// Test binding an already bound socket
TEST(TestServerSocket, test_bind_already_bound) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48149);
    tinysockets::ServerSocket server_socket1(listen_address);
    EXPECT_TRUE(server_socket1.bind());
    EXPECT_TRUE(server_socket1.listen());

    tinysockets::ServerSocket server_socket2(listen_address);
    EXPECT_TRUE(server_socket2.bind());
    EXPECT_FALSE(server_socket2.listen());
}

// Test listening on a bound socket
TEST(TestServerSocket, test_listen_after_bind) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48150);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
}

// Test listening on an unbound socket
TEST(TestServerSocket, test_listen_without_bind) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48151);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_FALSE(server_socket.listen()); // Should fail because socket is not bound
}

// Test running the server asynchronously
TEST(TestServerSocket, test_run_async) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48152);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Allow some time for the server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test running async when already running
TEST(TestServerSocket, test_run_async_already_running) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48153);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());
    EXPECT_FALSE(server_socket.runAsync()); // Should fail because server is already running

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test interrupting and joining the server
TEST(TestServerSocket, test_interrupt_and_join) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48154);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Allow some time for the server to run
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

struct DummyPacket final : ccoip::Packet {
    std::vector<std::uint8_t> payload;
    static constexpr ccoip::packetId_t packet_id = 0x1234;

    void serialize(PacketWriteBuffer &buffer) const override {
        buffer.writeContents(payload.data(), payload.size());
    }

    void deserialize(PacketReadBuffer &buffer) override {
        payload.resize(buffer.remaining());
        buffer.readContents(payload.data(), payload.size());
    }
};

// Test handling client connections and callbacks
TEST(TestServerSocket, test_client_connection_and_callbacks) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48155);
    tinysockets::ServerSocket server_socket(listen_address);

    // Variables to verify callbacks
    std::atomic read_callback_called(false);
    std::atomic close_callback_called(false);
    std::vector<std::uint8_t> received_data;

    // Set up callbacks
    server_socket.addReadCallback(
        [&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
            PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
            const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
            if (!packet) {
                FAIL() << "Failed to receive packet";
            }
            received_data = packet->payload;
            read_callback_called = true;
        });

    server_socket.addCloseCallback([&](const ccoip_socket_address_t &) {
        close_callback_called = true;
    });

    // Bind and listen
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());

    // Run the server asynchronously
    EXPECT_TRUE(server_socket.runAsync());

    // Allow some time for the server to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create a client and connect to the server
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send some data
    std::vector<std::uint8_t> data_to_send = {1, 2, 3, 4, 5};

    DummyPacket packet{};
    packet.payload = data_to_send;
    EXPECT_TRUE(client_socket.sendPacket(packet));

    // Closing the connection before the server has received the data
    // may result in data loss
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Close the client connection
    EXPECT_TRUE(client_socket.closeConnection());

    // Cleanup
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // close_callback_called can take a while to be set, but we don't want to wait forever
    for (int i = 0; i < 10; ++i) {
        if (close_callback_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Verify that callbacks were called
    EXPECT_TRUE(read_callback_called);
    EXPECT_TRUE(close_callback_called);
    EXPECT_EQ(received_data, data_to_send);
}

// Test closing a client connection
TEST(TestServerSocket, test_close_client_connection) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48156);
    tinysockets::ServerSocket server_socket(listen_address);

    // add a read callback to the server
    std::atomic read_callback_called(false), payload_match(false), client_close_success(false);
    server_socket.addReadCallback(
        [&](const ccoip_socket_address_t &addr, const std::span<std::uint8_t> &data) {
            PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
            const auto recv_packet = server_socket.receivePacket<DummyPacket>(buffer);
            if (!recv_packet) {
                FAIL() << "Failed to receive packet";
            }
            payload_match = recv_packet->payload == std::vector<std::uint8_t>({1, 2, 3, 4, 5});

            // Attempt to close the client connection from the server side
            client_close_success = server_socket.closeClientConnection(addr);

            read_callback_called = true;
        });

    // Bind and listen
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());

    // Run the server asynchronously
    EXPECT_TRUE(server_socket.runAsync());

    // Create a client and connect to the server
    tinysockets::BlockingIOSocket client_socket(listen_address);

    EXPECT_TRUE(client_socket.establishConnection());

    DummyPacket packet{};
    packet.payload = {1, 2, 3, 4, 5};
    EXPECT_TRUE(client_socket.sendPacket(packet));

    // read callback may take a while to be called, but we don't want to wait forever
    for (int i = 0; i < 15; ++i) {
        if (read_callback_called) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    EXPECT_TRUE(read_callback_called);
    EXPECT_TRUE(payload_match);
    EXPECT_TRUE(client_close_success);
    EXPECT_FALSE(client_socket.isOpen()); // Should fail because the connection is closed

    // Cleanup
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test running server async without calling listen first
TEST(TestServerSocket, test_run_async_without_listen) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48157);
    tinysockets::ServerSocket server_socket(listen_address);
    // Bind is done but not listen
    EXPECT_TRUE(server_socket.bind());
    EXPECT_FALSE(server_socket.runAsync()); // Should fail because not listening yet
}

// Test binding the same socket twice
TEST(TestServerSocket, test_bind_twice) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48158);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_FALSE(server_socket.bind()); // Second bind should fail
}

// Test listening twice
TEST(TestServerSocket, test_listen_twice) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48159);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_FALSE(server_socket.listen()); // Second listen should fail
}

// Test run async after server interrupted
TEST(TestServerSocket, test_run_async_after_interrupt) {
    const auto listen_address = create_ipv4_address(0, 0, 0, 0, 48160);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    EXPECT_FALSE(server_socket.runAsync()); // Server should not be restarted after join
}

// Test multiple clients connecting simultaneously
TEST(TestServerSocket, test_multiple_clients) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48161);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic packets_received(0);
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        if (auto packet = server_socket.receivePacket<DummyPacket>(buffer)) {
            ++packets_received;
        }
    });

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // We'll connect multiple clients and send packets
    constexpr int num_clients = 5;
    std::vector<std::unique_ptr<tinysockets::BlockingIOSocket> > clients;
    for (int i = 0; i < num_clients; ++i) {
        clients.emplace_back(std::make_unique<tinysockets::BlockingIOSocket>(listen_address));
        EXPECT_TRUE(clients.back()->establishConnection());
    }

    // Send packets from all clients
    DummyPacket p;
    p.payload = {42};
    for (const auto &client: clients) {
        EXPECT_TRUE(client->sendPacket(p));
    }

    // Wait for all packets to arrive
    for (int i = 0; i < 50 && packets_received.load() < num_clients; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_EQ(packets_received.load(), num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending a large packet
TEST(TestServerSocket, test_large_packet) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48162);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic read_callback_called(false);
    const std::vector<uint8_t> large_data(1024 * 1024, 0xAB); // 1 MB of data
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        if (!packet) {
            FAIL() << "Failed to receive large packet";
        }
        EXPECT_EQ(packet->payload.size(), large_data.size());
        EXPECT_EQ(packet->payload, large_data);
        read_callback_called = true;
    });

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Create client and send large packet
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    DummyPacket p{};
    p.payload = large_data;
    EXPECT_TRUE(client_socket.sendPacket(p));

    for (int i = 0; i < 50 && !read_callback_called.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(read_callback_called);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test closing a non-existent client connection
TEST(TestServerSocket, test_close_non_existent_client) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48163);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Dummy address that won't exist as a client
    const ccoip_socket_address_t non_existent_client = create_ipv4_address(127, 0, 0, 1, 9999);
    EXPECT_FALSE(server_socket.closeClientConnection(non_existent_client));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test adding multiple callbacks
TEST(TestServerSocket, test_multiple_callbacks) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48164);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic read_calls(0);
    std::atomic close_calls(0);

    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        ++read_calls;
    });
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        ++read_calls;
    });

    server_socket.addCloseCallback([&](const ccoip_socket_address_t &) {
        ++close_calls;
    });
    server_socket.addCloseCallback([&](const ccoip_socket_address_t &) {
        ++close_calls;
    });

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Connect a client and send something
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());
    DummyPacket p{};
    p.payload = {1, 2, 3};
    EXPECT_TRUE(client_socket.sendPacket(p));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Close client
    EXPECT_TRUE(client_socket.closeConnection());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_GE(read_calls.load(), 2); // Both read callbacks should have fired
    EXPECT_GE(close_calls.load(), 2); // Both close callbacks should have fired

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test connecting a client when the server is not listening
TEST(TestServerSocket, test_client_connect_when_not_listening) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49100);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.bind());
    // Notice we do not call listen()
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_FALSE(client_socket.establishConnection()); // Should fail because server not listening
}

// Test multiple interrupts
TEST(TestServerSocket, test_multiple_interrupts) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49102);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Interrupt the server multiple times
    EXPECT_TRUE(server_socket.interrupt());
    EXPECT_TRUE(server_socket.interrupt());

    server_socket.join();
}

// Test attempting to re-run the server after it has been joined
TEST(TestServerSocket, test_rerun_after_join) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49103);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Attempting to run again after join. Expect failure.
    EXPECT_FALSE(server_socket.runAsync());
}

// Test sending zero-length payload
TEST(TestServerSocket, test_zero_length_packet) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49105);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic callback_called(false);
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        ASSERT_TRUE(packet.has_value());
        EXPECT_TRUE(packet->payload.empty());
        callback_called = true;
    });

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());

    EXPECT_TRUE(server_socket.runAsync());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Connect client
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Send zero-length payload
    DummyPacket p{};
    p.payload = {};
    EXPECT_TRUE(client_socket.sendPacket(p));

    for (int i = 0; i < 10 && !callback_called.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(callback_called);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending unknown packet ID
struct UnknownPacket final : ccoip::Packet {
    std::vector<std::uint8_t> payload;
    static constexpr ccoip::packetId_t packet_id = 0x9999;
    void serialize(PacketWriteBuffer &buffer) const override { buffer.writeContents(payload.data(), payload.size()); }

    void deserialize(PacketReadBuffer &buffer) override {
        payload.resize(buffer.remaining());
        buffer.readContents(payload.data(), payload.size());
    }
};

TEST(TestServerSocket, test_unknown_packet_id) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49106);
    tinysockets::ServerSocket server_socket(listen_address);

    std::atomic callback_called(false);
    server_socket.addReadCallback([&](ccoip_socket_address_t, const std::span<std::uint8_t> &data) {
        // Try to parse as a DummyPacket, but we receive an UnknownPacket
        PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
        const auto packet = server_socket.receivePacket<DummyPacket>(buffer);
        EXPECT_FALSE(packet.has_value()); // Should fail to parse correctly
        callback_called = true;
    });

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());

    EXPECT_TRUE(server_socket.runAsync());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    UnknownPacket unknown{};
    unknown.payload = {0xAB, 0xCD};
    EXPECT_TRUE(client_socket.sendPacket(unknown));

    for (int i = 0; i < 10 && !callback_called.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    EXPECT_TRUE(callback_called);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test concurrent client connections (attempting to connect multiple clients in parallel)
TEST(TestServerSocket, test_concurrent_client_connections) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49107);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    constexpr int num_clients = 10;
    std::vector<std::future<bool> > futures;
    futures.reserve(num_clients);
    for (int i = 0; i < num_clients; i++) {
        futures.emplace_back(std::async(std::launch::async, [&] {
            tinysockets::BlockingIOSocket client(listen_address);
            return client.establishConnection();
        }));
    }

    int success_count = 0;
    for (auto &f: futures) {
        if (f.get()) {
            success_count++;
        }
    }

    // All should succeed if enough backlog/concurrency is supported
    EXPECT_EQ(success_count, num_clients);

    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();
}

// Test sending data from the client after the server has been stopped
TEST(TestServerSocket, test_client_send_after_server_stop) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49108);
    tinysockets::ServerSocket server_socket(listen_address);

    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Stop the server
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Now try sending a packet after server is stopped
    DummyPacket p{};
    p.payload = {0x01};
    // Sending might succeed or fail silently; we just ensure it doesn't crash.
    // If the server is truly down, the send might fail.
    if (!client_socket.sendPacket(p)) {
        SUCCEED();
        return;
    }

    // isOpen() might still return true for some time, but we don't want to wait forever
    for (int i = 0; i < 10; ++i) {
        if (!client_socket.isOpen()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

// Test attempting to close client connections after the server has stopped
TEST(TestServerSocket, test_close_client_after_server_stopped) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 49109);
    tinysockets::ServerSocket server_socket(listen_address);
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());
    EXPECT_TRUE(server_socket.runAsync());

    // Establish a connection
    tinysockets::BlockingIOSocket client_socket(listen_address);
    EXPECT_TRUE(client_socket.establishConnection());

    // Interrupt the server and wait for it to stop
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Now server is stopped, try to close client connection
    EXPECT_FALSE(server_socket.closeClientConnection(listen_address)); // Should fail since server is not running
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
