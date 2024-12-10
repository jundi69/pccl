#include <tinysockets.hpp>
#include <gtest/gtest.h>

inline ccoip_socket_address_t create_ipv4_address(uint8_t a, uint8_t b, uint8_t c, uint8_t d, uint16_t port) {
    return ccoip_socket_address_t{
        .inet = {
            .protocol = inetIPv4,
            .address = {
                .ipv4 = {
                    .data = {a, b, c, d}
                }
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
            .address = {
                .ipv6 = {
                    .data = {},
                },
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
            .address = {
                .ipv6 = {
                    .data = {},
                },
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

    // Bind and listen
    EXPECT_TRUE(server_socket.bind());
    EXPECT_TRUE(server_socket.listen());

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

    // Close the client connection
    EXPECT_TRUE(client_socket.closeConnection());

    // Cleanup
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    // Verify that callbacks were called
    EXPECT_TRUE(read_callback_called);
    EXPECT_TRUE(close_callback_called);
    EXPECT_EQ(received_data, data_to_send);
}

// Test closing a client connection
TEST(TestServerSocket, test_close_client_connection) {
    const auto listen_address = create_ipv4_address(127, 0, 0, 1, 48156);
    tinysockets::ServerSocket server_socket(listen_address);

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

    DummyPacket packet{};
    packet.payload = {1, 2, 3, 4, 5};
    EXPECT_TRUE(client_socket.sendPacket(packet));

    // add a read callback to the server
    std::atomic read_callback_called(false);
    server_socket.addReadCallback(
        [&](const ccoip_socket_address_t &addr, const std::span<std::uint8_t> &data) {
            PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
            const auto recv_packet = server_socket.receivePacket<DummyPacket>(buffer);
            if (!recv_packet) {
                FAIL() << "Failed to receive packet";
            }
            EXPECT_EQ(recv_packet->payload, std::vector<std::uint8_t>({1, 2, 3, 4, 5}));

            // Attempt to close the client connection from the server side
            EXPECT_TRUE(server_socket.closeClientConnection(addr));
            read_callback_called = true;
        });

    // Close the client side as well
    EXPECT_TRUE(client_socket.closeConnection());

    // Cleanup
    EXPECT_TRUE(server_socket.interrupt());
    server_socket.join();

    EXPECT_TRUE(read_callback_called);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
