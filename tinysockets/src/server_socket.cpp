#include <ccoip_types.hpp>

#include "tinysockets.hpp"

#include <iostream>
#include <uv.h>

#include "ccoip_utils.hpp"


void uv_err_check(const int status) {
    if (status < 0) {
        fprintf(stderr, "UV Error: %s\n", uv_strerror(status));
        exit(1);
    }
}

#define UV_ERR_CHECK(status) uv_err_check(status)

namespace tinysockets {
    struct ServerSocketState {
        std::unique_ptr<uv_loop_s> loop;
        std::unique_ptr<uv_tcp_s> tcp_server;
        std::unique_ptr<uv_async_s> async_handle;

        /// List of callbacks invoked on client read
        std::vector<ServerSocketReadCallback> read_callbacks{};
    };
}

tinysockets::ServerSocket::ServerSocket(const ccoip_socket_address_t &listen_address) : listen_address(listen_address),
    server_socket_state(new ServerSocketState{}) {
}

bool tinysockets::ServerSocket::bind() {
    if (bound) {
        return false;
    }
    sockaddr_in addr{};
    if (convert_to_sockaddr(listen_address, addr) != 0) {
        return false;
    }

    server_socket_state->loop = std::make_unique<uv_loop_t>();
    UV_ERR_CHECK(uv_loop_init(server_socket_state->loop.get()));

    server_socket_state->tcp_server = std::make_unique<uv_tcp_t>();
    server_socket_state->tcp_server->data = this;
    UV_ERR_CHECK(uv_tcp_init(server_socket_state->loop.get(), server_socket_state->tcp_server.get()));

    UV_ERR_CHECK(uv_tcp_bind(server_socket_state->tcp_server.get(), reinterpret_cast<const sockaddr *>(&addr), 0));

    server_socket_state->async_handle = std::make_unique<uv_async_t>();
    server_socket_state->async_handle->data = this;
    UV_ERR_CHECK(
        uv_async_init(server_socket_state->loop.get(), server_socket_state->async_handle.get(), [](uv_async_t *handle) {
            const auto *this_ptr = static_cast<ServerSocket *>(handle->data);
            this_ptr->onAsyncSignal();
            }));

    bound = true;
    return true;
}

bool tinysockets::ServerSocket::listen() {
    if (listening) {
        return false;
    }
    UV_ERR_CHECK(uv_listen(reinterpret_cast<uv_stream_t *>(server_socket_state->tcp_server.get()), 128,
        [](uv_stream_t *server, const int status) {
        auto *this_ptr = static_cast<ServerSocket *>(server->data);
        this_ptr->onNewConnection(reinterpret_cast<uv_server_stream_t *>(server), status);
        }));
    listening = true;
    return true;
}

bool tinysockets::ServerSocket::runAsync() {
    if (running) {
        return false;
    }
    server_thread = std::thread([this] {
        UV_ERR_CHECK(uv_run(server_socket_state->loop.get(), UV_RUN_DEFAULT));
        uv_close(reinterpret_cast<uv_handle_t *>(server_socket_state->tcp_server.get()), nullptr);
        uv_close(reinterpret_cast<uv_handle_t *>(server_socket_state->async_handle.get()), nullptr);
        UV_ERR_CHECK(uv_run(server_socket_state->loop.get(), UV_RUN_NOWAIT));
    });
    running = true;
    return true;
}

bool tinysockets::ServerSocket::interrupt() const {
    if (!running) {
        return false;
    }
    uv_async_send(server_socket_state->async_handle.get());
    return true;
}

void tinysockets::ServerSocket::join() {
    server_thread.join();
}

void tinysockets::ServerSocket::addReadCallback(const ServerSocketReadCallback &callback) const {
    server_socket_state->read_callbacks.push_back(callback);
}

void tinysockets::ServerSocket::onAsyncSignal() const {
    uv_stop(server_socket_state->async_handle->loop);
}

void createBuffer(uv_handle_t *, const size_t suggested_size, uv_buf_t *buf) {
    buf->base = new char[suggested_size];
    buf->len = suggested_size;
}

void tinysockets::ServerSocket::onNewConnection(uv_server_stream_t *server, const int status) {
    if (status < 0) {
        return;
    }
    auto *client = static_cast<uv_stream_t *>(malloc(sizeof(uv_stream_t)));
    UV_ERR_CHECK(uv_tcp_init(server_socket_state->loop.get(), reinterpret_cast<uv_tcp_t *>(client)));
    client->data = this;
    if (uv_accept(reinterpret_cast<uv_stream_t *>(server), client) == 0) {
        LOG(INFO) << "New connection accepted";
        uv_read_start(client, createBuffer, [](uv_stream_t *stream, const ssize_t n_read, const uv_buf_t *buf) {
            const auto *this_ptr = static_cast<ServerSocket *>(stream->data);
            this_ptr->onClientRead(stream, n_read, buf);
        });
    } else {
        uv_close(reinterpret_cast<uv_handle_t *>(client), nullptr);
    }
}

void tinysockets::ServerSocket::onClientRead(uv_stream_t *stream, const ssize_t n_read, const uv_buf_t *buf) const {
    // get client address
    sockaddr_storage addr{};
    int addr_len = sizeof(addr);
    UV_ERR_CHECK(
        uv_tcp_getpeername(reinterpret_cast<uv_tcp_t *>(stream), reinterpret_cast<sockaddr *>(&addr), &addr_len)
    );
    ccoip_socket_address_t client_addr{};
    convert_from_sockaddr(reinterpret_cast<const sockaddr *>(&addr), client_addr);

    if (n_read < 0) {
        uv_close(reinterpret_cast<uv_handle_t *>(stream), nullptr);
    } else {
        // handle read
        for (const auto &callback: server_socket_state->read_callbacks) {
            callback(client_addr, std::span(reinterpret_cast<uint8_t *>(buf->base), n_read));
        }
    }
    delete[] buf->base;
}

tinysockets::ServerSocket::~ServerSocket() {
    delete server_socket_state;
}
