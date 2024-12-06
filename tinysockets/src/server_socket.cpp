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

tinysockets::ServerSocket::ServerSocket(const ccoip_socket_address_t &listen_address) : listen_address(listen_address),
    loop(nullptr),
    tcp_server(nullptr),
    async_handle(nullptr) {
}

bool tinysockets::ServerSocket::bind() {
    if (bound) {
        return false;
    }
    sockaddr_in addr{};
    if (convert_to_sockaddr(listen_address, addr) != 0) {
        return false;
    }

    loop = new uv_loop_t{};
    UV_ERR_CHECK(uv_loop_init(loop));

    tcp_server = new uv_tcp_t{};
    tcp_server->data = this;
    UV_ERR_CHECK(uv_tcp_init(loop, tcp_server));

    UV_ERR_CHECK(uv_tcp_bind(tcp_server, reinterpret_cast<const sockaddr *>(&addr), 0));

    async_handle = new uv_async_t{};
    async_handle->data = this;
    UV_ERR_CHECK(uv_async_init(loop, async_handle, [](uv_async_t *handle) {
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
    UV_ERR_CHECK(uv_listen(reinterpret_cast<uv_stream_t *>(tcp_server), 128,
        [](uv_stream_t *server, const int status) {
        const auto *this_ptr = static_cast<ServerSocket *>(server->data);
        this_ptr->onNewConnection(reinterpret_cast<uv_server_stream_t *>(server),
            status);
        }));
    listening = true;
    return true;
}

bool tinysockets::ServerSocket::runAsync() {
    if (running) {
        return false;
    }
    server_thread = std::thread([this] {
        UV_ERR_CHECK(uv_run(loop, UV_RUN_DEFAULT));

        uv_close(reinterpret_cast<uv_handle_t *>(tcp_server), nullptr);
        uv_close(reinterpret_cast<uv_handle_t *>(async_handle), nullptr);
        UV_ERR_CHECK(uv_run(loop, UV_RUN_NOWAIT));
    });
    running = true;
    return true;
}

bool tinysockets::ServerSocket::interrupt() const {
    if (!running) {
        return false;
    }
    uv_async_send(async_handle);
    return true;
}

void tinysockets::ServerSocket::join() {
    server_thread.join();
}

void tinysockets::ServerSocket::onAsyncSignal() const {
    uv_stop(async_handle->loop);
}

void createBuffer(uv_handle_t *, const size_t suggested_size, uv_buf_t *buf) {
    buf->base = new char[suggested_size];
    buf->len = suggested_size;
}

void tinysockets::ServerSocket::onNewConnection(uv_server_stream_t *server, const int status) const {
    if (status < 0) {
        return;
    }
    auto *client = static_cast<uv_stream_t *>(malloc(sizeof(uv_stream_t)));
    UV_ERR_CHECK(uv_tcp_init(loop, reinterpret_cast<uv_tcp_t *>(client)));
    if (uv_accept(reinterpret_cast<uv_stream_t *>(server), client) == 0) {
        LOG(INFO) << "New connection accepted";
        uv_read_start(client, createBuffer, [](uv_stream_t *stream, const ssize_t n_read, const uv_buf_t *buf) {
            auto *this_ptr = static_cast<ServerSocket *>(stream->data);
            this_ptr->onClientRead(stream, n_read, buf);
        });
    } else {
        uv_close(reinterpret_cast<uv_handle_t *>(client), nullptr);
    }
}

void tinysockets::ServerSocket::onClientRead(uv_stream_t *stream, const ssize_t n_read, const uv_buf_t *buf) {
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
    }

    delete[] buf->base;
}

tinysockets::ServerSocket::~ServerSocket() {
    delete tcp_server;
    delete async_handle;
    delete loop;
}
