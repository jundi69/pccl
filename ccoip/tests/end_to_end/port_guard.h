#pragma once

#include <cstdio>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>

/**
 * Verifies that no TCP socket is bound to this port.
 * If the port is not available (i.e. a socket is already bound), the function
 * will wait (sleeping 1 second between retries) until it is.
 *
 * @param port the port to check
 */
static void __guard_port(int port) {
    sockaddr_in addr{};
    int optval = 1;

    // Set up the address structure for binding to any interface on the given port.
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY; // Listen on all interfaces
    addr.sin_port = htons(port);

    while (true) {
        // Create a TCP socket.
        const int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            perror("socket");
            sleep(1);
            continue;
        }

        // Allow the socket to bind to an address that is in TIME_WAIT.
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
            perror("setsockopt");
            close(sock);
            sleep(1);
            continue;
        }

        // Try to bind the socket to the specified port.
        if (bind(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
            // Successfully bound: no TCP socket is using this port.
            close(sock);
            break;
        }
        // Check if the bind failed because the port is in use.
        if (errno == EADDRINUSE) {
            fprintf(stderr, "Port %d in use. Waiting...\n", port);
        } else {
            // Some other error occurred.
            fprintf(stderr, "bind() failed on port %d: %s\n", port, strerror(errno));
        }
        close(sock);
        sleep(1); // Wait a bit before trying again.
    }
}


#define GUARD_PORT(port) __guard_port(port)