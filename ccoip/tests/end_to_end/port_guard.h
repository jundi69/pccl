#pragma once

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
    #include <cstdio>
    #include <cstdlib>
#else
    #include <unistd.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <cerrno>
    #include <cstring>
    #include <cstdio>
#endif

/**
 * Verifies that no TCP socket is bound to this port.
 * If the port is not available (i.e. a socket is already bound), the function
 * will wait (sleeping 1 second between retries) until it is.
 *
 * @param port the port to check
 */
static void guard_port(int port) {
#ifdef _WIN32
    // Ensure Winsock is initialized (only once).
    static bool wsa_initialized = false;
    if (!wsa_initialized) {
        WSADATA wsaData;
        int iResult = WSAStartup(MAKEWORD(2,2), &wsaData);
        if (iResult != 0) {
            fprintf(stderr, "WSAStartup failed: %d\n", iResult);
            exit(1);
        }
        wsa_initialized = true;
    }
#endif

    sockaddr_in addr{};
    int optval = 1;

    // Set up the address structure for binding to any interface on the given port.
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY; // Listen on all interfaces
    addr.sin_port = htons(port);

    while (true) {
#ifdef _WIN32
        // Create a TCP socket.
        SOCKET sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == INVALID_SOCKET) {
            fprintf(stderr, "socket() failed with error: %d\n", WSAGetLastError());
            Sleep(1000);
            continue;
        }
#else
        const int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            perror("socket");
            sleep(1);
            continue;
        }
#endif

        // Allow the socket to bind to an address that is in TIME_WAIT.
#ifdef _WIN32
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                       reinterpret_cast<const char*>(&optval), sizeof(optval)) == SOCKET_ERROR) {
            fprintf(stderr, "setsockopt() failed with error: %d\n", WSAGetLastError());
            closesocket(sock);
            Sleep(1000);
            continue;
        }
#else
        if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
            perror("setsockopt");
            close(sock);
            sleep(1);
            continue;
        }
#endif

        // Try to bind the socket to the specified port.
#ifdef _WIN32
        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == SOCKET_ERROR) {
            int err = WSAGetLastError();
            if (err == WSAEADDRINUSE) {
                fprintf(stderr, "Port %d in use. Waiting...\n", port);
            } else {
                fprintf(stderr, "bind() failed on port %d: error %d\n", port, err);
            }
            closesocket(sock);
            Sleep(1000);
            continue;
        }
        // Successfully bound: no TCP socket is using this port.
        closesocket(sock);
        break;
#else
        if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
            // Successfully bound.
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
#endif
    }
}

#define GUARD_PORT(port) guard_port(port)

#undef min
#undef max