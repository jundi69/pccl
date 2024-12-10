#include "ccoip_utils.hpp"

#include <cstring>
#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

int convert_to_sockaddr_ipv4(const ccoip_socket_address_t &ccoip_addr, sockaddr_in *sock_addr_out) {
    *sock_addr_out = {};
    if (ccoip_addr.inet.protocol == inetIPv4) {
        // IPv4 conversion
        sock_addr_out->sin_family = AF_INET;
        sock_addr_out->sin_port = htons(ccoip_addr.port); // Convert port to network byte order
        sock_addr_out->sin_addr.s_addr = ccoip_addr.inet.address.ipv4.data[0] |
                                         (ccoip_addr.inet.address.ipv4.data[1] << 8) |
                                         (ccoip_addr.inet.address.ipv4.data[2] << 16) |
                                         (ccoip_addr.inet.address.ipv4.data[3] << 24);
    } else if (ccoip_addr.inet.protocol == inetIPv6) {
        return -1; // Unsupported protocol
    }
    return 0; // Success
}

int convert_to_sockaddr_ipv6(const ccoip_socket_address_t &ccoip_addr, sockaddr_in6 *sock_addr_out) {
    *sock_addr_out = {};
    if (ccoip_addr.inet.protocol == inetIPv6) {
        // IPv6 conversion
        sock_addr_out->sin6_family = AF_INET6;
        sock_addr_out->sin6_port = htons(ccoip_addr.port); // Convert port to network byte order
        for (int i = 0; i < 16; i++) {
            sock_addr_out->sin6_addr.s6_addr[i] = ccoip_addr.inet.address.ipv6.data[i];
        }
    } else if (ccoip_addr.inet.protocol == inetIPv4) {
        return -1; // Unsupported protocol
    }
    return 0; // Success
}


int convert_from_sockaddr(const sockaddr *sock_addr, ccoip_socket_address_t *ccoip_addr) {
    if (sock_addr->sa_family == AF_INET) {
        // IPv4 conversion
        auto *addr_in = reinterpret_cast<const sockaddr_in *>(sock_addr);
        ccoip_addr->inet.protocol = inetIPv4;
        ccoip_addr->port = ntohs(addr_in->sin_port);
        ccoip_addr->inet.address.ipv4.data[0] = addr_in->sin_addr.s_addr & 0xFF;
        ccoip_addr->inet.address.ipv4.data[1] = (addr_in->sin_addr.s_addr >> 8) & 0xFF;
        ccoip_addr->inet.address.ipv4.data[2] = (addr_in->sin_addr.s_addr >> 16) & 0xFF;
        ccoip_addr->inet.address.ipv4.data[3] = (addr_in->sin_addr.s_addr >> 24) & 0xFF;
    } else if (sock_addr->sa_family == AF_INET6) {
        // IPv6 conversion
        auto *addr_in6 = reinterpret_cast<const sockaddr_in6 *>(sock_addr);
        ccoip_addr->inet.protocol = inetIPv6;
        ccoip_addr->port = ntohs(addr_in6->sin6_port);
        for (int i = 0; i < 16; i++) {
            ccoip_addr->inet.address.ipv6.data[i] = addr_in6->sin6_addr.s6_addr[i];
        }
    } else [[unlikely]] {
        return -1; // Unsupported protocol
    }
    return 0; // Success
}
