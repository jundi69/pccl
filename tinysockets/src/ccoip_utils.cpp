#include "ccoip_utils.hpp"

#include <cstdio>
#include <cstring>
#include <arpa/inet.h>
#include <netinet/in.h>

int convert_to_uv_sockaddr(const ccoip_socket_address_t &ccoip_addr, sockaddr_in &sock_addr_out) {
    memset(&sock_addr_out, 0, sizeof(sock_addr_out));

    if (ccoip_addr.inet_address.protocol == inetIPv4) {
        // IPv4 conversion
        auto *addr_in = reinterpret_cast<sockaddr_in *>(&sock_addr_out);
        addr_in->sin_family = AF_INET;
        addr_in->sin_port = htons(ccoip_addr.port); // Convert port to network byte order
        addr_in->sin_addr.s_addr = ccoip_addr.inet_address.address.ipv4.data[0] |
                                   (ccoip_addr.inet_address.address.ipv4.data[1] << 8) |
                                   (ccoip_addr.inet_address.address.ipv4.data[2] << 16) |
                                   (ccoip_addr.inet_address.address.ipv4.data[3] << 24);
    } else if (ccoip_addr.inet_address.protocol == inetIPv6) {
        // IPv6 conversion
        auto *addr_in6 = reinterpret_cast<sockaddr_in6 *>(&sock_addr_out);
        addr_in6->sin6_family = AF_INET6;
        addr_in6->sin6_port = htons(ccoip_addr.port);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            addr_in6->sin6_addr.s6_addr[i] = ccoip_addr.inet_address.address.ipv6.data[i];
        }
    } else {
        return -1; // Unsupported protocol
    }

    return 0; // Success
}

int convert_from_uv_sockaddr(const sockaddr *sock_addr, ccoip_socket_address_t &ccoip_addr) {
    if (sock_addr->sa_family == AF_INET) {
        // IPv4 conversion
        auto *addr_in = reinterpret_cast<const sockaddr_in *>(sock_addr);
        ccoip_addr.inet_address.protocol = inetIPv4;
        ccoip_addr.port = ntohs(addr_in->sin_port);
        ccoip_addr.inet_address.address.ipv4.data[0] = addr_in->sin_addr.s_addr & 0xFF;
        ccoip_addr.inet_address.address.ipv4.data[1] = (addr_in->sin_addr.s_addr >> 8) & 0xFF;
        ccoip_addr.inet_address.address.ipv4.data[2] = (addr_in->sin_addr.s_addr >> 16) & 0xFF;
        ccoip_addr.inet_address.address.ipv4.data[3] = (addr_in->sin_addr.s_addr >> 24) & 0xFF;
    } else if (sock_addr->sa_family == AF_INET6) {
        // IPv6 conversion
        auto *addr_in6 = reinterpret_cast<const sockaddr_in6 *>(sock_addr);
        ccoip_addr.inet_address.protocol = inetIPv6;
        ccoip_addr.port = ntohs(addr_in6->sin6_port);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            ccoip_addr.inet_address.address.ipv6.data[i] = addr_in6->sin6_addr.s6_addr[i];
        }
    } else {
        return -1; // Unsupported protocol
    }
    return 0; // Success
}
