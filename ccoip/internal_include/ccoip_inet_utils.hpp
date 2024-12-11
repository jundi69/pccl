#pragma once

#include <ccoip_inet.h>
#include <cstring>
#include <string>
#include <sstream>

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#undef min
#else
#include <arpa/inet.h>
#endif

inline ccoip_ipv4_address_t from_octets(const uint8_t octet1, const uint8_t octet2, const uint8_t octet3,
                                        const uint8_t octet4) {
    ccoip_ipv4_address_t address;
    address.data[0] = octet1;
    address.data[1] = octet2;
    address.data[2] = octet3;
    address.data[3] = octet4;
    return address;
}

[[nodiscard]] inline std::string CCOIP_SOCKET_ADDR_TO_STRING(const ccoip_socket_address_t &addr) {
    std::ostringstream oss;
    if (addr.inet.protocol == inetIPv4) {
        oss << static_cast<int>(addr.inet.ipv4.data[0]) << "."
                << static_cast<int>(addr.inet.ipv4.data[1]) << "."
                << static_cast<int>(addr.inet.ipv4.data[2]) << "."
                << static_cast<int>(addr.inet.ipv4.data[3]) << ":"
                << ntohs((addr).port);
    } else if (addr.inet.protocol == inetIPv6) {
        char ip_str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &addr.inet.ipv6, ip_str, sizeof(ip_str));
        oss << ip_str << ":" << std::to_string(ntohs(addr.port));
    } else {
        oss << "Unknown Protocol";
    }
    return oss.str();
}

[[nodiscard]] inline std::string CCOIP_INET_ADDR_TO_STRING(const ccoip_inet_address_t &addr) {
    if (addr.protocol == inetIPv4) {
        std::ostringstream oss;
        oss << static_cast<int>(addr.ipv4.data[0]) << "."
                << static_cast<int>(addr.ipv4.data[1]) << "."
                << static_cast<int>(addr.ipv4.data[2]) << "."
                << static_cast<int>(addr.ipv4.data[3]);
        return oss.str();
    }
    if (addr.protocol == inetIPv6) {
        char ip_str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &addr.ipv6, ip_str, sizeof(ip_str));
        return ip_str;
    }
    return "Unknown Protocol";
}


struct internal_inet_address_t {
    ccoip_inet_protocol_t protocol;

    ccoip_ipv4_address_t ipv4;
    ccoip_ipv6_address_t ipv6;

    bool operator==(const internal_inet_address_t &rhs) const {
        if (protocol != rhs.protocol) {
            return false;
        }
        if (protocol == inetIPv4) {
            return memcmp(ipv4.data, rhs.ipv4.data, 4) == 0;
        }
        if (protocol == inetIPv6) {
            return memcmp(ipv6.data, rhs.ipv6.data, 16) == 0;
        }
        return false;
    }
};

template<>
struct std::hash<internal_inet_address_t> {
    std::size_t operator()(const internal_inet_address_t &inet_addr) const noexcept {
        std::size_t hash_value = 0;
        hash_value = hash_value * 31 + inet_addr.protocol;
        for (const auto &byte: inet_addr.ipv4.data) {
            hash_value = hash_value * 31 + byte;
        }
        for (const auto &byte: inet_addr.ipv6.data) {
            hash_value = hash_value * 31 + byte;
        }
        return hash_value;
    }
};

struct internal_inet_socket_address_t {
    internal_inet_address_t inet_address;
    uint16_t port;

    bool operator==(const internal_inet_socket_address_t &rhs) const {
        return inet_address == rhs.inet_address && port == rhs.port;
    }
};

template<>
struct std::hash<internal_inet_socket_address_t> {
    std::size_t operator()(const internal_inet_socket_address_t &inet_addr) const noexcept {
        std::size_t hash_value = 0;
        hash_value = hash_value * 31 + std::hash<internal_inet_address_t>{}(inet_addr.inet_address);
        hash_value = hash_value * 31 + inet_addr.port;
        return hash_value;
    }
};

inline internal_inet_address_t ccoip_inet_to_internal(const ccoip_inet_address_t &inet_addr) {
    internal_inet_address_t internal_addr{};
    internal_addr.protocol = inet_addr.protocol;
    internal_addr.ipv4 = inet_addr.ipv4;
    internal_addr.ipv6 = inet_addr.ipv6;
    return internal_addr;
}

inline internal_inet_socket_address_t ccoip_socket_to_internal(const ccoip_socket_address_t &socket_addr) {
    internal_inet_socket_address_t internal_socket{};
    internal_socket.inet_address = ccoip_inet_to_internal(socket_addr.inet);
    internal_socket.port = socket_addr.port;
    return internal_socket;
}

inline ccoip_inet_address_t internal_to_ccoip_inet(const internal_inet_address_t &internal_addr) {
    ccoip_inet_address_t inet_addr{};
    inet_addr.protocol = internal_addr.protocol;
    inet_addr.ipv4 = internal_addr.ipv4;
    inet_addr.ipv6 = internal_addr.ipv6;
    return inet_addr;
}

inline ccoip_socket_address_t internal_to_ccoip_sockaddr(const internal_inet_socket_address_t &internal_socket) {
    ccoip_socket_address_t socket_addr{};
    socket_addr.inet = internal_to_ccoip_inet(internal_socket.inet_address);
    socket_addr.port = internal_socket.port;
    return socket_addr;
}
