#pragma once

#include <cstring>
#include <arpa/inet.h>

[[nodiscard]] inline std::string CCOIP_SOCKET_ADDR_TO_STRING(const ccoip_socket_address_t &addr) {
    std::ostringstream oss;
    if (addr.inet.protocol == inetIPv4) {
        oss << static_cast<int>(addr.inet.address.ipv4.data[0]) << "."
            << static_cast<int>(addr.inet.address.ipv4.data[1]) << "."
            << static_cast<int>(addr.inet.address.ipv4.data[2]) << "."
            << static_cast<int>(addr.inet.address.ipv4.data[3]) << ":"
            << ntohs((addr).port);
    } else if (addr.inet.protocol == inetIPv6) {
        char ip_str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &addr.inet.address.ipv6, ip_str, sizeof(ip_str));
        oss << ip_str << ":" << std::to_string(ntohs(addr.port));
    } else {
        oss << "Unknown Protocol";
    }
    return oss.str();
}

[[nodiscard]] inline std::string CCOIP_INET_ADDR_TO_STRING(const ccoip_inet_address_t &addr) {
    if (addr.protocol == inetIPv4) {
        std::ostringstream oss;
        oss << static_cast<int>(addr.address.ipv4.data[0]) << "."
            << static_cast<int>(addr.address.ipv4.data[1]) << "."
            << static_cast<int>(addr.address.ipv4.data[2]) << "."
            << static_cast<int>(addr.address.ipv4.data[3]);
        return oss.str();
    }
    if (addr.protocol == inetIPv6) {
        char ip_str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &addr.address.ipv6, ip_str, sizeof(ip_str));
        return ip_str;
    }
    return "Unknown Protocol";
}



struct internal_inet_address_t {
    ccoip_inet_protocol_t protocol;

    union {
        ccoip_ipv4_address_t ipv4;
        ccoip_ipv6_address_t ipv6;
    } address;

    bool operator==(const internal_inet_address_t &rhs) const {
        if (protocol != rhs.protocol) {
            return false;
        }
        if (protocol == inetIPv4) {
            return memcmp(address.ipv4.data, rhs.address.ipv4.data, 4) == 0;
        }
        if (protocol == inetIPv6) {
            return memcmp(address.ipv6.data, rhs.address.ipv6.data, 16) == 0;
        }
        return false;
    }
};

template<>
struct std::hash<internal_inet_address_t> {
    std::size_t operator()(const internal_inet_address_t &inet_addr) const noexcept {
        std::size_t hash_value = 0;
        hash_value = hash_value * 31 + inet_addr.protocol;
        for (const auto &byte: inet_addr.address.ipv4.data) {
            hash_value = hash_value * 31 + byte;
        }
        for (const auto &byte: inet_addr.address.ipv6.data) {
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
