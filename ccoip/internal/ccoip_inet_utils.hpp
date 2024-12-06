#pragma once

#include <arpa/inet.h>

#define CCOIP_SOCKET_ADDR_TO_STRING(addr) \
    [&]() -> std::string { \
        std::ostringstream oss; \
        if ((addr).protocol == inetIPv4) { \
            oss << static_cast<int>((addr).address.ipv4.data[0]) << "." \
                << static_cast<int>((addr).address.ipv4.data[1]) << "." \
                << static_cast<int>((addr).address.ipv4.data[2]) << "." \
                << static_cast<int>((addr).address.ipv4.data[3]) << ":" \
                << ntohs((addr).port); \
        } else if ((addr).protocol == inetIPv6) { \
            char ip_str[INET6_ADDRSTRLEN]; \
            inet_ntop(AF_INET6, &addr.address.ipv6, ip_str, sizeof(ip_str)); \
            oss << ip_str << ":" << std::to_string(ntohs(addr.port)); \
        } else { \
            oss << "Unknown Protocol"; \
        } \
        return oss.str(); \
    }()

#define CCOIP_INET_ADDR_TO_STRING(addr) \
    [&]() -> std::string { \
        if ((addr).protocol == inetIPv4) { \
            std::ostringstream oss; \
            oss << static_cast<int>((addr).address.ipv4.data[0]) << "." \
                << static_cast<int>((addr).address.ipv4.data[1]) << "." \
                << static_cast<int>((addr).address.ipv4.data[2]) << "." \
                << static_cast<int>((addr).address.ipv4.data[3]); \
            return oss.str(); \
        } else if ((addr).protocol == inetIPv6) { \
            char ip_str[INET6_ADDRSTRLEN]; \
            inet_ntop(AF_INET6, &addr.address.ipv6, ip_str, sizeof(ip_str)); \
            return ip_str; \
        } else { \
            return "Unknown Protocol"; \
        } \
    }()
