#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

enum ccoip_inet_protocol_t
{
    inetIPv4,
    inetIPv6
};

struct ccoip_ipv4_address_t
{
    uint8_t data[4];
};

inline ccoip_ipv4_address_t from_octets(const uint8_t octet1, const uint8_t octet2, const uint8_t octet3, const uint8_t octet4)
{
    return (struct ccoip_ipv4_address_t){octet1, octet2, octet3, octet4};
}

struct ccoip_ipv6_address_t
{
    uint8_t data[16];
};


struct ccoip_socket_address_t {
    enum ccoip_inet_protocol_t protocol;
    union {
        struct ccoip_ipv4_address_t ipv4;
        struct ccoip_ipv6_address_t ipv6;
    } address;
    uint16_t port;
};