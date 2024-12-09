#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ccoip_inet_protocol_t {
    inetIPv4,
    inetIPv6
} ccoip_inet_protocol_t;

typedef struct ccoip_ipv4_address_t {
    uint8_t data[4];
} ccoip_ipv4_address_t;

typedef struct ccoip_ipv6_address_t {
    uint8_t data[16];
} ccoip_ipv6_address_t;

typedef struct ccoip_inet_address_t {
    ccoip_inet_protocol_t protocol;
    union {
        ccoip_ipv4_address_t ipv4;
        ccoip_ipv6_address_t ipv6;
    } address;
} ccoip_inet_address_t;

typedef struct ccoip_socket_address_t {
    ccoip_inet_address_t inet;
    uint16_t port;
} ccoip_socket_address_t;

#ifdef __cplusplus
}
#endif
