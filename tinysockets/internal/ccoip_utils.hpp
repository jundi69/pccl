#pragma once

#include <ccoip_inet.h>

#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netinet/in.h>
#endif

int convert_to_sockaddr_ipv4(const ccoip_socket_address_t &ccoip_addr, sockaddr_in *sock_addr_out);
int convert_to_sockaddr_ipv6(const ccoip_socket_address_t &ccoip_addr, sockaddr_in6 *sock_addr_out);

int convert_from_sockaddr(const sockaddr *sock_addr, ccoip_socket_address_t *ccoip_addr);