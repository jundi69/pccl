#pragma once

#include <ccoip_inet.h>

#ifdef WIN32
#include <winsock2.h>
#else
#include <netinet/in.h>
#endif

int convert_to_sockaddr(const ccoip_socket_address_t &ccoip_addr, sockaddr_in &sock_addr_out);

int convert_from_sockaddr(const sockaddr *sock_addr, ccoip_socket_address_t &ccoip_addr);