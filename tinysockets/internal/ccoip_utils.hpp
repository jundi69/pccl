#pragma once

#include <ccoip_inet.h>
#include <netinet/in.h>

int convert_to_uv_sockaddr(const ccoip_socket_address_t &ccoip_addr, sockaddr_in &sock_addr_out);

int convert_from_uv_sockaddr(const sockaddr *sock_addr, ccoip_socket_address_t &ccoip_addr);