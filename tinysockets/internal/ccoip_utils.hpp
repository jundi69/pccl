#pragma once

#include <ccoip_inet.h>

int convert_to_uv_sockaddr(const ccoip_socket_address_t& custom_addr, struct sockaddr_storage& uv_addr);

