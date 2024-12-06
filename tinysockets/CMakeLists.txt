add_library(tinysockets STATIC src/server_socket.cpp src/client_socket.cpp src/ccoip_utils.cpp)

target_include_directories(tinysockets PUBLIC include)
target_include_directories(tinysockets PRIVATE internal)
target_link_libraries(tinysockets PRIVATE ccoip_private)
target_link_libraries(tinysockets PRIVATE pccl_log)

target_link_libraries(tinysockets PRIVATE libuv::libuv)