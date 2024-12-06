add_executable(ccoip_master src/main.cpp)

target_link_libraries(ccoip_master PRIVATE ccoip_private ccoip_public)