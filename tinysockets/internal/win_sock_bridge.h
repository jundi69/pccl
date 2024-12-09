#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

#ifdef WIN32
typedef long long int ssize_t;
#else
#include <sys/types.h>
#endif

#ifndef WIN32
#include <unistd.h>

inline void closesocket(const int socket_fd) {
    close(socket_fd);
}

inline int setsockoptvp(const int socket_fd, const int level, const int optname, const void *optval,
                        const socklen_t optlen) {
    return setsockopt(socket_fd, level, optname, optval, optlen);
}
inline ssize_t recvvp(const int socket_fd, void *buffer, const size_t length, const int flags) {
    return read(socket_fd, buffer, length);
}
#endif
#endif //SOCKET_UTILS_H
