#ifndef KV_SSD_POSIX_H
#define KV_SSD_POSIX_H

#include <cstddef>
#include <cstdint>
#include <cstdarg>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstdio>

#include <BaseTsd.h>
typedef SSIZE_T ssize_t;

#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

static inline int portable_mkdir(const char * path, int /*mode*/) {
    return _mkdir(path);
}
#define mkdir(path, mode) portable_mkdir(path, mode)

static inline int portable_open(const char * path, int flags, ...) {
    flags |= _O_BINARY;
    if (flags & _O_CREAT) {
        va_list args;
        va_start(args, flags);
        int mode = va_arg(args, int);
        va_end(args);
        return _open(path, flags, mode);
    }
    return _open(path, flags);
}
#define open portable_open

static inline int portable_close(int fd) {
    return _close(fd);
}
#define close portable_close

static inline int portable_unlink(const char * path) {
    return _unlink(path);
}
#define unlink portable_unlink

static inline int portable_fsync(int fd) {
    return _commit(fd);
}
#define fsync portable_fsync

static inline ssize_t portable_pwrite(int fd, const void * buf, size_t count, off_t offset) {
    if (_lseek(fd, offset, SEEK_SET) == -1) return -1;
    int n = _write(fd, buf, (unsigned int)count);
    return (n < 0) ? (ssize_t)-1 : (ssize_t)n;
}
#define pwrite portable_pwrite

static inline ssize_t portable_pread(int fd, void * buf, size_t count, off_t offset) {
    if (_lseek(fd, offset, SEEK_SET) == -1) return -1;
    int n = _read(fd, buf, (unsigned int)count);
    return (n < 0) ? (ssize_t)-1 : (ssize_t)n;
}
#define pread portable_pread

#else
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#endif

#endif
