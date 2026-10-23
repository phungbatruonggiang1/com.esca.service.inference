#include "buffer_allocator.hpp"

#include <fcntl.h>
#include <grp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "esca/inference/errors.hpp"
#include "logger.hpp"

namespace esca::inference {

SharedBuffer::SharedBuffer(std::string name, void *data, size_t size)
    : name_(std::move(name)), data_(data), size_(size)
{
}

SharedBuffer::~SharedBuffer()
{
    if (data_ && data_ != MAP_FAILED)
        ::munmap(data_, size_);
    ::shm_unlink(name_.c_str());
}

BufferAllocator::BufferAllocator(const std::string &group)
{
    if (const struct group *g = ::getgrnam(group.c_str()))
        gid_ = g->gr_gid;
    else
        ESCA_LOG_WARN("group \"%s\" does not exist: tensor buffers will be root-only",
                      group.c_str());
}

std::shared_ptr<SharedBuffer> BufferAllocator::allocate(const std::string &name, size_t bytes)
{
    // A zero-sized mapping is an error to mmap; a tensor with no elements
    // still gets a page so its name exists.
    const size_t size = bytes == 0 ? 1 : bytes;

    const int fd = ::shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
    if (fd < 0)
        raise(ErrorCode::BufferError, "shm_open " + name + ": " + std::strerror(errno));

    auto fail = [&](const char *what) {
        const std::string message = std::string(what) + " " + name + ": " + std::strerror(errno);
        ::close(fd);
        ::shm_unlink(name.c_str());
        raise(ErrorCode::BufferError, message);
    };

    // The umask may have taken the group bits; the mode is the contract.
    if (::fchmod(fd, 0660) != 0)
        fail("fchmod");
    if (gid_ != static_cast<gid_t>(-1) && ::fchown(fd, static_cast<uid_t>(-1), gid_) != 0)
        fail("fchown");
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0)
        fail("ftruncate");

    void *data = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED)
        fail("mmap");
    ::close(fd); // the mapping keeps the region; the name keeps it findable
    return std::make_shared<SharedBuffer>(name, data, size);
}

} // namespace esca::inference
