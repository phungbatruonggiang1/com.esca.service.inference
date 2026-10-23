// Tensor buffers in POSIX shared memory.
//
// The buffer allocator. Each tensor of each context gets a named region
// the service maps and the client maps too, so input pixels and output
// scores never cross D-Bus: the bus only carries the names.

#pragma once

#include <sys/types.h>

#include <cstddef>
#include <memory>
#include <string>

namespace esca::inference {

/// One mapped region. Unmapped and unlinked when the last reference goes,
/// which may be after its context is destroyed if an inference was still
/// writing into it.
class SharedBuffer {
public:
    SharedBuffer(std::string name, void *data, size_t size);
    ~SharedBuffer();
    SharedBuffer(const SharedBuffer &) = delete;
    SharedBuffer &operator=(const SharedBuffer &) = delete;

    const std::string &name() const { return name_; }
    void *data() const { return data_; }
    size_t size() const { return size_; }

private:
    std::string name_;
    void *data_;
    size_t size_;
};

class BufferAllocator {
public:
    /// Regions are created 0660 and given to `group`; root and that group
    /// can map them. An unknown group leaves them root-only, with a warning.
    explicit BufferAllocator(const std::string &group);

    /// Throws MemoryException (BufferError) when the region cannot be made.
    std::shared_ptr<SharedBuffer> allocate(const std::string &name, size_t bytes);

private:
    gid_t gid_ = static_cast<gid_t>(-1);
};

} // namespace esca::inference
