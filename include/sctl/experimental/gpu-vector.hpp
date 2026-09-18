// Backend containers for gpu_tree: HostVector and DeviceVector, whose resize leaves trivial elements
// uninitialized, and DataView, the non-owning view GetData hands out.

#ifndef _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_
#define _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>
#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "sctl/common.hpp"
#include "sctl/morton.hpp"  // SCTL_GPU_HD
#include "sctl/experimental/device_scratch.hpp"  // is_device_vector_v

namespace gpu_tree {

namespace detail {
/// `std::allocator` whose default construction is a no-op, so `std::vector::resize` leaves trivial elements uninitialized.
template <class T> struct DefaultInitAllocator : std::allocator<T> {
  template <class U> struct rebind { using other = DefaultInitAllocator<U>; };
  template <class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) { ::new (static_cast<void*>(p)) U; }
  template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
/// Device blocks go back to a process-wide pool instead of to the driver: cudaMalloc costs a few microseconds, cudaFree synchronizes the whole device, and one tree build makes dozens of each at sizes that repeat exactly from build to build. Only an exact size match is reused, so a block is never handed out larger than asked for and the pool cannot drift into holding one oversized block per size; past the cap, blocks are released as before. Not thread-safe and single-device, the same assumptions device_scratch already makes.
inline std::vector<std::pair<void*, std::size_t>>& device_block_pool() {
  static std::vector<std::pair<void*, std::size_t>> pool;
  return pool;
}

inline void* device_block_alloc(std::size_t bytes) {
  auto& pool = device_block_pool();
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    if (it->second != bytes) continue;
    void* const p = it->first;
    pool.erase(it);
    return p;
  }
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
  return p;
}

inline void device_block_free(void* p, std::size_t bytes) {
  if (!p) return;
  constexpr std::size_t retain_max = std::size_t(1) << 30;
  auto& pool = device_block_pool();
  std::size_t held = bytes;
  for (const auto& b : pool) held += b.second;
  if (held > retain_max) {
    cudaFree(p);
    return;
  }
  pool.emplace_back(p, bytes);
}

/// `thrust::device_allocator` whose construct is a no-op (thrust's uninitialized_vector idiom), and whose storage comes from the pool above.
template <class T> struct DeviceUninitAllocator : thrust::device_allocator<T> {
  using pointer = thrust::device_ptr<T>;
  using size_type = std::size_t;
  template <class U> struct rebind { using other = DeviceUninitAllocator<U>; };
  SCTL_GPU_HD void construct(T*) {}
  pointer allocate(size_type n) {
    void* const p = device_block_alloc(n * sizeof(T));
    if (!p && n) throw thrust::system::detail::bad_alloc("DeviceUninitAllocator::allocate");
    return pointer(static_cast<T*>(p));
  }
  void deallocate(pointer p, size_type n) { device_block_free(thrust::raw_pointer_cast(p), n * sizeof(T)); }
};
}  // namespace detail

/// Host backend: `std::vector` that leaves new elements uninitialized, as `sctl::Vector` does. A class rather than an alias, so it deduces as the tree's container template parameter.
template <class T> class HostVector : public std::vector<T, detail::DefaultInitAllocator<T>> {
 public:
  using std::vector<T, detail::DefaultInitAllocator<T>>::vector;
};

/// Device backend: `thrust::device_vector` without value-initialization on resize.
template <class T> class DeviceVector : public thrust::device_vector<T, detail::DeviceUninitAllocator<T>> {
 public:
  using thrust::device_vector<T, detail::DeviceUninitAllocator<T>>::device_vector;
};

/// Non-owning view of a data set: `n` values at `ptr` in node order, with the iterator thrust dispatches on for the backend. Valid until the set is reallocated.
template <class T, template <class...> class DevVec> struct DataView {
  using value_type = T;
  using iterator = std::conditional_t<detail::is_device_vector_v<DevVec<char>>, thrust::device_ptr<T>, T*>;  // probed on DevVec<char>: T may be const, which no container holds
  T* ptr = nullptr;
  sctl::Long n = 0;
  T* data() const { return ptr; }
  sctl::Long size() const { return n; }
  iterator begin() const { return iterator(ptr); }
  iterator end() const { return iterator(ptr) + n; }
};

}  // namespace gpu_tree

#endif  // _SCTL_EXPERIMENTAL_GPU_VECTOR_HPP_
