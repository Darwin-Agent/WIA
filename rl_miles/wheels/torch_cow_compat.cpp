#include <dlfcn.h>

namespace c10 {
class StorageImpl;
namespace impl {
namespace cow {
void materialize_cow(StorageImpl*);
}
}
}

namespace at {
class TensorBase;
namespace cuda {
void* getCurrentCUDABlasHandle(bool);
}
}

// Prebuilt FA2/TE/sgl-kernel wheels built against a slightly different torch
// commit expect the old TensorBase::const_data_ptr<T, 0>() ABI.  Current torch
// only exports the non-templated-dim overload, so forward it.
extern "C" const int* _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v(
    const at::TensorBase* self) {
  using Fn = const int* (*)(const at::TensorBase*);
  static Fn fn = []() -> Fn {
    return reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "_ZNK2at10TensorBase14const_data_ptrIiEEPKT_v"));
  }();
  return fn ? fn(self) : nullptr;
}

extern "C" void _ZN3c104impl3cow23materialize_cow_storageERNS_11StorageImplE(
    c10::StorageImpl* storage) {
  using Fn = void (*)(c10::StorageImpl*);
  static Fn fn = []() -> Fn {
    return reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "_ZN3c104impl3cow15materialize_cowEPNS_11StorageImplE"));
  }();
  if (fn) {
    fn(storage);
  }
}

extern "C" void* _ZN2at4cuda24getCurrentCUDABlasHandleEv() {
  using Fn = void* (*)(bool);
  static Fn fn = []() -> Fn {
    return reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "_ZN2at4cuda24getCurrentCUDABlasHandleEb"));
  }();
  return fn ? fn(false) : nullptr;
}
