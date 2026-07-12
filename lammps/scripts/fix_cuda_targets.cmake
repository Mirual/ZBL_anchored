# Do not search for packages in base-conda: its protobuf/abseil get into linking via
# Caffe2Config (protobuf is optional for torch) and crash the process on load.
set(CMAKE_IGNORE_PREFIX_PATH "/path/to/anaconda3")  # TODO: set for your machine

# Stub for building deepmd op_pt without the CUDA toolkit: torch (CUDA build) references
# the imported target CUDA::nvrtc, but find_package(CUDAToolkit) without nvcc does not
# create it. The real library lives in the conda env deepmd — we create the target manually.
if(NOT TARGET CUDA::nvrtc)
  add_library(CUDA::nvrtc SHARED IMPORTED)
  set_target_properties(
    CUDA::nvrtc PROPERTIES IMPORTED_LOCATION
                           "/path/to/anaconda3/envs/deepmd/lib/libnvrtc.so.12")  # TODO: set for your machine
endif()

# The same trick as in deepmd master (DEEPMD_BYPASS_TORCH_CUDA_CHECK): if
# torch::cudart is already defined, Caffe2Config skips the CUDA toolkit search.
# CUDA symbols when linking the op library are resolved from torch's own libraries.
if(NOT TARGET torch::cudart)
  add_library(torch::cudart INTERFACE IMPORTED)
endif()

# Empty protobuf targets: Caffe2/public/protobuf.cmake accepts already
# existing new-style targets and skips the search. Linking the real
# protobuf is not needed — pip-torch carries it inside libtorch_cpu statically;
# the dynamic protobuf from base-conda crashed the process on op load.
if(NOT TARGET protobuf::libprotobuf)
  add_library(protobuf::libprotobuf INTERFACE IMPORTED)
endif()
if(NOT TARGET protobuf::protoc)
  add_executable(protobuf::protoc IMPORTED)
endif()
