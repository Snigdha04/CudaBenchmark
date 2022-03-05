

get_property(MISC_SRCS GLOBAL PROPERTY misc_srcs)


add_executable(matMul matrixMul2.cu)
target_include_directories(matMul PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(
  matMul
  PUBLIC  CONAN_PKG::abseil
          CONAN_PKG::boost
          CONAN_PKG::grpc
          CONAN_PKG::libtorch)

set_target_properties(matMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                                  CUDA_STANDARD 14
                                                  CUDA_STANDARD_REQUIRED ON
                                                  CUDA_EXTENSIONS OFF
                                                  CXX_STANDARD 14
                                                  CXX_STANDARD_REQUIRED ON
                                                  CXX_EXTENSIONS OFF)


