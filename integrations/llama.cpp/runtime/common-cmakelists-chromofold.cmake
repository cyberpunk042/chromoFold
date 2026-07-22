# Append this block to llama.cpp's `common/CMakeLists.txt` (after the final
# `target_link_libraries(${TARGET} PUBLIC llama Threads::Threads)`), and configure with
# `-DGGML_CHROMOFOLD=ON`. First run `integrations/llama.cpp/runtime/build_kv_lib.sh` to produce the
# prebuilt `-fPIC` static lib. Also disable the auto-generated `include(cmake/chromofold-runtime.cmake)`
# line that `apply_runtime_patch.py` appends to the top-level CMakeLists (it hits CUDA::cudart scope
# errors; this prebuilt-lib path replaces it).

# ChromoFold compressed-KV engine (precompiled with nvcc) linked into llama-common so the graph-eval
# callback can drive it. Prebuilt static lib avoids enabling the CUDA language in this CMake scope.
if (GGML_CHROMOFOLD)
    find_package(CUDAToolkit)
    target_include_directories(${TARGET} PUBLIC
        /home/jfortin/chromoFold/include
        /home/jfortin/chromoFold/integrations/llama.cpp)
    target_compile_definitions(${TARGET} PUBLIC GGML_CHROMOFOLD)
    target_link_libraries(${TARGET} PUBLIC
        /home/jfortin/chromoFold/build/libchromofold_kv/libchromofold_kv.a)
    if (TARGET CUDA::cudart)
        target_link_libraries(${TARGET} PUBLIC CUDA::cudart)
    endif()
endif()
