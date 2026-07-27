add_library(aero_solver_c SHARED ../../src/aero/panel/aero_solver_c_api.cpp
    "${CMAKE_BINARY_DIR}/cuda_link_stub.cu")
target_compile_definitions(aero_solver_c PRIVATE AERO_SOLVER_EXPORTS)
target_link_libraries(aero_solver_c PRIVATE missile_lib)
target_include_directories(aero_solver_c PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${eigen_SOURCE_DIR}")
set_target_properties(aero_solver_c PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_copy_nrlmsise_dll(aero_solver_c)
