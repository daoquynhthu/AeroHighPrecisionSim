add_cuda_executable(DartSim sim.cpp)
set_source_files_properties(sim.cpp PROPERTIES LANGUAGE CUDA)
add_cuda_executable(DartMC dart_mc.cu)
add_cuda_executable(DartDebug dart_debug.cu)
if(TARGET DartDebug)
    set_target_properties(DartDebug PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endif()
