# Only cfd/test_cfd_gpu.cpp contains CUDA code and needs nvcc.
set_source_files_properties(cfd/test_cfd_gpu.cpp PROPERTIES LANGUAGE CUDA)

# --- GPU test targets (link missile_lib — requires device link) ---
add_cuda_executable(TestAeroViscous test_aero_viscous.cu)
add_cuda_executable(TestCfdGpu cfd/test_cfd_gpu.cpp)
add_cuda_executable(TestGpuTopology cfd/test_gpu_topology.cpp)
add_cuda_executable(TestCfdFuzz cfd/test_cfd_fuzz_gpu.cpp)

# --- CPU-only test targets (link missile_cpu — no device link) ---
add_cpu_executable(TestPropulsion test_propulsion.cpp)
add_cpu_executable(TestAutopilot test_autopilot.cpp)
add_cpu_executable(TestIntegrator test_integrator.cpp)
add_cpu_executable(TestCfdMesh cfd/test_cfd_mesh.cpp)
add_cpu_executable(TestCfdEuler cfd/test_cfd_euler.cpp)
add_cpu_executable(TestCfdDiagnostics cfd/test_cfd_diagnostics.cpp)
add_cpu_executable(TestCfdReconstruction cfd/test_cfd_reconstruction.cpp)
add_cpu_executable(TestCfdViscous cfd/test_cfd_viscous.cpp)
add_cpu_executable(TestCfdState cfd/test_cfd_state.cpp)
add_cpu_executable(TestCfdRans cfd/test_cfd_rans.cpp)
add_cpu_executable(TestMms cfd/test_mms.cpp)

# These test targets need symbols from .cu files (GPU-only implementations).
# Skip them in CPU-only builds.
if(AEROSIM_USE_CUDA)
    add_cpu_executable(TestAero test_aero.cpp)
    add_cpu_executable(TestGuidance test_guidance.cpp)
    add_cpu_executable(TestLaunch test_launch.cpp)
    add_cpu_executable(TestAeroTableGen test_aero_table_gen.cpp)
    add_cpu_executable(TestCompareAtm compare_atm.cpp)
endif()

function(add_test_with_cwd name target)
  add_test(NAME ${name} COMMAND ${target})
  set_tests_properties(${name} PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endfunction()

add_test_with_cwd(PropulsionTest TestPropulsion)
add_test_with_cwd(AutopilotTest TestAutopilot)
add_test_with_cwd(IntegratorTest TestIntegrator)
add_test_with_cwd(CfdMeshTest TestCfdMesh)
add_test_with_cwd(CfdEulerTest TestCfdEuler)
add_test_with_cwd(CfdDiagnosticsTest TestCfdDiagnostics)
add_test_with_cwd(CfdReconstructionTest TestCfdReconstruction)
add_test_with_cwd(CfdViscousTest TestCfdViscous)
add_test_with_cwd(CfdStateTest TestCfdState)
add_test_with_cwd(CfdRansTest TestCfdRans)
add_test_with_cwd(MmsTest TestMms)

if(AEROSIM_USE_CUDA)
    add_test_with_cwd(AeroTest TestAero)
    add_test_with_cwd(GuidanceTest TestGuidance)
    add_test_with_cwd(LaunchTest TestLaunch)
    add_test_with_cwd(AeroTableGenTest TestAeroTableGen)
    add_test_with_cwd(AeroViscousTest TestAeroViscous)
    add_test_with_cwd(CfdGpuTest TestCfdGpu)
    add_test_with_cwd(GpuTopologyTest TestGpuTopology)
    add_test_with_cwd(CfdFuzzTest TestCfdFuzz)
    add_test_with_cwd(CompareAtmTest TestCompareAtm)
endif()

# FP64 CPU oracle — pure C++, no CUDA
set(CPU_ORACLE_SOURCES
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/cfd_solver.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/cfd_residual.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/reconstruction.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/viscous.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/rans.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/diagnostics.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/mesh_metrics.cpp
    ${CMAKE_SOURCE_DIR}/src/aero/cfd/mesh_validator.cpp
)
add_executable(test_oracle_fp64
    cfd/test_oracle_fp64.cpp
    ${CPU_ORACLE_SOURCES}
)
target_compile_definitions(test_oracle_fp64 PRIVATE AEROSP_REAL_DOUBLE)
target_include_directories(test_oracle_fp64 PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(test_oracle_fp64 PRIVATE Eigen3::Eigen)
set_target_properties(test_oracle_fp64 PROPERTIES
    LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
