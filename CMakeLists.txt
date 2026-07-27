cmake_minimum_required(VERSION 3.18)
project(AeroHighPrecisionSim LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(compiler_flags)

option(AEROSIM_USE_CUDA "Enable CUDA GPU support" ON)
option(AEROSIM_REAL_DOUBLE "Use double precision for Real type" OFF)
if(AEROSIM_REAL_DOUBLE)
    add_compile_definitions(AEROSIM_REAL_DOUBLE)
endif()

option(AEROSIM_USE_CGNS "Enable CGNS mesh import" OFF)

# ─── CUDA (optional) ─────────────────────────────────────────────
if(AEROSIM_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Hybrid")

    set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90;100" CACHE STRING
        "CUDA architectures to build for" FORCE)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -O3 -g")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "${CMAKE_CUDA_FLAGS_MINSIZEREL} -O1")

    # CUDA stub: ensures device linking is triggered for C++-only targets linking missile_lib
    set(CUDA_LINK_STUB "${CMAKE_BINARY_DIR}/cuda_link_stub.cu")
    file(WRITE "${CUDA_LINK_STUB}"
        "// Auto-generated CUDA stub for device linking\n"
        "__device__ int __cuda_link_stub = 0;\n")

    if(AEROSIM_USE_CCACHE AND CCACHE_PROGRAM)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

option(BUILD_TESTING "Build test executables" ON)
set(CMAKE_Fortran_COMPILER "FALSE" CACHE INTERNAL "Fortran compiler (disabled)" FORCE)

include(fetch_eigen)

# CPU-only static library (no CUDA device code; no device link needed for consumers)
# Subdirectories add .cpp sources via target_sources(missile_cpu PRIVATE ...)
add_library(missile_cpu STATIC "")
target_include_directories(missile_cpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(missile_cpu PUBLIC Eigen3::Eigen)

# GPU static library (has .cu sources; requires separable compilation)
# Subdirectories add .cu sources via target_sources(missile_lib PRIVATE ...)
if(AEROSIM_USE_CUDA)
    add_library(missile_lib STATIC "")
    target_include_directories(missile_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
    set_target_properties(missile_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(missile_lib PUBLIC missile_cpu)
endif()

# NRLMSISE-00 stable location (outside volatile build directory)
set(NRLMSISE00_DIR "${CMAKE_SOURCE_DIR}/data/NRLMSISE-00")
set(NRLMSISE00_DEF "${NRLMSISE00_DIR}/nrlmsise00.def")
set(NRLMSISE00_LIB "${NRLMSISE00_DIR}/nrlmsise00.lib")
set(NRLMSISE00_DLL "${NRLMSISE00_DIR}/nrlmsise00.dll")

# If pre-built files already exist, skip Fortran compilation
if(EXISTS "${NRLMSISE00_LIB}" AND EXISTS "${NRLMSISE00_DLL}")
    message(STATUS "NRLMSISE-00: using pre-built library from ${NRLMSISE00_DIR}")
    add_custom_target(nrlmsise00 DEPENDS "${NRLMSISE00_LIB}" "${NRLMSISE00_DLL}")
    target_link_libraries(missile_cpu PUBLIC "${NRLMSISE00_LIB}")
    target_compile_definitions(missile_cpu PRIVATE AEROSP_HAS_NRLMSISE00=1)
else()
    find_program(GFORTRAN gfortran)
    if(GFORTRAN)
        message(STATUS "NRLMSISE-00: building from Fortran source")
        add_custom_command(OUTPUT "${NRLMSISE00_DLL}" "${NRLMSISE00_DEF}"
            COMMAND ${GFORTRAN} -shared -o "${NRLMSISE00_DLL}"
                "${CMAKE_SOURCE_DIR}/src/sim/atmosphere/nrlmsise00.f"
                -Wl,--output-def,"${NRLMSISE00_DEF}"
                -static -std=legacy -Wno-unused-variable
            DEPENDS "${CMAKE_SOURCE_DIR}/src/sim/atmosphere/nrlmsise00.f"
            COMMENT "Compiling NRLMSISE-00 (Fortran)...")
        if(MSVC)
            add_custom_command(OUTPUT "${NRLMSISE00_LIB}"
                COMMAND ${CMAKE_AR} /def:"${NRLMSISE00_DEF}" /out:"${NRLMSISE00_LIB}" /machine:x64
                DEPENDS "${NRLMSISE00_DEF}"
                COMMENT "Generating MSVC NRLMSISE-00 import library...")
        else()
            find_program(DLLTOOL dlltool)
            if(DLLTOOL)
                add_custom_command(OUTPUT "${NRLMSISE00_LIB}"
                    COMMAND ${DLLTOOL} --def "${NRLMSISE00_DEF}" --dllname "nrlmsise00.dll" --output-lib "${NRLMSISE00_LIB}"
                    DEPENDS "${NRLMSISE00_DEF}"
                    COMMENT "Generating MinGW NRLMSISE-00 import library...")
            else()
                set(NRLMSISE00_LIB "${NRLMSISE00_DLL}")
                message(WARNING "dlltool not found — linking NRLMSISE-00 DLL directly")
            endif()
        endif()
        add_custom_target(nrlmsise00 DEPENDS "${NRLMSISE00_LIB}" "${NRLMSISE00_DLL}")
        add_dependencies(missile_cpu nrlmsise00)
        if(AEROSIM_USE_CUDA)
            add_dependencies(missile_lib nrlmsise00)
        endif()
        target_link_libraries(missile_cpu PUBLIC "${NRLMSISE00_LIB}")
        target_compile_definitions(missile_cpu PRIVATE AEROSP_HAS_NRLMSISE00=1)
    else()
        message(WARNING "gfortran not found — NRLMSISE-00 disabled")
        target_compile_definitions(missile_cpu PRIVATE AEROSP_HAS_NRLMSISE00=0)
    endif()
endif()

if(AEROSIM_USE_CUDA)
    target_link_libraries(missile_cpu PUBLIC CUDA::cudart)
    target_compile_definitions(missile_cpu PUBLIC AEROSP_HAS_CUDA=1)
endif()

if(AEROSIM_USE_CGNS)
    find_package(CGNS)
    if(CGNS_FOUND)
        add_compile_definitions(WITH_CGNS)
        target_include_directories(missile_cpu PUBLIC ${CGNS_INCLUDE_DIRS})
        target_link_libraries(missile_cpu PUBLIC ${CGNS_LIBRARIES})
        message(STATUS "CGNS found: ${CGNS_INCLUDE_DIRS}")
    else()
        message(WARNING "AEROSIM_USE_CGNS is ON but CGNS not found — CGNS reader disabled")
    endif()
endif()

option(AEROSIM_MPI "Enable MPI distributed memory support" OFF)
if(AEROSIM_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_include_directories(missile_cpu PUBLIC ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(missile_cpu PUBLIC ${MPI_CXX_LIBRARIES})
    add_compile_definitions(MPI_ENABLED)
    message(STATUS "MPI found: ${MPI_CXX_INCLUDE_DIRS}")
endif()

if(MSVC)
    target_compile_options(missile_cpu PUBLIC $<$<COMPILE_LANGUAGE:CXX>:/wd4819 /wd4068>)
    if(AEROSIM_USE_CUDA)
        target_compile_options(missile_lib PUBLIC
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4819 -Xcompiler=/wd4068 -Xcudafe --diag_suppress=20012>)
    endif()
endif()

function(target_copy_nrlmsise_dll TARGET)
    if(TARGET nrlmsise00)
        add_dependencies(${TARGET} nrlmsise00)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${NRLMSISE00_DLL}"
            $<TARGET_FILE_DIR:${TARGET}>)
    endif()
endfunction()

macro(add_cuda_executable NAME)
    if(AEROSIM_USE_CUDA)
        add_executable(${NAME} ${ARGN} "${CUDA_LINK_STUB}")
        target_link_libraries(${NAME} PRIVATE missile_lib)
        set_target_properties(${NAME} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
        target_copy_nrlmsise_dll(${NAME})
    else()
        message(STATUS "CUDA disabled — skipping target ${NAME}")
    endif()
endmacro()

macro(add_cpu_executable NAME)
    add_executable(${NAME} ${ARGN})
    target_link_libraries(${NAME} PRIVATE missile_cpu)
    set_target_properties(${NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    target_copy_nrlmsise_dll(${NAME})
endmacro()

# yaml-cpp (config file parsing)
if(EXISTS "${CMAKE_SOURCE_DIR}/third-party/yaml-cpp/CMakeLists.txt")
    add_subdirectory(third-party/yaml-cpp third-party/yaml-cpp EXCLUDE_FROM_ALL)
    target_link_libraries(missile_cpu PRIVATE yaml-cpp)
    target_compile_definitions(missile_cpu PRIVATE AEROSP_HAS_YAML_CPP=1)
endif()

# Library subdirectories (add sources to missile_lib via target_sources)
add_subdirectory(src/infra)
add_subdirectory(src/sim)
add_subdirectory(src/aero)
add_subdirectory(src/config)

# Executables
add_subdirectory(app)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
add_subdirectory(examples)

# --- Install rules ---
include(GNUInstallDirs)
if(AEROSIM_USE_CUDA)
    set(_MISSILE_TARGETS missile_cpu missile_lib)
else()
    set(_MISSILE_TARGETS missile_cpu)
endif()
if(TARGET yaml-cpp)
    list(APPEND _MISSILE_TARGETS yaml-cpp)
endif()
install(TARGETS ${_MISSILE_TARGETS}
    EXPORT AeroSimTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp")
install(EXPORT AeroSimTargets
    FILE AeroSimTargets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/AeroSim)
