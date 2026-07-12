cmake_minimum_required(VERSION 3.18)
project(AeroHighPrecisionSim LANGUAGES CXX)

# Suppress warnings from Eigen's bundled Find*.cmake modules
cmake_policy(SET CMP0135 NEW)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(AEROSIM_USE_CUDA "Enable CUDA GPU support" ON)
option(AEROSIM_REAL_DOUBLE "Use double precision for Real type" OFF)
if(AEROSIM_REAL_DOUBLE)
    add_compile_definitions(AEROSIM_REAL_DOUBLE)
endif()

option(AEROSIM_USE_CGNS "Enable CGNS mesh import" OFF)

option(AEROSIM_COVERAGE "Enable code coverage flags (gcov, GCC/Clang only)" OFF)
if(AEROSIM_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
        message(STATUS "Coverage instrumentation enabled")
    else()
        message(WARNING "AEROSIM_COVERAGE requires GCC or Clang — MSVC not supported")
    endif()
endif()

option(AEROSIM_SANITIZE "Enable address+undefined behavior sanitizers (GCC/Clang only)" OFF)
if(AEROSIM_SANITIZE)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer")
    check_cxx_source_compiles("int main(){return 0;}" SANITIZE_WORKS)
    if(SANITIZE_WORKS)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
        add_compile_options(-fno-sanitize-recover=all)
        message(STATUS "ASAN+UBSAN instrumentation enabled")
    else()
        message(WARNING "AEROSIM_SANITIZE=ON but -fsanitize not supported by this toolchain — disabling")
    endif()
endif()

option(AEROSIM_USE_CCACHE "Use ccache to accelerate rebuilds" ON)
if(AEROSIM_USE_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        message(STATUS "ccache found: ${CCACHE_PROGRAM}")
    endif()
endif()

# ─── CUDA (optional) ─────────────────────────────────────────────
if(AEROSIM_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Hybrid")

    # GPU architecture detection
    set(CMAKE_CUDA_ARCHITECTURES "75;80;89;90;100" CACHE STRING
        "CUDA architectures to build for" FORCE)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

    if(AEROSIM_USE_CCACHE AND CCACHE_PROGRAM)
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    endif()
endif()

set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

option(BUILD_TESTING "Build test executables" ON)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen test suite" FORCE)
set(CMAKE_Fortran_COMPILER "FALSE" CACHE INTERNAL "Fortran compiler (disabled)" FORCE)
include(FetchContent)
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)
FetchContent_MakeAvailable(eigen)

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

# NRLMSISE-00 Fortran compilation
find_program(GFORTRAN gfortran)
if(GFORTRAN)
    set(NRLMSISE00_DEF "${CMAKE_BINARY_DIR}/nrlmsise00.def")
    set(NRLMSISE00_LIB "${CMAKE_BINARY_DIR}/nrlmsise00.lib")
    set(NRLMSISE00_DLL "${CMAKE_BINARY_DIR}/nrlmsise00.dll")
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
else()
    message(WARNING "gfortran not found — NRLMSISE-00 disabled")
endif()

if(AEROSIM_USE_CUDA)
    target_link_libraries(missile_cpu PUBLIC CUDA::cudart)
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
    if(GFORTRAN)
        add_dependencies(${TARGET} nrlmsise00)
        add_custom_command(TARGET ${TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/nrlmsise00.dll"
            $<TARGET_FILE_DIR:${TARGET}>)
    endif()
endfunction()

macro(add_cuda_executable NAME)
    if(AEROSIM_USE_CUDA)
        add_executable(${NAME} ${ARGN})
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
