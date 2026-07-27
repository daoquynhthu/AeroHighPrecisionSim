set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen test suite" FORCE)
include(FetchContent)
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip)
FetchContent_MakeAvailable(eigen)
