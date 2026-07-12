target_sources(missile_cpu PRIVATE missile_config.cpp)
if(AEROSIM_USE_CUDA)
    target_sources(missile_lib PRIVATE missile_config.cpp)
endif()
