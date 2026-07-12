if(AEROSIM_USE_CUDA)
    target_sources(missile_lib PRIVATE
        atmosphere/atmosphere_model.cu
        coord/coordinate_transform.cu
        dynamics/dynamics_6dof.cu
        gravity/gravity_model.cu
    )
endif()
