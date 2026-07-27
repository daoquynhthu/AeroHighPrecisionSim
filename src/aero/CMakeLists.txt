# CPU sources (plain C++17 — no CUDA device code)
target_sources(missile_cpu PRIVATE
    panel/aero_table_gen.cpp
    cfd/amr_refine.cpp
    cfd/amr_hanging.cpp
    cfd/amr_interpolate.cpp
    cfd/amr_sensor.cpp
    cfd/cfd_residual.cpp
    cfd/cfd_solver.cpp
    cfd/diagnostics.cpp
    cfd/mesh_gen_stl.cpp
    cfd/mesh_metrics.cpp
    cfd/mesh_io_su2.cpp
    cfd/mesh_io_cgns.cpp
    cfd/mesh_validator.cpp
    cfd/mms.cpp
    cfd/rans.cpp
    cfd/reconstruction.cpp
    cfd/thermo.cpp
    cfd/thermo_db.cpp
    cfd/gpu_thermo_table.cpp
    cfd/nasa9_eval.cpp
    cfd/transport_eval.cpp
    cfd/t_from_e.cpp
    cfd/viscous.cpp
)

if(AEROSIM_USE_CUDA)
    # GPU host code (.cpp using CUDA APIs, compiled by host compiler)
    target_sources(missile_cpu PRIVATE
        cfd/cuda_utils.cpp
        cfd/gpu_communicator.cpp
        cfd/gpu_topology.cpp
        cfd/partition.cpp
        cfd/cfd_solver_gpu.cpp
        cfd/device_mesh.cpp
    )

    # GPU device code (compiled by nvcc, needs separable compilation)
    target_sources(missile_lib PRIVATE
        panel/aero_solver.cu
        cfd/cfd_residual_gpu.cu
        cfd/gpu_thermo_table.cu
        cfd/exchange_halo.cu
        cfd/fgmres_gpu.cu
        cfd/jacobian_free.cu
        cfd/lusgs_gpu.cu
        cfd/gpu_diagnostics.cu
        cfd/gpu_diagnostics.cu
        cfd/gpu_ddes.cu
        cfd/gpu_rans.cu
        cfd/gpu_sst.cu
        cfd/gpu_solver.cu
        cfd/gpu_timestep.cu
        cfd/gpu_update.cu
        cfd/gpu_viscous.cu
        cfd/gpu_wall.cu
        cfd/reconstruction_gpu.cu
    )
endif()
