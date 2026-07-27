#include "sim/gravity/gravity_model.hpp"
#include <iostream>
#include <Eigen/Dense>

int main() {
    aerosp::GravityModel model(360);


    std::cout << "Loading EGM2008 coefficients..." << std::endl;
    if (!model.load_coefficients("data/EGM2008.gfc")) {
        std::cerr << "Failed to load coefficients!" << std::endl;
        return 1;
    }

    std::cout << "Preparing CUDA..." << std::endl;
    model.prepare_cuda();

    // Test point (Jiuquan)
    Eigen::Vector3d r_ecef(-1115745.0, 5082121.0, 3652876.0);


    std::cout << "Calculating CPU Gravity..." << std::endl;
    Eigen::Vector3d g_cpu = model.calculate_acceleration(r_ecef);


    std::cout << "Calculating GPU Gravity..." << std::endl;
    Eigen::Vector3d g_gpu = model.calculate_acceleration_cuda(r_ecef);

    std::cout << "CPU Gravity: " << g_cpu.transpose() << std::endl;
    std::cout << "GPU Gravity: " << g_gpu.transpose() << std::endl;


    double diff = (g_cpu - g_gpu).norm();
    std::cout << "Difference: " << diff << std::endl;

    if (diff < 1e-10) {
        std::cout << "Test PASSED: CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Test FAILED: CPU and GPU results differ significantly." << std::endl;
    }

    return 0;
}
