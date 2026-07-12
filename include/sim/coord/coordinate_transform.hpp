#pragma once

#include <Eigen/Core>
#include "infra/math/constants.hpp"
#include "infra/common.hpp"

namespace aerosp {

/**
 * @brief Geodetic coordinates (Latitude, Longitude, Altitude)
 */
struct LLA {
    double lat; // radians
    double lon; // radians
    double alt; // meters
};

class CoordinateTransform {
public:
    // LLA <-> ECEF
    static CUDA_HOST_DEVICE Eigen::Vector3d lla_to_ecef(const LLA& lla);
    static CUDA_HOST_DEVICE LLA ecef_to_lla(const Eigen::Vector3d& ecef);

    // ECEF <-> NED (at a given LLA reference point)
    static CUDA_HOST_DEVICE Eigen::Vector3d ecef_to_ned(const Eigen::Vector3d& ecef, const LLA& ref_lla);
    static CUDA_HOST_DEVICE Eigen::Vector3d ecef_to_ned_vector(const Eigen::Vector3d& ecef_vec, const LLA& ref_lla);
    static CUDA_HOST_DEVICE Eigen::Vector3d ned_to_ecef(const Eigen::Vector3d& ned, const LLA& ref_lla);

    // NED <-> Body (using Euler angles: roll, pitch, yaw)
    // angles in radians
    static CUDA_HOST_DEVICE Eigen::Matrix3d dcm_ned_to_body(double roll, double pitch, double yaw);
    static CUDA_HOST_DEVICE Eigen::Matrix3d dcm_body_to_ned(double roll, double pitch, double yaw);

    // Vector rotation using DCM
    static CUDA_HOST_DEVICE Eigen::Vector3d rotate_vector(const Eigen::Matrix3d& dcm, const Eigen::Vector3d& vec);

    // ECI <-> ECEF (Simplified: Earth rotation only, ignoring precession/nutation for now)
    static CUDA_HOST_DEVICE Eigen::Vector3d ecef_to_eci(const Eigen::Vector3d& ecef, double gmst);
    static CUDA_HOST_DEVICE Eigen::Vector3d eci_to_ecef(const Eigen::Vector3d& eci, double gmst);
};

} // namespace aerosp
