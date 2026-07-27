#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "sim/dynamics/dynamics_6dof.hpp"
#include "sim/gravity/gravity_model.hpp"
#include "sim/atmosphere/atmosphere_model.hpp"
#include "sim/coord/coordinate_transform.hpp"
#include "infra/math/constants.hpp"
#include "dart_config.hpp"
#include "dart_aero_table.hpp"
#include "sim/control/dart_guidance.hpp"

using namespace aerosp;
using namespace aerosp::dart;
using namespace aerosp::sim::control;

/**
 * @brief High-Fidelity Simulation for Dart
 * Uses the same 6-DOF dynamics, RK4 integrator, and environmental models
 * as the ballistic missile simulation, but with subsonic aerodynamic coefficients
 * from a lookup table (LUT) and closed-loop vision guidance.
 */
int main(int argc, char* argv[]) {
    std::cout << "--- aerosp: Dart High-Fidelity Simulator ---" << std::endl;

    // 1. Initial Configuration
    DartConfig config;
    auto resolve_path = [](const std::vector<std::string>& candidates) {
        for (const auto& candidate : candidates) {
            std::ifstream file(candidate);
            if (file.good()) {
                return candidate;
            }
        }
        return candidates.front();
    };
    std::string gravity_path = resolve_path({"data/EGM2008.gfc", "../data/EGM2008.gfc", "../data/EGM2008.gfc"});
    std::string aero_path = resolve_path({"data/dart/dart_aero_table.csv", "../data/dart/dart_aero_table.csv", "dart_aero_table.csv"});
    DartAeroTable aero_table(aero_path);
    if (!aero_table.is_loaded()) {
        std::cerr << "Failed to load dart aero table." << std::endl;
        if (!aero_table.is_loaded()) {
            std::cerr << "Error: Failed to load aero table. Using zero aerodynamics!" << std::endl;
        }
    }

    DartGuidance::Config guid_cfg;
    guid_cfg.nav_ratio = config.nav_ratio;
    guid_cfg.guidance_start_time = config.guid_start_time;
    guid_cfg.sensor_latency = config.guid_sensor_latency;
    guid_cfg.ctrl_gain = config.ctrl_gain;
    guid_cfg.ctrl_q_bias = config.ctrl_q_bias;
    guid_cfg.max_ctrl_coeff = config.max_ctrl_coeff;
    guid_cfg.terminal_range = config.guid_terminal_range;
    guid_cfg.update_hz = config.guid_update_hz;
    guid_cfg.min_tgo = config.guid_min_tgo;
    guid_cfg.max_accel = config.guid_max_accel;
    guid_cfg.visual_fov_deg = config.guid_fov_deg;
    guid_cfg.y_pos_gain = config.guid_y_pos_gain;
    guid_cfg.y_vel_gain = config.guid_y_vel_gain;
    guid_cfg.z_pos_gain = config.guid_z_pos_gain;
    guid_cfg.z_vel_gain = config.guid_z_vel_gain;
    guid_cfg.rate_damp = config.guid_rate_damp;
    guid_cfg.dp_gain = config.guid_dp_gain;
    guid_cfg.dy_gain = config.guid_dy_gain;
    guid_cfg.vz_gain = config.guid_vz_gain;
    guid_cfg.dp_step = config.guid_dp_step;
    guid_cfg.dy_step = config.guid_dy_step;
    guid_cfg.y_deadband = config.guid_y_deadband;
    guid_cfg.z_deadband = config.guid_z_deadband;
    guid_cfg.lookup_err_1 = config.guid_lookup_err_1;
    guid_cfg.lookup_err_2 = config.guid_lookup_err_2;
    guid_cfg.lookup_err_3 = config.guid_lookup_err_3;
    guid_cfg.lookup_err_4 = config.guid_lookup_err_4;
    guid_cfg.lookup_err_5 = config.guid_lookup_err_5;
    guid_cfg.lookup_dp_1 = config.guid_lookup_dp_1;
    guid_cfg.lookup_dp_2 = config.guid_lookup_dp_2;
    guid_cfg.lookup_dp_3 = config.guid_lookup_dp_3;
    guid_cfg.lookup_dp_4 = config.guid_lookup_dp_4;
    guid_cfg.lookup_dp_5 = config.guid_lookup_dp_5;
    guid_cfg.lookup_dp_6 = config.guid_lookup_dp_6;
    guid_cfg.gravity_mag = config.gravity_mag;
    DartGuidance guidance(guid_cfg);

    double initial_vel = 25.0;
    double pitch_deg = 6.81;   // Optimized pitch
    double azimuth_deg = 7.3;  // Target Base
    std::string output_path = "dart_trajectory.csv";
    bool silent = false;
    bool use_guidance = true; // NEW: Toggle guidance

    // Command line overrides
    if (argc >= 4) {
        initial_vel = std::atof(argv[1]);
        pitch_deg = std::atof(argv[2]);
        azimuth_deg = std::atof(argv[3]);
    }
    if (argc >= 5) {
        output_path = argv[4];
        silent = true; // Assume batch mode if output path provided
    }

    if (!silent) std::cout << "--- aerosp: Dart High-Fidelity Simulator ---" << std::endl;

    // 2. Initialize State
    State6DOF state;
    LLA origin_lla = {config.launch_lat * M_PI / 180.0,
                      config.launch_lon * M_PI / 180.0,
                      config.launch_alt};
    Eigen::Vector3d origin_ecef = CoordinateTransform::lla_to_ecef(origin_lla);

    // Compute Precise Gravity Magnitude (using EGM2008)
    GravityModel gravity_model(360);
    if (gravity_model.load_coefficients(gravity_path)) {
        Eigen::Vector3d g_vec = gravity_model.calculate_acceleration(origin_ecef);
        config.gravity_mag = g_vec.norm();
    } else {
        // Fallback to Somigliana formula (WGS84) if file missing
        double sin_lat = sin(origin_lla.lat);
        config.gravity_mag = 9.7803253359 * (1 + 0.00193185265241 * sin_lat * sin_lat) /
                             sqrt(1 - 0.00669437999014 * sin_lat * sin_lat);
    }

    // Compute Precise Atmosphere Baseline
    AtmosphereData atm_baseline = AtmosphereModel::calculate_ussa76(config.launch_alt);
    config.atm_density = atm_baseline.density;
    config.sound_speed = atm_baseline.sound_speed;

    if (argc >= 6) {
        use_guidance = (std::string(argv[5]) == "1" || std::string(argv[5]) == "true");
    }

    guid_cfg.gravity_mag = config.gravity_mag;
    guidance = DartGuidance(guid_cfg);

    if (!silent) {
        std::cout << "Location: (" << config.launch_lat << ", " << config.launch_lon << ")" << std::endl;
        std::cout << "Env: Gravity=" << config.gravity_mag << " m/s2, Density=" << config.atm_density << " kg/m3" << std::endl;
    }

    // Velocity Vector in NED
    double theta = pitch_deg * M_PI / 180.0;
    double psi = azimuth_deg * M_PI / 180.0;
    Eigen::Vector3d vel_ned(initial_vel * std::cos(theta) * std::cos(psi),
                            initial_vel * std::cos(theta) * std::sin(psi),
                            -initial_vel * std::sin(theta));

    // R_en (ECEF to NED)
    double sin_lat = sin(origin_lla.lat);
    double cos_lat = cos(origin_lla.lat);
    double sin_lon = sin(origin_lla.lon);
    double cos_lon = cos(origin_lla.lon);
    Eigen::Matrix3d R_en;
    R_en(0, 0) = -sin_lat * cos_lon; R_en(0, 1) = -sin_lat * sin_lon; R_en(0, 2) = cos_lat;
    R_en(1, 0) = -sin_lon;           R_en(1, 1) =  cos_lon;           R_en(1, 2) = 0.0;
    R_en(2, 0) = -cos_lat * cos_lon; R_en(2, 1) = -cos_lat * sin_lon; R_en(2, 2) = -sin_lat;
    Eigen::Matrix3d R_ne = R_en.transpose();

    // RELATIVE ECEF: Position (0,0,0) at origin
    state.pos_ecef = Eigen::Vector3d::Zero();
    state.vel_ecef = R_ne * vel_ned;

    // Initial Orientation (aligned with velocity vector)
    Eigen::AngleAxisd yaw_rot(psi, Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitch_rot(theta, Eigen::Vector3d::UnitY());
    Eigen::Quaterniond quat_nb = yaw_rot * pitch_rot;

    Eigen::Quaterniond q_ne(R_ne);
    state.quat_be = q_ne * quat_nb;
    state.omega_body.setZero();
    state.mass = config.mass;

    // 3. Simulation Loop
    double dt = 0.001; // Consistent with MC (1000Hz)
    double t = 0.0;

    // Constant gravity in NED
    Eigen::Vector3d g_ecef = R_ne * Eigen::Vector3d(0, 0, config.gravity_mag);

    InertialProps inertia_props;
    inertia_props.mass = config.mass;
    inertia_props.inertia = config.inertia;
    inertia_props.com = config.com;

    // Target Position (Base)
    Eigen::Vector3d target_ned = config.get_base(config.target_alt).pos_local;

    std::ofstream outfile(output_path);
    outfile << "Time,X,Y,Z,Vel,Pitch,Yaw,Alpha,Beta,Alt,p,q,r\n";

    while (t < 5.0) {
        Eigen::Vector3d pos_ned = R_en * state.pos_ecef;
        double current_alt = config.launch_alt - pos_ned.z();
        if (current_alt < 0.0 && t > 0.1) {
            if (!silent) std::cout << "Impact detected at t=" << t << "s" << std::endl;
            break;
        }

        AtmosphereData atm = atm_baseline;

        // B. Gravity (Already set as constant g_ecef)

        // C. Aerodynamics
        // v_body = quat_be^-1 * vel_ecef
        Eigen::Vector3d v_body = state.quat_be.inverse() * state.vel_ecef;
        double v_mag = v_body.norm();
        double alpha = std::atan2(v_body.z(), v_body.x());
        double beta = std::asin(v_body.y() / (v_mag + 1e-6));
        double mach = v_mag / (atm.sound_speed + 1e-6);

        // Get coefficients from LUT
        auto coeffs = aero_table.get_coeffs(mach, alpha * 180.0 / M_PI, beta * 180.0 / M_PI);

        // D. Guidance
        Eigen::Vector3d vel_ned = R_en * state.vel_ecef;

        DartGuidance::GuidanceOutput guid_out = {0, 0, false};
        if (use_guidance) {
            guid_out = guidance.update(t, pos_ned, vel_ned, state.omega_body, target_ned, dt);
        }

        double q_inf = 0.5 * atm.density * v_mag * v_mag;

        // Add control moments (Rudder effectiveness)
        // Heuristic: 1 deg rudder ~ 0.1 Cl/Cm/Cn increment at Mach 0.1
        double ctrl_m = guid_out.delta_pitch * config.ctrl_moment_coeff_pitch;
        double ctrl_n = guid_out.delta_yaw * config.ctrl_moment_coeff_yaw;

        Eigen::Vector3d force_aero_body(coeffs.CX * q_inf * config.ref_area,
                                         coeffs.CY * q_inf * config.ref_area,
                                         coeffs.CZ * q_inf * config.ref_area);

        // Add damping moments (from table)
        double Cl_p = -0.5;

        Eigen::Vector3d moment_aero_body(
            (coeffs.Cl + Cl_p * state.omega_body.x()) * q_inf * config.ref_area * config.ref_length,
            (coeffs.Cm + coeffs.Cmq * state.omega_body.y() + ctrl_m) * q_inf * config.ref_area * config.ref_length,
            (coeffs.Cn + coeffs.Cnr * state.omega_body.z() + ctrl_n) * q_inf * config.ref_area * config.ref_length
        );

        ForcesMoments fm;
        fm.force_body = force_aero_body;
        fm.moment_body = moment_aero_body;
        fm.mass_flow_rate = 0.0; // No propulsion for dart

        // D. Integration (RK4)
        auto system = [&](const aerosp::State6DOF& s, double /*time*/) {
            return Dynamics6DOF::compute_derivatives(s, fm, inertia_props, g_ecef, SimulationProfile::LOCAL_TACTICAL);
        };
        state = Dynamics6DOF::integrate_rk4(state, system, t, dt);
        state.normalize();

        // E. Logging
        outfile << t << "," << pos_ned.x() << "," << pos_ned.y() << "," << pos_ned.z() << ","
                << v_mag << "," << -theta*180/M_PI << "," << psi*180/M_PI << ","
                << alpha*180/M_PI << "," << beta*180/M_PI << "," << current_alt << ","
                << state.omega_body.x() << "," << state.omega_body.y() << "," << state.omega_body.z() << "\n";

        t += dt;
    }

    if (!silent) std::cout << "Simulation finished. Trajectory saved to " << output_path << std::endl;
    return 0;
}

