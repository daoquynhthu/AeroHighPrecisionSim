#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include "sim/dynamics/dynamics_6dof.hpp"
#include "sim/coord/coordinate_transform.hpp"
#include "sim/atmosphere/atmosphere_model.hpp"
#include "dart_aero_table.hpp"
#include "dart_config.hpp"
#include "sim/control/dart_guidance.hpp"
#include "sim/gravity/gravity_model.hpp"
#include "infra/math/constants.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace aerosp;

void run_debug_trajectory(double v0, double pitch_deg, double yaw_deg, double nav_ratio, double ctrl_gain, double forced_dp, double forced_mom_y) {
    std::ofstream log("dart_debug_log.csv");
    log << "Time,X,Y,Z,Vx,Vy,Vz,Qw,Qx,Qy,Qz,p,q,r,Alpha,Mach,CX,CY,CZ,Cm,Cn,dp,dy,ForceX,ForceY,ForceZ,MomX,MomY,MomZ\n";

    dart::DartConfig dart_cfg;
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

    // 1. Precise Environment Initialization (Based on LLA: 22.58, 113.96)
    LLA origin_lla = {dart_cfg.launch_lat * M_PI / 180.0,
                      dart_cfg.launch_lon * M_PI / 180.0,
                      dart_cfg.launch_alt};
    Eigen::Vector3d origin_ecef = CoordinateTransform::lla_to_ecef(origin_lla);

    // Compute Precise Gravity Magnitude (using EGM2008)
    GravityModel gravity_model(360);
    if (gravity_model.load_coefficients(gravity_path)) {
        Eigen::Vector3d g_vec = gravity_model.calculate_acceleration(origin_ecef);
        dart_cfg.gravity_mag = g_vec.norm();
    } else {
        // Fallback to Somigliana formula (WGS84) if file missing
        double sin_lat = sin(origin_lla.lat);
        dart_cfg.gravity_mag = 9.7803253359 * (1 + 0.00193185265241 * sin_lat * sin_lat) /
                               sqrt(1 - 0.00669437999014 * sin_lat * sin_lat);
    }

    // Compute Precise Atmosphere Baseline
    AtmosphereData atm_baseline = AtmosphereModel::calculate_ussa76(dart_cfg.launch_alt);
    dart_cfg.atm_density = atm_baseline.density;
    dart_cfg.sound_speed = atm_baseline.sound_speed;
    dart_cfg.nav_ratio = nav_ratio;
    dart_cfg.ctrl_gain = ctrl_gain;

    sim::control::DartGuidance::Config guid_cfg;
    guid_cfg.nav_ratio = dart_cfg.nav_ratio;
    guid_cfg.guidance_start_time = dart_cfg.guid_start_time;
    guid_cfg.visual_fov_deg = dart_cfg.guid_fov_deg;
    guid_cfg.sensor_latency = dart_cfg.guid_sensor_latency;
    guid_cfg.ctrl_gain = dart_cfg.ctrl_gain;
    guid_cfg.ctrl_q_bias = dart_cfg.ctrl_q_bias;
    guid_cfg.max_ctrl_coeff = dart_cfg.max_ctrl_coeff;
    guid_cfg.terminal_range = dart_cfg.guid_terminal_range;
    guid_cfg.update_hz = dart_cfg.guid_update_hz;
    guid_cfg.min_tgo = dart_cfg.guid_min_tgo;
    guid_cfg.max_accel = dart_cfg.guid_max_accel;
    guid_cfg.y_pos_gain = dart_cfg.guid_y_pos_gain;
    guid_cfg.y_vel_gain = dart_cfg.guid_y_vel_gain;
    guid_cfg.z_pos_gain = dart_cfg.guid_z_pos_gain;
    guid_cfg.z_vel_gain = dart_cfg.guid_z_vel_gain;
    guid_cfg.rate_damp = dart_cfg.guid_rate_damp;
    guid_cfg.dp_gain = dart_cfg.guid_dp_gain;
    guid_cfg.dy_gain = dart_cfg.guid_dy_gain;
    guid_cfg.vz_gain = dart_cfg.guid_vz_gain;
    guid_cfg.dp_step = dart_cfg.guid_dp_step;
    guid_cfg.dy_step = dart_cfg.guid_dy_step;
    guid_cfg.y_deadband = dart_cfg.guid_y_deadband;
    guid_cfg.z_deadband = dart_cfg.guid_z_deadband;
    guid_cfg.gravity_mag = dart_cfg.gravity_mag;

    std::cout << "Starting Debug Simulation..." << std::endl;
    std::cout << "Location: (" << dart_cfg.launch_lat << ", " << dart_cfg.launch_lon << ")" << std::endl;
    std::cout << "Env: Gravity=" << dart_cfg.gravity_mag << " m/s2, Density=" << dart_cfg.atm_density << " kg/m3" << std::endl;

    double sin_lat = sin(origin_lla.lat);
    double cos_lat = cos(origin_lla.lat);
    double sin_lon = sin(origin_lla.lon);
    double cos_lon = cos(origin_lla.lon);

    Eigen::Matrix3d R_en; // ECEF to NED
    R_en(0, 0) = -sin_lat * cos_lon; R_en(0, 1) = -sin_lat * sin_lon; R_en(0, 2) = cos_lat;
    R_en(1, 0) = -sin_lon;           R_en(1, 1) =  cos_lon;           R_en(1, 2) = 0.0;
    R_en(2, 0) = -cos_lat * cos_lon; R_en(2, 1) = -cos_lat * sin_lon; R_en(2, 2) = -sin_lat;

    Eigen::Matrix3d R_ne = R_en.transpose();

    Eigen::Vector3d target_ned = dart_cfg.get_base(dart_cfg.target_alt).pos_local;

    double pitch = pitch_deg * M_PI / 180.0;
    double yaw = yaw_deg * M_PI / 180.0;

    State6DOF state;
    state.pos_ecef = Eigen::Vector3d::Zero(); // Relative to origin
    state.mass = dart_cfg.mass;

    Eigen::Vector3d v_ned(v0 * cos(pitch) * cos(yaw), v0 * cos(pitch) * sin(yaw), -v0 * sin(pitch));
    state.vel_ecef = R_ne * v_ned;

    // Initial orientation: body points along velocity in NED
    // Rotation sequence: Yaw (psi) then Pitch (theta)
    // In NED, positive rotation around Y is Pitch UP.
    Eigen::Quaterniond q_yaw_quat(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_pitch_quat(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond q_nb = q_yaw_quat * q_pitch_quat;

    // Body to ECEF: q_be = q_ne * q_nb
    Eigen::Quaterniond q_ne(R_ne);
    state.quat_be = q_ne * q_nb;
    state.quat_be.normalize();
    state.omega_body.setZero();

    InertialProps inertia;
    inertia.mass = dart_cfg.mass;
    inertia.inertia = dart_cfg.inertia;
    inertia.com = dart_cfg.com;

    aerosp::dart::DartAeroTable aero_table(aero_path);

    double t = 0;
    double dt = 0.001;
    double dp = 0, dy = 0;
    sim::control::DartGuidance::GuidanceState guid_state;
    sim::control::DartGuidance::reset_state(guid_state);

    std::cout << "Starting Debug Simulation..." << std::endl;
    // Location log already done above

    while (t < 5.0) {
        // Use NED for local altitude
        Eigen::Vector3d p_ned = R_en * state.pos_ecef;
        double alt = dart_cfg.launch_alt - p_ned.z();

        if (alt < -0.5 && t > 0.1) break;
        if (p_ned.x() > target_ned.x() + 5.0) break;

        // Guidance Step
        Eigen::Vector3d v_ned_cur = R_en * state.vel_ecef;
        dp = forced_dp; dy = 0;
        if (std::abs(forced_dp) < 1e-12) {
            auto guid_out = sim::control::DartGuidance::update_closed_loop(
                t, p_ned, v_ned_cur, state.omega_body, target_ned, dt, guid_cfg, guid_state
            );
            if (guid_out.active) {
                dp = guid_out.delta_pitch;
                dy = guid_out.delta_yaw;
            }
        }

        Eigen::Vector3d g_ecef = R_ne * Eigen::Vector3d(0, 0, dart_cfg.gravity_mag);

        auto system = [&](const State6DOF& s, double cur_t) {
            AtmosphereData atm_cur = atm_baseline;

            Eigen::Vector3d v_b = s.quat_be.inverse() * s.vel_ecef;
            double v_m = v_b.norm();
            double a_deg = atan2(v_b.z(), v_b.x()) * 180.0 / M_PI;
            double m = v_m / (atm_cur.sound_speed + 1e-6);

            auto c = aero_table.get_coeffs(m, a_deg);
            double q = 0.5 * atm_cur.density * v_m * v_m;

            // Use config values
            double S = dart_cfg.ref_area;
            double L = dart_cfg.ref_length;
            Eigen::Vector3d f_b(c.CX * q * S, c.CY * q * S, c.CZ * q * S);
            Eigen::Vector3d m_b(
                (-0.5 * s.omega_body.x()) * q * S * L,
                (c.Cm + c.Cmq * s.omega_body.y() + dp * dart_cfg.ctrl_moment_coeff_pitch) * q * S * L + forced_mom_y,
                (c.Cn + c.Cnr * s.omega_body.z() + dy * dart_cfg.ctrl_moment_coeff_yaw) * q * S * L
            );

            if (std::abs(cur_t - t) < 1e-9) {
                log << std::fixed << std::setprecision(6)
                    << t << "," << p_ned.x() << "," << p_ned.y() << "," << p_ned.z() << ","
                    << v_ned_cur.x() << "," << v_ned_cur.y() << "," << v_ned_cur.z() << ","
                    << s.quat_be.w() << "," << s.quat_be.x() << "," << s.quat_be.y() << "," << s.quat_be.z() << ","
                    << s.omega_body.x() << "," << s.omega_body.y() << "," << s.omega_body.z() << ","
                    << a_deg << "," << m << "," << c.CX << "," << c.CY << "," << c.CZ << "," << c.Cm << "," << c.Cn << ","
                    << dp << "," << dy << "," << f_b.x() << "," << f_b.y() << "," << f_b.z() << ","
                    << m_b.x() << "," << m_b.y() << "," << m_b.z() << "\n";
            }

            ForcesMoments fm{f_b, m_b, 0.0};
            return Dynamics6DOF::compute_derivatives(s, fm, inertia, g_ecef, SimulationProfile::LOCAL_TACTICAL);
        };

        state = Dynamics6DOF::integrate_rk4(state, system, t, dt);
        t += dt;

        if (std::isnan(state.pos_ecef.x()) || std::isinf(state.pos_ecef.x())) {
            std::cout << "NaN/Inf detected at t=" << t << std::endl;
            break;
        }
    }
    log.close();
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cout << "Usage: DartDebug <v0> <pitch_deg> <yaw_deg> <nav_ratio> <ctrl_gain> [forced_dp] [forced_mom_y]" << std::endl;
        return -1;
    }
    double v0 = std::stod(argv[1]);
    double pitch_deg = std::stod(argv[2]);
    double yaw_deg = std::stod(argv[3]);
    double nav_ratio = std::stod(argv[4]);
    double ctrl_gain = std::stod(argv[5]);
    double forced_dp = (argc > 6) ? std::stod(argv[6]) : 0.0;
    double forced_mom_y = (argc > 7) ? std::stod(argv[7]) : 0.0;

    std::cout << "Starting Debug Simulation..." << std::endl;
    run_debug_trajectory(v0, pitch_deg, yaw_deg, nav_ratio, ctrl_gain, forced_dp, forced_mom_y);
    std::cout << "Debug Simulation Finished. Log saved to dart_debug_log.csv" << std::endl;
    return 0;
}

