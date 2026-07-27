#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "sim/dynamics/dynamics_6dof.hpp"
#include "sim/coord/coordinate_transform.hpp"
#include "infra/math/constants.hpp"
#include "dart_config.hpp"
#include "dart_aero_table.hpp"
#include "sim/atmosphere/atmosphere_model.hpp"
#include "sim/gravity/gravity_model.hpp"
#include "sim/control/dart_guidance.hpp"

using namespace aerosp;
using namespace aerosp::dart;

// Simulation parameters
struct MCSimParams {
    double v0_mean;
    double v0_sigma;
    double pitch_mean_deg;
    double pitch_sigma_deg;
    double yaw_mean_deg;
    double yaw_sigma_deg;
    double nav_ratio;
    double control_gain;
    double forced_dp;
    double forced_dp_2;
    double forced_switch_x;
    int num_sims;
    unsigned long long seed;
};

struct MCSimResult {
    double x, y, z;
    double flight_time;
    bool guidance_active;
};

struct TrajectoryPoint {
    double t;
    double x, y, z;
    double vx, vy, vz;
    double qw, qx, qy, qz;
    double alpha, mach;
    double fx, fy, fz;
};

struct MCSummary {
    double hit_rate;
    double mean_miss_distance;
    double max_miss_distance;
};

struct MCBatchOutput {
    std::vector<MCSimResult> results;
    std::vector<TrajectoryPoint> debug_traj;
    double elapsed_sec = 0.0;
};

std::vector<double> parse_double_list(const std::string& text) {
    std::vector<double> values;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            values.push_back(std::stod(item));
        }
    }
    return values;
}

MCSummary summarize_results(const std::vector<MCSimResult>& results, const Eigen::Vector3d& target_ned) {
    MCSummary summary{0.0, 0.0, 0.0};
    if (results.empty()) {
        return summary;
    }

    const double hit_radius = 0.055 * 0.5;
    size_t hit_count = 0;
    double miss_sum = 0.0;

    for (const auto& result : results) {
        double err_x = result.x - target_ned.x();
        double err_y = result.y - target_ned.y();
        double err_z = result.z - target_ned.z();
        double miss_dist = std::sqrt(err_x * err_x + err_y * err_y + err_z * err_z);
        miss_sum += miss_dist;
        summary.max_miss_distance = std::max(summary.max_miss_distance, miss_dist);
        if (miss_dist <= hit_radius) {
            ++hit_count;
        }
    }

    summary.hit_rate = 100.0 * static_cast<double>(hit_count) / static_cast<double>(results.size());
    summary.mean_miss_distance = miss_sum / static_cast<double>(results.size());
    return summary;
}

CUDA_HOST_DEVICE sim::control::DartGuidance::Config build_guidance_config(const DartConfig& cfg) {
    sim::control::DartGuidance::Config guid_cfg;
    guid_cfg.nav_ratio = cfg.nav_ratio;
    guid_cfg.guidance_start_time = cfg.guid_start_time;
    guid_cfg.visual_fov_deg = cfg.guid_fov_deg;
    guid_cfg.sensor_latency = cfg.guid_sensor_latency;
    guid_cfg.ctrl_gain = cfg.ctrl_gain;
    guid_cfg.ctrl_q_bias = cfg.ctrl_q_bias;
    guid_cfg.max_ctrl_coeff = cfg.max_ctrl_coeff;
    guid_cfg.terminal_range = cfg.guid_terminal_range;
    guid_cfg.update_hz = cfg.guid_update_hz;
    guid_cfg.min_tgo = cfg.guid_min_tgo;
    guid_cfg.max_accel = cfg.guid_max_accel;
    guid_cfg.y_pos_gain = cfg.guid_y_pos_gain;
    guid_cfg.y_vel_gain = cfg.guid_y_vel_gain;
    guid_cfg.z_pos_gain = cfg.guid_z_pos_gain;
    guid_cfg.z_vel_gain = cfg.guid_z_vel_gain;
    guid_cfg.rate_damp = cfg.guid_rate_damp;
    guid_cfg.dp_gain = cfg.guid_dp_gain;
    guid_cfg.dy_gain = cfg.guid_dy_gain;
    guid_cfg.vz_gain = cfg.guid_vz_gain;
    guid_cfg.dp_step = cfg.guid_dp_step;
    guid_cfg.dy_step = cfg.guid_dy_step;
    guid_cfg.y_deadband = cfg.guid_y_deadband;
    guid_cfg.z_deadband = cfg.guid_z_deadband;
    guid_cfg.gravity_mag = cfg.gravity_mag;
    return guid_cfg;
}

/**
 * @brief CUDA Kernel for Monte Carlo Simulation
 */
__global__ void dart_mc_kernel(
    MCSimParams params,
    DartAeroTableGPU aero_table,
    DartConfig dart_cfg,
    LLA origin_lla,
    Eigen::Vector3d origin_ecef,
    Eigen::Vector3d target_ned,
    double g_mag,       // NEW: Constant g
    double rho_base,    // NEW: Constant rho
    MCSimResult* results,
    TrajectoryPoint* debug_traj, // New parameter
    unsigned long long seed,
    bool use_guidance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= params.num_sims) return;

    // 1. Initialize Random Number Generator
    curandState state_rand;
    curand_init(seed + params.seed, idx, 0, &state_rand);

    // 2. Sample Initial Conditions (Deterministic for 1 sim for verification)
    double v0 = params.v0_mean;
    double pitch_deg = params.pitch_mean_deg;
    double yaw_deg = params.yaw_mean_deg;

    if (params.num_sims > 1) {
        v0 += curand_normal(&state_rand) * params.v0_sigma;
        pitch_deg += curand_normal(&state_rand) * params.pitch_sigma_deg;
        yaw_deg += curand_normal(&state_rand) * params.yaw_sigma_deg;
    }

    // 3. Setup Initial State
    double theta = pitch_deg * 3.141592653589793 / 180.0;
    double psi = yaw_deg * 3.141592653589793 / 180.0;

    // Rotation matrix from ECEF to NED
    double sin_lat = sin(origin_lla.lat);
    double cos_lat = cos(origin_lla.lat);
    double sin_lon = sin(origin_lla.lon);
    double cos_lon = cos(origin_lla.lon);

    Eigen::Matrix3d R_en;
    R_en(0, 0) = -sin_lat * cos_lon; R_en(0, 1) = -sin_lat * sin_lon; R_en(0, 2) = cos_lat;
    R_en(1, 0) = -sin_lon;           R_en(1, 1) =  cos_lon;           R_en(1, 2) = 0.0;
    R_en(2, 0) = -cos_lat * cos_lon; R_en(2, 1) = -cos_lat * sin_lon; R_en(2, 2) = -sin_lat;

    // NED to ECEF is the transpose
    Eigen::Matrix3d R_ne = R_en.transpose();

    // Initial velocity in NED: [North, East, Down]
    // Pitch UP is negative Z
    Eigen::Vector3d vel_ned(v0 * cos(theta) * cos(psi),
                            v0 * cos(theta) * sin(psi),
                            -v0 * sin(theta));

    State6DOF state;
    state.pos_ecef = Eigen::Vector3d::Zero(); // RELATIVE ECEF
    state.mass = dart_cfg.mass;
    state.vel_ecef = R_ne * vel_ned;

    // Initial orientation: body points along velocity in NED
    // Rotation sequence: Yaw (psi) then Pitch (theta)
    // In NED, positive rotation around Y is Pitch UP.
    Eigen::Quaterniond q_yaw_quat(Eigen::AngleAxisd(psi, Eigen::Vector3d::UnitZ()));
    Eigen::Quaterniond q_pitch_quat(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitY()));
    Eigen::Quaterniond q_nb = q_yaw_quat * q_pitch_quat;

    // Body to ECEF: q_be = q_ne * q_nb
    Eigen::Quaterniond q_ne_quat(R_ne);
    state.quat_be = q_ne_quat * q_nb;
    state.quat_be.normalize();
    state.omega_body.setZero();

    // 4. Integration Loop
    double t = 0.0;
    double dt = 0.001;

    // Physical Properties from Config
    InertialProps inertia;
    inertia.mass = dart_cfg.mass;
    inertia.inertia = dart_cfg.inertia;
    inertia.com = dart_cfg.com;
    sim::control::DartGuidance::Config guid_cfg = build_guidance_config(dart_cfg);

    bool guid_active_ever = false;
    double dp = 0, dy = 0;
    sim::control::DartGuidance::GuidanceState guid_state;
    sim::control::DartGuidance::reset_state(guid_state);
    Eigen::Vector3d g_ecef = R_ne * Eigen::Vector3d(0, 0, g_mag);
    Eigen::Vector3d prev_p_ned = Eigen::Vector3d::Zero();
    Eigen::Vector3d eval_point_ned = prev_p_ned;
    double best_abs_x_err = fabs(target_ned.x() - prev_p_ned.x());
    bool captured_target_plane = false;

    int step_cnt = 0;
    while (t < 5.0) {
        // 1. Stable Local Coordinates (state.pos_ecef is relative to origin)
        Eigen::Vector3d p_ned = R_en * state.pos_ecef;
        double current_alt = dart_cfg.launch_alt - p_ned.z();

        if (current_alt < -0.5 && t > 0.05) break;
        if (p_ned.x() > target_ned.x() + 2.0) break;
        if (t > 4.5) break;

        if (state.pos_ecef.x() != state.pos_ecef.x()) {
            results[idx].x = -777.0; // NaN in integration
            return;
        }

        // Log for debug (first sim only, limit steps)
        if (idx == 0 && debug_traj != nullptr && step_cnt < 2000) {
            Eigen::Vector3d v_ned_cur = R_en * state.vel_ecef;
            Eigen::Vector3d v_b = state.quat_be.inverse() * state.vel_ecef;
            double a_deg = atan2(v_b.z(), v_b.x()) * 180.0 / 3.14159265;
            double mach = v_b.norm() / 340.0;

            debug_traj[step_cnt] = {
                t, p_ned.x(), p_ned.y(), p_ned.z(),
                v_ned_cur.x(), v_ned_cur.y(), v_ned_cur.z(),
                state.quat_be.w(), state.quat_be.x(), state.quat_be.y(), state.quat_be.z(),
                a_deg, mach, 0, 0, 0 // Forces logged in system lambda if needed
            };
            step_cnt++;
        }

        // 2. Guidance
        bool active = false;
        if (use_guidance) {
            if (fabs(params.forced_dp) > 1e-12 || fabs(params.forced_dp_2) > 1e-12) {
                double x_rem = target_ned.x() - p_ned.x();
                if (params.forced_switch_x > 0.0) {
                    dp = (x_rem > params.forced_switch_x) ? params.forced_dp : params.forced_dp_2;
                } else {
                    dp = params.forced_dp;
                }
                dy = 0.0;
                active = true;
            } else {
                Eigen::Vector3d v_ned_cur = R_en * state.vel_ecef;
                auto guid_out = sim::control::DartGuidance::update_closed_loop(
                    t, p_ned, v_ned_cur, state.omega_body, target_ned, dt, guid_cfg, guid_state
                );
                dp = guid_out.delta_pitch;
                dy = guid_out.delta_yaw;
                active = guid_out.active;
            }
            if (active) guid_active_ever = true;
        }

        auto system = [&](const State6DOF& s, double cur_t) {
            // Constant local density for performance
            Eigen::Vector3d v_b = s.quat_be.inverse() * s.vel_ecef;
            double v_m = v_b.norm();
            if (v_m < 0.1) return Dynamics6DOF::compute_derivatives(s, ForcesMoments{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 0}, inertia, g_ecef, SimulationProfile::LOCAL_TACTICAL);

            double a_deg = atan2(v_b.z(), v_b.x()) * 180.0 / 3.14159265;
            double m = v_m / 340.0;

            AeroCoeffsGPU c = DartAeroTable::interpolate_gpu(aero_table, m, a_deg);

            double q = 0.5 * rho_base * v_m * v_m;
            double S = dart_cfg.ref_area;
            double L = dart_cfg.ref_length;

            Eigen::Vector3d f_b(c.CX * q * S, c.CY * q * S, c.CZ * q * S);

            Eigen::Vector3d m_b(
                (-0.5 * s.omega_body.x()) * q * S * L,
                (c.Cm + c.Cmq * s.omega_body.y() + dp * dart_cfg.ctrl_moment_coeff_pitch) * q * S * L,
                (c.Cn + c.Cnr * s.omega_body.z() + dy * dart_cfg.ctrl_moment_coeff_yaw) * q * S * L
            );

            return Dynamics6DOF::compute_derivatives(s, ForcesMoments{f_b, m_b, 0}, inertia, g_ecef, SimulationProfile::LOCAL_TACTICAL);
        };

        state = Dynamics6DOF::integrate_rk4(state, system, t, dt);
        t += dt;

        Eigen::Vector3d curr_p_ned = R_en * state.pos_ecef;
        double curr_abs_x_err = fabs(target_ned.x() - curr_p_ned.x());
        if (curr_abs_x_err < best_abs_x_err) {
            best_abs_x_err = curr_abs_x_err;
            eval_point_ned = curr_p_ned;
        }

        double dx_segment = curr_p_ned.x() - prev_p_ned.x();
        if (!captured_target_plane && fabs(dx_segment) > 1e-9) {
            double ratio = (target_ned.x() - prev_p_ned.x()) / dx_segment;
            if (ratio >= 0.0 && ratio <= 1.0) {
                eval_point_ned = prev_p_ned + ratio * (curr_p_ned - prev_p_ned);
                captured_target_plane = true;
                break;
            }
        }
        prev_p_ned = curr_p_ned;

        if (state.pos_ecef.x() != state.pos_ecef.x()) {
            results[idx].x = -777.0; // NaN in integration
            return;
        }
    }

    results[idx] = {eval_point_ned.x(), eval_point_ned.y(), eval_point_ned.z(), t, guid_active_ever};
}

MCBatchOutput run_mc_batch(
    const MCSimParams& params,
    const DartAeroTableGPU& aero_table,
    const DartConfig& dart_cfg,
    const LLA& origin_lla,
    const Eigen::Vector3d& origin_ecef,
    const Eigen::Vector3d& target_ned,
    MCSimResult* d_results,
    TrajectoryPoint* d_debug_traj,
    bool write_debug_traj,
    bool use_guidance
) {
    MCBatchOutput output;

    int threadsPerBlock = 256;
    int blocksPerGrid = (params.num_sims + threadsPerBlock - 1) / threadsPerBlock;
    if (blocksPerGrid == 0) {
        blocksPerGrid = 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    dart_mc_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        params, aero_table, dart_cfg, origin_lla, origin_ecef, target_ned,
        dart_cfg.gravity_mag, dart_cfg.atm_density,
        d_results, write_debug_traj ? d_debug_traj : nullptr, 0ULL, use_guidance
    );
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    output.elapsed_sec = std::chrono::duration<double>(end - start).count();
    output.results.resize(params.num_sims);
    cudaMemcpy(output.results.data(), d_results, params.num_sims * sizeof(MCSimResult), cudaMemcpyDeviceToHost);

    if (write_debug_traj && d_debug_traj != nullptr) {
        output.debug_traj.resize(2000);
        cudaMemcpy(output.debug_traj.data(), d_debug_traj, 2000 * sizeof(TrajectoryPoint), cudaMemcpyDeviceToHost);
    }

    return output;
}

void write_debug_traj_csv(const std::vector<TrajectoryPoint>& debug_traj) {
    if (debug_traj.empty()) {
        return;
    }

    std::ofstream debug_file("output/logs/dart_mc_debug_traj.csv");
    debug_file << "Time,X,Y,Z,Vx,Vy,Vz,Qw,Qx,Qy,Qz,Alpha,Mach\n";
    for (const auto& p : debug_traj) {
        if (p.t == 0 && p.x == 0 && p.y == 0 && p.z == 0 && debug_file.tellp() > 50) {
            break;
        }
        debug_file << p.t << "," << p.x << "," << p.y << "," << p.z << ","
                   << p.vx << "," << p.vy << "," << p.vz << ","
                   << p.qw << "," << p.qx << "," << p.qy << "," << p.qz << ","
                   << p.alpha << "," << p.mach << "\n";
    }
}

void write_results_csv(const std::vector<MCSimResult>& results) {
    std::ofstream outfile("output/logs/dart_mc_results.csv");
    outfile << "ID,X,Y,Z,Time,Guidance\n";
    for (int i = 0; i < static_cast<int>(results.size()); ++i) {
        outfile << i << "," << results[i].x << "," << results[i].y << "," << results[i].z << ","
                << results[i].flight_time << "," << (results[i].guidance_active ? 1 : 0) << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: DartMC <num_sims> [v0] [pitch] [yaw] [nav_ratio] [ctrl_gain] [v0_sigma] [pitch_sigma] [yaw_sigma] [seed] [forced_dp] [forced_dp_2] [forced_switch_x]" << std::endl;
        std::cout << "   or: DartMC --sweep <num_sims> <v0> <pitch> <yaw> <nav_list> <ctrl_list> [seed]" << std::endl;
        std::cout << "   or: DartMC --closed-loop-sweep <num_sims> <v0> <pitch> <yaw> <v0_sigma> <pitch_sigma> <yaw_sigma> <y_pos_list> <z_pos_list> <dp_gain_list> <dy_gain_list> [seed]" << std::endl;
        std::cout << "   or: DartMC --closed-loop-structure-sweep <num_sims> <v0> <pitch> <yaw> <v0_sigma> <pitch_sigma> <yaw_sigma> <terminal_range_list> <max_accel_list> <rate_damp_list> <pitch_moment_list> <yaw_moment_list> [seed]" << std::endl;
        std::cout << "   or: DartMC --forced-dp-sweep <v0> <pitch_list> <yaw> <dp_list> [seed]" << std::endl;
        std::cout << "   or: DartMC --forced-dp-2stage-sweep <v0> <pitch_list> <yaw> <dp1_list> <dp2_list> <switch_list> [seed]" << std::endl;
        std::cout << "   or: DartMC --forced-dp-2stage-mc-sweep <num_sims> <v0> <pitch> <yaw> <v0_sigma> <pitch_sigma> <yaw_sigma> <dp1_list> <dp2_list> <switch_list> [seed]" << std::endl;
        return 0;
    }

    bool sweep_mode = std::string(argv[1]) == "--sweep";
    bool closed_loop_sweep_mode = std::string(argv[1]) == "--closed-loop-sweep";
    bool closed_loop_structure_sweep_mode = std::string(argv[1]) == "--closed-loop-structure-sweep";
    bool authority_mode = std::string(argv[1]) == "--forced-dp-sweep";
    bool authority_2stage_mode = std::string(argv[1]) == "--forced-dp-2stage-sweep";
    bool authority_2stage_mc_mode = std::string(argv[1]) == "--forced-dp-2stage-mc-sweep";
    if (sweep_mode && argc < 8) {
        std::cerr << "Sweep mode requires: --sweep <num_sims> <v0> <pitch> <yaw> <nav_list> <ctrl_list> [seed]" << std::endl;
        return -1;
    }
    if (closed_loop_sweep_mode && argc < 13) {
        std::cerr << "Closed-loop sweep requires: --closed-loop-sweep <num_sims> <v0> <pitch> <yaw> <v0_sigma> <pitch_sigma> <yaw_sigma> <y_pos_list> <z_pos_list> <dp_gain_list> <dy_gain_list> [seed]" << std::endl;
        return -1;
    }
    if (closed_loop_structure_sweep_mode && argc < 14) {
        std::cerr << "Closed-loop structure sweep requires: --closed-loop-structure-sweep <num_sims> <v0> <pitch> <yaw> <v0_sigma> <pitch_sigma> <yaw_sigma> <terminal_range_list> <max_accel_list> <rate_damp_list> <pitch_moment_list> <yaw_moment_list> [seed]" << std::endl;
        return -1;
    }
    if (authority_mode && argc < 6) {
        std::cerr << "Forced-dp sweep requires: --forced-dp-sweep <v0> <pitch_list> <yaw> <dp_list> [seed]" << std::endl;
        return -1;
    }
    if (authority_2stage_mode && argc < 8) {
        std::cerr << "Forced-dp 2-stage sweep requires: --forced-dp-2stage-sweep <v0> <pitch_list> <yaw> <dp1_list> <dp2_list> <switch_list> [seed]" << std::endl;
        return -1;
    }
    if (authority_2stage_mc_mode && argc < 12) {
        std::cerr << "Forced-dp 2-stage MC sweep requires: --forced-dp-2stage-mc-sweep <num_sims> <v0> <pitch> <yaw> <v0_sigma> <pitch_sigma> <yaw_sigma> <dp1_list> <dp2_list> <switch_list> [seed]" << std::endl;
        return -1;
    }
    MCSimParams params;
    params.num_sims = (authority_mode || authority_2stage_mode || authority_2stage_mc_mode) ? 1 : std::stoi(argv[(sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 2 : 1]);
    if (authority_2stage_mc_mode) {
        params.num_sims = std::stoi(argv[2]);
    }
    params.v0_mean = (authority_mode || authority_2stage_mode) ? std::stod(argv[2]) : ((argc > ((sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 3 : 2)) ? std::stod(argv[(sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 3 : 2]) : 25.0);
    if (authority_2stage_mc_mode) {
        params.v0_mean = std::stod(argv[3]);
    }
    params.v0_sigma = 0.3;
    params.pitch_mean_deg = (authority_mode || authority_2stage_mode) ? 0.0 : ((argc > ((sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 4 : 3)) ? std::stod(argv[(sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 4 : 3]) : 3.93);
    if (authority_2stage_mc_mode) {
        params.pitch_mean_deg = std::stod(argv[4]);
    }
    params.pitch_sigma_deg = 0.2;
    params.yaw_mean_deg = (authority_mode || authority_2stage_mode) ? std::stod(argv[4]) : ((argc > ((sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 5 : 4)) ? std::stod(argv[(sweep_mode || closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 5 : 4]) : 7.3);
    if (authority_2stage_mc_mode) {
        params.yaw_mean_deg = std::stod(argv[5]);
    }
    params.yaw_sigma_deg = 0.2;
    params.nav_ratio = (authority_mode || authority_2stage_mode || authority_2stage_mc_mode) ? 0.0 : ((closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 2.0 : ((argc > (sweep_mode ? 6 : 5)) ? std::stod(argv[sweep_mode ? 6 : 5]) : 2.0));
    params.control_gain = (authority_mode || authority_2stage_mode || authority_2stage_mc_mode) ? 0.0 : ((closed_loop_sweep_mode || closed_loop_structure_sweep_mode) ? 2.5 : ((argc > (sweep_mode ? 7 : 6)) ? std::stod(argv[sweep_mode ? 7 : 6]) : 1.0));
    params.forced_dp = (!sweep_mode && !closed_loop_sweep_mode && !closed_loop_structure_sweep_mode && !authority_mode && !authority_2stage_mode && !authority_2stage_mc_mode && argc > 11) ? std::stod(argv[11]) : 0.0;
    params.forced_dp_2 = (!sweep_mode && !closed_loop_sweep_mode && !closed_loop_structure_sweep_mode && !authority_mode && !authority_2stage_mode && !authority_2stage_mc_mode && argc > 12) ? std::stod(argv[12]) : 0.0;
    params.forced_switch_x = (!sweep_mode && !closed_loop_sweep_mode && !closed_loop_structure_sweep_mode && !authority_mode && !authority_2stage_mode && !authority_2stage_mc_mode && argc > 13) ? std::stod(argv[13]) : 0.0;
    params.seed = authority_2stage_mc_mode ? ((argc > 12) ? std::stoull(argv[12]) : 123456789ULL)
                                 : authority_2stage_mode ? ((argc > 8) ? std::stoull(argv[8]) : 123456789ULL)
                                 : authority_mode ? ((argc > 6) ? std::stoull(argv[6]) : 123456789ULL)
                                 : (closed_loop_structure_sweep_mode ? ((argc > 14) ? std::stoull(argv[14]) : 123456789ULL)
                                 : (closed_loop_sweep_mode ? ((argc > 13) ? std::stoull(argv[13]) : 123456789ULL)
                                 : (sweep_mode ? ((argc > 8) ? std::stoull(argv[8]) : 123456789ULL)
                                               : ((argc > 10) ? std::stoull(argv[10]) : 123456789ULL))));
    if (authority_2stage_mc_mode) {
        params.v0_sigma = std::stod(argv[6]);
        params.pitch_sigma_deg = std::stod(argv[7]);
        params.yaw_sigma_deg = std::stod(argv[8]);
    } else if (closed_loop_sweep_mode || closed_loop_structure_sweep_mode) {
        params.v0_sigma = std::stod(argv[6]);
        params.pitch_sigma_deg = std::stod(argv[7]);
        params.yaw_sigma_deg = std::stod(argv[8]);
    } else if (!sweep_mode && !authority_mode && !authority_2stage_mode) {
        params.v0_sigma = (argc > 7) ? std::stod(argv[7]) : ((params.num_sims == 1) ? 0.0 : params.v0_sigma);
        params.pitch_sigma_deg = (argc > 8) ? std::stod(argv[8]) : ((params.num_sims == 1) ? 0.0 : params.pitch_sigma_deg);
        params.yaw_sigma_deg = (argc > 9) ? std::stod(argv[9]) : ((params.num_sims == 1) ? 0.0 : params.yaw_sigma_deg);
    } else if (authority_mode || authority_2stage_mode) {
        params.v0_sigma = 0.0;
        params.pitch_sigma_deg = 0.0;
        params.yaw_sigma_deg = 0.0;
    }

    if (params.v0_mean < 0.1) params.v0_mean = 25.0;
    if (params.nav_ratio < 0) params.nav_ratio = 2.0;
    if (params.control_gain < 0) params.control_gain = 1.0;
    if (params.v0_sigma < 0) params.v0_sigma = 0.0;
    if (params.pitch_sigma_deg < 0) params.pitch_sigma_deg = 0.0;
    if (params.yaw_sigma_deg < 0) params.yaw_sigma_deg = 0.0;

    DartConfig dart_cfg;
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
    LLA origin_lla = {dart_cfg.launch_lat * 3.14159265 / 180.0,
                      dart_cfg.launch_lon * 3.14159265 / 180.0,
                      dart_cfg.launch_alt};
    Eigen::Vector3d origin_ecef = CoordinateTransform::lla_to_ecef(origin_lla);

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

    AtmosphereData atm_baseline = AtmosphereModel::calculate_ussa76(dart_cfg.launch_alt);
    dart_cfg.atm_density = atm_baseline.density;
    dart_cfg.sound_speed = atm_baseline.sound_speed;

    dart_cfg.nav_ratio = params.nav_ratio;
    dart_cfg.ctrl_gain = params.control_gain;

    std::cout << "--- aerosp: Dart GPU Monte Carlo ---" << std::endl;
    std::cout << "Location: (" << dart_cfg.launch_lat << ", " << dart_cfg.launch_lon << ")" << std::endl;
    std::cout << "Env: Gravity=" << dart_cfg.gravity_mag << " m/s2, Density=" << dart_cfg.atm_density << " kg/m3" << std::endl;
    std::cout << "Params: v0=" << params.v0_mean << ", pitch=" << params.pitch_mean_deg
              << ", yaw=" << params.yaw_mean_deg << ", sims=" << params.num_sims << std::endl;
    std::cout << "Guidance: PN Ratio=" << params.nav_ratio << ", Gain=" << params.control_gain << ", Seed=" << params.seed
              << ", ForcedDP1=" << params.forced_dp << ", ForcedDP2=" << params.forced_dp_2 << ", SwitchX=" << params.forced_switch_x << std::endl;

    DartAeroTable table(aero_path);
    if (!table.is_loaded()) {
        std::cerr << "Failed to load dart aero table." << std::endl;
        if (!table.is_loaded()) return -1;
    }
    table.prepare_gpu();

    Eigen::Vector3d target_ned = dart_cfg.get_base(dart_cfg.target_alt).pos_local;

    MCSimResult* d_results;
    cudaMalloc(&d_results, params.num_sims * sizeof(MCSimResult));

    TrajectoryPoint* d_debug_traj = nullptr;
    if (!sweep_mode && !closed_loop_sweep_mode && !closed_loop_structure_sweep_mode && params.num_sims >= 1) {
        cudaMalloc(&d_debug_traj, 2000 * sizeof(TrajectoryPoint));
    }

    if (sweep_mode) {
        std::vector<double> nav_values = parse_double_list(argv[6]);
        std::vector<double> ctrl_values = parse_double_list(argv[7]);
        std::ofstream sweep_file("output/logs/dart_mc_sweep.csv");
        sweep_file << "NavRatio,CtrlGain,HitRate,MeanMissMM,MaxMissMM,ElapsedSec\n";

        for (double nav_ratio : nav_values) {
            for (double ctrl_gain : ctrl_values) {
                params.nav_ratio = nav_ratio;
                params.control_gain = ctrl_gain;
                dart_cfg.nav_ratio = nav_ratio;
                dart_cfg.ctrl_gain = ctrl_gain;

                auto batch = run_mc_batch(
                    params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
                    d_results, nullptr, false, true
                );
                auto summary = summarize_results(batch.results, target_ned);
                sweep_file << nav_ratio << "," << ctrl_gain << "," << summary.hit_rate << ","
                           << summary.mean_miss_distance * 1000.0 << "," << summary.max_miss_distance * 1000.0 << ","
                           << batch.elapsed_sec << "\n";
                std::cout << "NR=" << nav_ratio << " CG=" << ctrl_gain
                          << " HitRate=" << summary.hit_rate << "% MeanMiss=" << summary.mean_miss_distance * 1000.0
                          << "mm Time=" << batch.elapsed_sec << "s" << std::endl;
            }
        }
    } else if (closed_loop_sweep_mode) {
        std::vector<double> y_pos_values = parse_double_list(argv[9]);
        std::vector<double> z_pos_values = parse_double_list(argv[10]);
        std::vector<double> dp_gain_values = parse_double_list(argv[11]);
        std::vector<double> dy_gain_values = parse_double_list(argv[12]);
        std::ofstream sweep_file("output/logs/dart_mc_closed_loop_sweep.csv");
        sweep_file << "YPosGain,ZPosGain,DPGain,DYGain,HitRate,MeanMissMM,MaxMissMM,ElapsedSec\n";

        for (double y_pos_gain : y_pos_values) {
            for (double z_pos_gain : z_pos_values) {
                for (double dp_gain : dp_gain_values) {
                    for (double dy_gain : dy_gain_values) {
                        dart_cfg.guid_y_pos_gain = y_pos_gain;
                        dart_cfg.guid_z_pos_gain = z_pos_gain;
                        dart_cfg.guid_dp_gain = dp_gain;
                        dart_cfg.guid_dy_gain = dy_gain;

                        auto batch = run_mc_batch(
                            params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
                            d_results, nullptr, false, true
                        );
                        auto summary = summarize_results(batch.results, target_ned);
                        sweep_file << y_pos_gain << "," << z_pos_gain << "," << dp_gain << "," << dy_gain << ","
                                   << summary.hit_rate << "," << summary.mean_miss_distance * 1000.0 << ","
                                   << summary.max_miss_distance * 1000.0 << "," << batch.elapsed_sec << "\n";
                        std::cout << "YP=" << y_pos_gain << " ZP=" << z_pos_gain
                                  << " DPG=" << dp_gain << " DYG=" << dy_gain
                                  << " HitRate=" << summary.hit_rate << "% MeanMiss="
                                  << summary.mean_miss_distance * 1000.0 << "mm Time="
                                  << batch.elapsed_sec << "s" << std::endl;
                    }
                }
            }
        }
    } else if (closed_loop_structure_sweep_mode) {
        std::vector<double> terminal_range_values = parse_double_list(argv[9]);
        std::vector<double> max_accel_values = parse_double_list(argv[10]);
        std::vector<double> rate_damp_values = parse_double_list(argv[11]);
        std::vector<double> pitch_moment_values = parse_double_list(argv[12]);
        std::vector<double> yaw_moment_values = parse_double_list(argv[13]);
        std::ofstream sweep_file("output/logs/dart_mc_closed_loop_structure_sweep.csv");
        sweep_file << "TerminalRange,MaxAccel,RateDamp,PitchMomentCoeff,YawMomentCoeff,HitRate,MeanMissMM,MaxMissMM,ElapsedSec\n";

        for (double terminal_range : terminal_range_values) {
            for (double max_accel : max_accel_values) {
                for (double rate_damp : rate_damp_values) {
                    for (double pitch_moment : pitch_moment_values) {
                        for (double yaw_moment : yaw_moment_values) {
                            dart_cfg.guid_terminal_range = terminal_range;
                            dart_cfg.guid_max_accel = max_accel;
                            dart_cfg.guid_rate_damp = rate_damp;
                            dart_cfg.ctrl_moment_coeff_pitch = pitch_moment;
                            dart_cfg.ctrl_moment_coeff_yaw = yaw_moment;

                            auto batch = run_mc_batch(
                                params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
                                d_results, nullptr, false, true
                            );
                            auto summary = summarize_results(batch.results, target_ned);
                            sweep_file << terminal_range << "," << max_accel << "," << rate_damp << ","
                                       << pitch_moment << "," << yaw_moment << "," << summary.hit_rate << ","
                                       << summary.mean_miss_distance * 1000.0 << ","
                                       << summary.max_miss_distance * 1000.0 << "," << batch.elapsed_sec << "\n";
                            std::cout << "TR=" << terminal_range << " MA=" << max_accel
                                      << " RD=" << rate_damp << " PM=" << pitch_moment
                                      << " YM=" << yaw_moment << " HitRate=" << summary.hit_rate
                                      << "% MeanMiss=" << summary.mean_miss_distance * 1000.0
                                      << "mm Time=" << batch.elapsed_sec << "s" << std::endl;
                        }
                    }
                }
            }
        }
    } else if (authority_mode) {
        std::vector<double> pitch_values = parse_double_list(argv[3]);
        std::vector<double> dp_values = parse_double_list(argv[5]);
        std::ofstream sweep_file("output/logs/dart_mc_forced_dp_sweep.csv");
        sweep_file << "PitchDeg,ForcedDP,HitRate,MeanMissMM,MaxMissMM,ElapsedSec\n";

        for (double pitch_deg : pitch_values) {
            for (double forced_dp : dp_values) {
                params.pitch_mean_deg = pitch_deg;
                params.forced_dp = forced_dp;
                auto batch = run_mc_batch(
                    params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
                    d_results, nullptr, false, true
                );
                auto summary = summarize_results(batch.results, target_ned);
                sweep_file << pitch_deg << "," << forced_dp << "," << summary.hit_rate << ","
                           << summary.mean_miss_distance * 1000.0 << "," << summary.max_miss_distance * 1000.0 << ","
                           << batch.elapsed_sec << "\n";
                std::cout << "Pitch=" << pitch_deg << " ForcedDP=" << forced_dp
                          << " HitRate=" << summary.hit_rate << "% MeanMiss=" << summary.mean_miss_distance * 1000.0
                          << "mm Time=" << batch.elapsed_sec << "s" << std::endl;
            }
        }
    } else if (authority_2stage_mode) {
        std::vector<double> pitch_values = parse_double_list(argv[3]);
        std::vector<double> dp1_values = parse_double_list(argv[5]);
        std::vector<double> dp2_values = parse_double_list(argv[6]);
        std::vector<double> switch_values = parse_double_list(argv[7]);
        std::ofstream sweep_file("output/logs/dart_mc_forced_dp_2stage_sweep.csv");
        sweep_file << "PitchDeg,ForcedDP1,ForcedDP2,SwitchX,HitRate,MeanMissMM,MaxMissMM,ElapsedSec\n";

        for (double pitch_deg : pitch_values) {
            for (double forced_dp1 : dp1_values) {
                for (double forced_dp2 : dp2_values) {
                    for (double switch_x : switch_values) {
                        params.pitch_mean_deg = pitch_deg;
                        params.forced_dp = forced_dp1;
                        params.forced_dp_2 = forced_dp2;
                        params.forced_switch_x = switch_x;
                        auto batch = run_mc_batch(
                            params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
                            d_results, nullptr, false, true
                        );
                        auto summary = summarize_results(batch.results, target_ned);
                        sweep_file << pitch_deg << "," << forced_dp1 << "," << forced_dp2 << "," << switch_x << ","
                                   << summary.hit_rate << "," << summary.mean_miss_distance * 1000.0 << ","
                                   << summary.max_miss_distance * 1000.0 << "," << batch.elapsed_sec << "\n";
                        std::cout << "Pitch=" << pitch_deg << " DP1=" << forced_dp1 << " DP2=" << forced_dp2
                                  << " SwitchX=" << switch_x << " MeanMiss=" << summary.mean_miss_distance * 1000.0
                                  << "mm Time=" << batch.elapsed_sec << "s" << std::endl;
                    }
                }
            }
        }
    } else if (authority_2stage_mc_mode) {
        std::vector<double> dp1_values = parse_double_list(argv[9]);
        std::vector<double> dp2_values = parse_double_list(argv[10]);
        std::vector<double> switch_values = parse_double_list(argv[11]);
        std::ofstream sweep_file("output/logs/dart_mc_forced_dp_2stage_mc_sweep.csv");
        sweep_file << "ForcedDP1,ForcedDP2,SwitchX,HitRate,MeanMissMM,MaxMissMM,ElapsedSec\n";

        for (double forced_dp1 : dp1_values) {
            for (double forced_dp2 : dp2_values) {
                for (double switch_x : switch_values) {
                    params.forced_dp = forced_dp1;
                    params.forced_dp_2 = forced_dp2;
                    params.forced_switch_x = switch_x;
                    auto batch = run_mc_batch(
                        params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
                        d_results, nullptr, false, true
                    );
                    auto summary = summarize_results(batch.results, target_ned);
                    sweep_file << forced_dp1 << "," << forced_dp2 << "," << switch_x << ","
                               << summary.hit_rate << "," << summary.mean_miss_distance * 1000.0 << ","
                               << summary.max_miss_distance * 1000.0 << "," << batch.elapsed_sec << "\n";
                    std::cout << "MC DP1=" << forced_dp1 << " DP2=" << forced_dp2
                              << " SwitchX=" << switch_x << " HitRate=" << summary.hit_rate
                              << "% MeanMiss=" << summary.mean_miss_distance * 1000.0
                              << "mm Time=" << batch.elapsed_sec << "s" << std::endl;
                }
            }
        }
    } else {
        bool use_guidance = (params.nav_ratio > 0.0 && params.control_gain > 0.0) || std::abs(params.forced_dp) > 1e-12 || std::abs(params.forced_dp_2) > 1e-12;
        std::cout << "Launching GPU Monte Carlo with " << (use_guidance ? "Guidance" : "Ballistic Mode") << "..." << std::endl;
        auto batch = run_mc_batch(
            params, table.get_gpu_data(), dart_cfg, origin_lla, origin_ecef, target_ned,
            d_results, d_debug_traj, true, use_guidance
        );
        auto summary = summarize_results(batch.results, target_ned);
        std::cout << "GPU Simulation finished in " << batch.elapsed_sec << "s" << std::endl;
        std::cout << "Hit Rate: " << summary.hit_rate << "%, Mean Miss: " << summary.mean_miss_distance * 1000.0 << " mm" << std::endl;
        write_results_csv(batch.results);
        write_debug_traj_csv(batch.debug_traj);
    }

    if (d_debug_traj) {
        cudaFree(d_debug_traj);
    }
    cudaFree(d_results);
    return 0;
}
