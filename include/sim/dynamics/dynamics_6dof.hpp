#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include "infra/common.hpp"

namespace aerosp {

/**
 * @brief 6-DOF state vector
 */
struct State6DOF {
    Eigen::Vector3d pos_ecef;      // Position in ECEF (m)
    Eigen::Vector3d vel_ecef;      // Velocity in ECEF (m/s)
    Eigen::Quaterniond quat_be;    // Quaternion from ECEF to Body
    Eigen::Vector3d omega_body;    // Angular velocity in Body frame (rad/s)
    double mass;                   // Mass (kg)

    // Operator overloading for RK4 integration
    CUDA_HOST_DEVICE State6DOF operator+(const State6DOF& other) const {
        State6DOF result;
        result.pos_ecef = pos_ecef + other.pos_ecef;
        result.vel_ecef = vel_ecef + other.vel_ecef;
        // Quaternion addition is component-wise for integration purposes
        result.quat_be.coeffs() = quat_be.coeffs() + other.quat_be.coeffs();
        result.omega_body = omega_body + other.omega_body;
        result.mass = mass + other.mass;
        return result;
    }

    CUDA_HOST_DEVICE State6DOF operator-(const State6DOF& other) const {
        State6DOF result;
        result.pos_ecef = pos_ecef - other.pos_ecef;
        result.vel_ecef = vel_ecef - other.vel_ecef;
        result.quat_be.coeffs() = quat_be.coeffs() - other.quat_be.coeffs();
        result.omega_body = omega_body - other.omega_body;
        result.mass = mass - other.mass;
        return result;
    }

    CUDA_HOST_DEVICE State6DOF operator*(double scalar) const {
        State6DOF result;
        result.pos_ecef = pos_ecef * scalar;
        result.vel_ecef = vel_ecef * scalar;
        result.quat_be.coeffs() = quat_be.coeffs() * scalar;
        result.omega_body = omega_body * scalar;
        result.mass = mass * scalar;
        return result;
    }
    
    CUDA_HOST_DEVICE void normalize() {
        quat_be.normalize();
    }
};

/**
 * @brief Forces and moments in Body frame
 */
struct ForcesMoments {
    Eigen::Vector3d force_body;    // Force in Body frame (N)
    Eigen::Vector3d moment_body;   // Moment in Body frame (N*m)
    double mass_flow_rate;         // Mass flow rate (kg/s)
};

/**
 * @brief Inertial properties
 */
struct InertialProps {
    double mass;                   // Mass (kg)
    Eigen::Matrix3d inertia;       // Moment of inertia in Body frame (kg*m^2)
    Eigen::Vector3d com;           // Center of mass offset (m)
};

class Dynamics6DOF {
public:
    /**
     * @brief Compute the state derivative (f = x_dot)
     * @param state Current state
     * @param fm Forces and moments in Body frame
     * @param inertia Inertial properties
     * @param gravity_ecef Gravity acceleration in ECEF frame (m/s^2)
     */
    static CUDA_HOST_DEVICE State6DOF compute_derivatives(
        const State6DOF& state,
        const ForcesMoments& fm,
        const InertialProps& inertia,
        const Eigen::Vector3d& gravity_ecef,
        SimulationProfile profile = SimulationProfile::GLOBAL_BALLISTIC
    );

    /**
     * @brief Integrate state using 4th order Runge-Kutta with dynamic force evaluation
     * @tparam SystemDynamics Functor type that computes derivative: State6DOF operator()(const State6DOF&, double t)
     * @param state Initial state
     * @param system Functor to compute derivatives (must include gravity, aero, etc.)
     * @param t0 Current time
     * @param dt Time step
     */
    template <typename SystemDynamics>
    static CUDA_HOST_DEVICE State6DOF integrate_rk4(
        const State6DOF& state,
        SystemDynamics& system,
        double t0,
        double dt
    ) {
        // k1 = f(t, y)
        State6DOF k1 = system(state, t0);
        
        // k2 = f(t + dt/2, y + k1 * dt/2)
        State6DOF s2 = state + k1 * (0.5 * dt);
        s2.normalize(); // Normalize quaternion after addition
        State6DOF k2 = system(s2, t0 + 0.5 * dt);
        
        // k3 = f(t + dt/2, y + k2 * dt/2)
        State6DOF s3 = state + k2 * (0.5 * dt);
        s3.normalize();
        State6DOF k3 = system(s3, t0 + 0.5 * dt);
        
        // k4 = f(t + dt, y + k3 * dt)
        State6DOF s4 = state + k3 * dt;
        s4.normalize();
        State6DOF k4 = system(s4, t0 + dt);
        
        // y_new = y + (k1 + 2k2 + 2k3 + k4) * dt / 6
        State6DOF next_state = state + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        next_state.normalize();
        
        return next_state;
    }

    /**
     * @brief Integrate state using Dormand-Prince 5(4) adaptive step size method
     * @tparam SystemDynamics Functor type
     * @param state Current state
     * @param system Functor to compute derivatives
     * @param t Current time (updated on return)
     * @param dt Current step size (updated to next suggested step size on return)
     * @param tol Relative tolerance
     * @return New state
     */
    template <typename SystemDynamics>
    static CUDA_HOST_DEVICE State6DOF integrate_adaptive(
        const State6DOF& state,
        SystemDynamics& system,
        double& t,
        double& dt,
        double tol = 1e-6
    ) {
        // DP54 Coefficients
        const double c2 = 1.0/5.0;
        const double a21 = 1.0/5.0;
        
        const double c3 = 3.0/10.0;
        const double a31 = 3.0/40.0;
        const double a32 = 9.0/40.0;
        
        const double c4 = 4.0/5.0;
        const double a41 = 44.0/45.0;
        const double a42 = -56.0/15.0;
        const double a43 = 32.0/9.0;
        
        const double c5 = 8.0/9.0;
        const double a51 = 19372.0/6561.0;
        const double a52 = -25360.0/2187.0;
        const double a53 = 64448.0/6561.0;
        const double a54 = -212.0/729.0;
        
        const double c6 = 1.0;
        const double a61 = 9017.0/3168.0;
        const double a62 = -355.0/33.0;
        const double a63 = 46732.0/5247.0;
        const double a64 = 49.0/176.0;
        const double a65 = -5103.0/18656.0;
        
        const double c7 = 1.0;
        const double a71 = 35.0/384.0;
        const double a72 = 0.0;
        const double a73 = 500.0/1113.0;
        const double a74 = 125.0/192.0;
        const double a75 = -2187.0/6784.0;
        const double a76 = 11.0/84.0;
        
        // 5th order solution weights (same as a7x because FSAL property, but k7 is the new k1)
        // Actually, for DP54, the result y_n+1 is calculated using these weights (b_i)
        // And y_hat (4th order) uses other weights.
        // The error is E = |y_n+1 - y_hat| = sum( (b_i - b_hat_i) * k_i )
        
        // Error weights (b_i - b_hat_i)
        const double e1 = 71.0/57600.0;
        const double e3 = -71.0/16695.0;
        const double e4 = 71.0/1920.0;
        const double e5 = -17253.0/339200.0;
        const double e6 = 22.0/525.0;
        const double e7 = -1.0/40.0;

        const double SAFETY = 0.9;
        const double MIN_SCALE = 0.1;
        const double MAX_SCALE = 5.0;
        const double MIN_DT = 1e-5;
        const double MAX_DT = 2.0; 

        // Make sure dt is positive (for now assuming forward integration)
        if (dt < MIN_DT) dt = MIN_DT;
        if (dt > MAX_DT) dt = MAX_DT;

        State6DOF final_state;
        bool accepted = false;
        
        // Pre-compute k1 once (optimizable if we store k from previous step, but keeping it simple)
        State6DOF k1 = system(state, t);

        while (!accepted) {
            // k2
            State6DOF s2 = state + k1 * (a21 * dt);
            s2.normalize();
            State6DOF k2 = system(s2, t + c2 * dt);
            
            // k3
            State6DOF s3 = state + k1 * (a31 * dt) + k2 * (a32 * dt);
            s3.normalize();
            State6DOF k3 = system(s3, t + c3 * dt);
            
            // k4
            State6DOF s4 = state + k1 * (a41 * dt) + k2 * (a42 * dt) + k3 * (a43 * dt);
            s4.normalize();
            State6DOF k4 = system(s4, t + c4 * dt);
            
            // k5
            State6DOF s5 = state + k1 * (a51 * dt) + k2 * (a52 * dt) + k3 * (a53 * dt) + k4 * (a54 * dt);
            s5.normalize();
            State6DOF k5 = system(s5, t + c5 * dt);
            
            // k6
            State6DOF s6 = state + k1 * (a61 * dt) + k2 * (a62 * dt) + k3 * (a63 * dt) + k4 * (a64 * dt) + k5 * (a65 * dt);
            s6.normalize();
            State6DOF k6 = system(s6, t + c6 * dt);
            
            // Result (5th order)
            // y_new = y + dt * (a71*k1 + ... + a76*k6)
            final_state = state + (k1 * a71 + k3 * a73 + k4 * a74 + k5 * a75 + k6 * a76) * dt;
            final_state.normalize();
            
            // k7 (FSAL: First Same As Last - k7 is k1 for next step if accepted)
            // But we need it for error estimate too? 
            // DP54 Error estimate uses k1..k7.
            // Wait, standard DP54 error coefficients E1..E7 involve k7?
            // Yes, b_i - b_hat_i usually involves k7.
            // The coefficients I used above (e1..e7) are correct for error estimation.
            State6DOF k7 = system(final_state, t + dt);
            
            // Error estimate
            State6DOF error = (k1 * e1 + k3 * e3 + k4 * e4 + k5 * e5 + k6 * e6 + k7 * e7) * dt;
            
            // Compute Max Relative Error
            double max_err = 0.0;
            double atol = 1e-6;
            
            auto check = [&](double e, double val1, double val2) {
                double scale = atol + std::max(std::abs(val1), std::abs(val2)) * tol;
                return std::abs(e) / scale;
            };

            for(int i=0; i<3; ++i) max_err = std::max(max_err, check(error.pos_ecef[i], state.pos_ecef[i], final_state.pos_ecef[i]));
            for(int i=0; i<3; ++i) max_err = std::max(max_err, check(error.vel_ecef[i], state.vel_ecef[i], final_state.vel_ecef[i]));
            // Treat quaternion as vector
            for(int i=0; i<4; ++i) max_err = std::max(max_err, check(error.quat_be.coeffs()[i], state.quat_be.coeffs()[i], final_state.quat_be.coeffs()[i]));
            for(int i=0; i<3; ++i) max_err = std::max(max_err, check(error.omega_body[i], state.omega_body[i], final_state.omega_body[i]));
            max_err = std::max(max_err, check(error.mass, state.mass, final_state.mass));
            
            if (max_err < 1.0) {
                accepted = true;
                t += dt;
                
                // Next step size
                if (max_err < 1.0e-10) max_err = 1.0e-10; // Avoid division by zero
                double scale = SAFETY * std::pow(max_err, -0.2); // Order 5 -> 1/5 = 0.2
                scale = std::max(MIN_SCALE, std::min(MAX_SCALE, scale));
                dt *= scale;
            } else {
                // Reject
                double scale = SAFETY * std::pow(max_err, -0.25); // Be more aggressive on reject? 
                // Usually just use same order or slightly lower
                scale = std::max(MIN_SCALE, std::min(MAX_SCALE, scale));
                dt *= scale;
                
                if (dt < MIN_DT) {
                    // Forced acceptance or abort? 
                    // For now, force accept but warn (or just clamp dt and continue)
                    // If we can't meet tolerance, we just take the step with min_dt.
                    dt = MIN_DT;
                    accepted = true;
                    t += dt;
                }
            }
        }
        
        return final_state;
    }
};

} // namespace aerosp
