#include "sim/atmosphere/atmosphere_model.hpp"
#include "infra/math/constants.hpp"
#include <cmath>
#include <iostream>

#if AEROSP_HAS_NRLMSISE00
extern "C" {
    // NRLMSISE-00 Fortran Subroutine
    // SUBROUTINE GTD7(IYD,SEC,ALT,GLAT,GLONG,STL,F107A,F107,AP,MASS,D,T)
    void gtd7_(int* iyd, float* sec, float* alt, float* glat, float* glong,               float* stl, float* f107a, float* f107, float* ap, int* mass,               float* d, float* t);
}
#endif

namespace aerosp {

/**
 * @brief Simple exponential atmosphere model (fallback)
 * rho = rho0 * exp(-h/H)
 * p = p0 * exp(-h/H)
 */
__host__ __device__ AtmosphereData AtmosphereModel::calculate_simple(double alt_m) {
    constexpr double p0 = 101325.0;      // Pa
    constexpr double T0 = 288.15;        // Sea level temperature (K)
    constexpr double L = -0.0065;        // Lapse rate (K/m)

    AtmosphereData data;

    if (alt_m < 0.0) alt_m = 0.0;

    if (alt_m < 11000.0) { // Troposphere
        data.temperature = T0 + L * alt_m;
        data.pressure = p0 * pow(data.temperature / T0, -sim::coord::G() / (L * (sim::atmosphere::R_GAS() / sim::atmosphere::M_AIR())));
        data.density = data.pressure / ((sim::atmosphere::R_GAS() / sim::atmosphere::M_AIR()) * data.temperature);
    } else { // Simplified stratosphere
        double T_tropo = T0 + L * 11000.0;
        double P_tropo = p0 * pow(T_tropo / T0, -sim::coord::G() / (L * (sim::atmosphere::R_GAS() / sim::atmosphere::M_AIR())));
        data.temperature = T_tropo;
        data.pressure = P_tropo * exp(-sim::coord::G() * (alt_m - 11000.0) / ((sim::atmosphere::R_GAS() / sim::atmosphere::M_AIR()) * T_tropo));
        data.density = data.pressure / ((sim::atmosphere::R_GAS() / sim::atmosphere::M_AIR()) * data.temperature);
    }

    data.sound_speed = sqrt(1.4 * (sim::atmosphere::R_GAS() / sim::atmosphere::M_AIR()) * data.temperature);

    return data;
}

__host__ __device__ AtmosphereData AtmosphereModel::calculate_ussa76(double z) {
    // US Standard Atmosphere 1976 Constants
    const double R_earth = 6356766.0; // Effective radius for geopotential height (m)
    const double g0 = 9.80665;        // Standard gravity (m/s^2)
    const double M0 = 0.0289644;      // Mean molecular weight (kg/mol)
    const double R_gas = 8.314462618; // Universal gas constant (J/(mol K))

    // Geopotential altitude H (m)
    // H = R * z / (R + z)
    double H = (R_earth * z) / (R_earth + z);

    // Layer parameters: Base H (m), Lapse Rate (K/m), Base Temp (K), Base Pressure (Pa)
    double H_b, L_b, T_b, P_b;

    if (H < 11000.0) {
        H_b = 0.0;     L_b = -0.0065; T_b = 288.15;   P_b = 101325.0;
    } else if (H < 20000.0) {
        H_b = 11000.0; L_b = 0.0;     T_b = 216.65;   P_b = 22632.10;
    } else if (H < 32000.0) {
        H_b = 20000.0; L_b = 0.0010;  T_b = 216.65;   P_b = 5474.89;
    } else if (H < 47000.0) {
        H_b = 32000.0; L_b = 0.0028;  T_b = 228.65;   P_b = 868.02;
    } else if (H < 51000.0) {
        H_b = 47000.0; L_b = 0.0;     T_b = 270.65;   P_b = 110.91;
    } else if (H < 71000.0) {
        H_b = 51000.0; L_b = -0.0028; T_b = 270.65;   P_b = 66.94;
    } else if (H < 84852.0) {
        H_b = 71000.0; L_b = -0.0020; T_b = 214.65;   P_b = 3.96;
    } else {
        // Above 86km, use simple exponential decay based on last layer
        // Or just return last layer properties if very high (not accurate but safe)
        // Better: Extrapolate last layer (Mesopause)
        H_b = 71000.0; L_b = -0.0020; T_b = 214.65;   P_b = 3.96;
    }

    double T_M, P_M;

    if (fabs(L_b) < 1e-9) { // Isothermal layer
        T_M = T_b;
        P_M = P_b * exp(-g0 * M0 * (H - H_b) / (R_gas * T_b));
    } else {
        T_M = T_b + L_b * (H - H_b);
        // Avoid negative temperature
        if (T_M < 0.0) T_M = 0.1;
        P_M = P_b * pow(T_b / T_M, (g0 * M0) / (R_gas * L_b));
    }

    AtmosphereData data;
    data.temperature = T_M;
    data.pressure = P_M;
    data.density = (P_M * M0) / (R_gas * T_M);
    data.sound_speed = sqrt(1.4 * R_gas * T_M / M0); // Gamma = 1.4

    return data;
}

/**
 * @brief NRLMSISE-00 implementation
 */
__host__ __device__ AtmosphereData AtmosphereModel::calculate(const NRLMSISE00Input& input) {
#if !defined(__CUDA_ARCH__) && AEROSP_HAS_NRLMSISE00
    // HOST CODE: Call NRLMSISE-00 DLL

    // Prepare inputs
    int iyd = input.year * 1000 + input.doy;
    float sec = static_cast<float>(input.sec);
    float alt = static_cast<float>(input.alt);
    float glat = static_cast<float>(input.lat);
    float glong = static_cast<float>(input.lon);
    float stl = static_cast<float>(input.lst);
    float f107a = static_cast<float>(input.f107A);
    float f107 = static_cast<float>(input.f107);

    float ap[7];
    for(int i=0; i<7; ++i) {
        ap[i] = static_cast<float>(input.ap_vector[i]);
        if (ap[i] == 0.0f && input.ap != 0.0f) {
             ap[i] = static_cast<float>(input.ap); // Fallback to daily AP
        }
    }

    int mass = 48; // Total mass density
    float d[9];    // Output densities
    float t[2];    // Output temperatures

    // Call Fortran subroutine
    gtd7_(&iyd, &sec, &alt, &glat, &glong, &stl, &f107a, &f107, ap, &mass, d, t);

    // Process outputs
    AtmosphereData data;
    data.temperature = static_cast<double>(t[1]); // T(2) is temp at alt

    // D(6) is total mass density in g/cm^3. Convert to kg/m^3.
    // 1 g/cm^3 = 1000 kg/m^3
    data.density = static_cast<double>(d[5]) * 1000.0;

    // Calculate Pressure
    // P = sum(n_i) * k_B * T
    // n_i in cm^-3 -> convert to m^-3 (multiply by 1e6)
    // k_B = 1.380649e-23 J/K

    double sum_n = 0.0;
    // Sum species: He(1), O(2), N2(3), O2(4), Ar(5), H(7), N(8), Anomalous O(9)
    // Note: D array indices are 0-based in C++, so D(1) is d[0]
    sum_n += d[0]; // He
    sum_n += d[1]; // O
    sum_n += d[2]; // N2
    sum_n += d[3]; // O2
    sum_n += d[4]; // Ar
    sum_n += d[6]; // H
    sum_n += d[7]; // N
    sum_n += d[8]; // Anomalous O

    double n_total_m3 = sum_n * 1.0e6;
    constexpr double k_B = 1.380649e-23;

    data.pressure = n_total_m3 * k_B * data.temperature;

    // Sound speed
    // a = sqrt(gamma * P / rho)
    // Assume gamma = 1.4 (good approx for lower thermosphere, may vary higher up)
    if (data.density > 0.0) {
        data.sound_speed = sqrt(1.4 * data.pressure / data.density);
    } else {
        data.sound_speed = 0.0;
    }

    return data;

#else
    // DEVICE CODE or NRLMSISE-00 unavailable: Fallback to USSA76
    return calculate_ussa76(input.alt * 1000.0);
#endif
}

} // namespace aerosp
