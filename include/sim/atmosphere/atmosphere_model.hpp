#pragma once

#include "infra/common.hpp"

namespace aerosp {

/**
 * @brief Atmospheric properties
 */
struct AtmosphereData {
    double density;      // kg/m^3
    double pressure;     // Pa
    double temperature;  // K
    double sound_speed;  // m/s
};

/**
 * @brief NRLMSISE-00 input parameters
 */
struct NRLMSISE00Input {
    int year;          // Year, e.g., 2026
    int doy;           // Day of year, 1-366
    double sec;        // Seconds in day (UT)
    double alt;        // Altitude (km)
    double lat;        // Latitude (deg)
    double lon;        // Longitude (deg)
    double lst;        // Local solar time (hours)
    double f107A;      // 81 day average of F10.7 flux
    double f107;       // Daily F10.7 flux for previous day
    double ap;         // Magnetic index (daily)
    double ap_vector[7]; // Detailed magnetic index (optional)
};

class AtmosphereModel {
public:
    /**
     * @brief Calculate atmospheric data using NRLMSISE-00 model
     * @note This version is optimized for performance and can run on CPU/GPU
     */
    static CUDA_HOST_DEVICE AtmosphereData calculate(const NRLMSISE00Input& input);

    /**
     * @brief US Standard Atmosphere 1976 (USSA76) model
     * @details High fidelity model up to 86km (Geopotential altitude)
     *          Uses 7 layers defined by standard lapse rates.
     * @param altitude_m Geopotential altitude (m). Note: Input is geometric, converted internally?
     *                   Standard defines layers in geopotential height.
     */
    static CUDA_HOST_DEVICE AtmosphereData calculate_ussa76(double altitude_geometric_m);

    /**
     * @brief Simplified atmosphere model (Exponential model) for quick testing
     */
    static CUDA_HOST_DEVICE AtmosphereData calculate_simple(double altitude_m);
};

} // namespace aerosp
