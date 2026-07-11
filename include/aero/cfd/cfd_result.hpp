#pragma once

#include "aero/cfd/real_fwd.hpp"
#include <string>

namespace aerosp {
namespace aero {
namespace cfd {

struct CfdForceResult {
    Real CX = 0.0f;
    Real CY = 0.0f;
    Real CZ = 0.0f;
    Real CD = 0.0f;
    Real CL = 0.0f;
    Real Cl = 0.0f;
    Real Cm = 0.0f;
    Real Cn = 0.0f;
    Real Q_wall = 0.0f;
    Real q_stag = 0.0f;
    int iterations = 0;
    Real residual = 0.0f;
    std::string fidelity = "cfd-cpu";
    std::string turbulence_model = "laminar";
};



} // namespace cfd
} // namespace aero
} // namespace aerosp

