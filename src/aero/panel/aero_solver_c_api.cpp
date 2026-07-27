#include "aero/panel/aero_solver.hpp"
#include "aero/panel/aero_solver_c_api.h"
#include <string>
#include <cstring>
#include <vector>

static thread_local std::string s_last_error;

int generate_aero_table_c(
    const char* stl_path,
    const char* csv_path,
    const double* mach_grid, int n_mach,
    const double* alpha_grid, int n_alpha,
    const double* beta_grid, int n_beta,
    float ref_area, float ref_length, float ref_span,
    float com_x)
{
    aerosp::aero::panel::AeroTableConfig cfg;
    cfg.ref_area = ref_area;
    cfg.ref_length = ref_length;
    cfg.ref_span = ref_span;
    cfg.com_x = com_x;

    std::vector<double> mach_vec(mach_grid, mach_grid + n_mach);
    std::vector<double> alpha_vec(alpha_grid, alpha_grid + n_alpha);
    std::vector<double> beta_vec(beta_grid, beta_grid + n_beta);

    bool ok = aerosp::aero::panel::generate_aero_table(
        stl_path, csv_path, mach_vec, alpha_vec, beta_vec, cfg);

    if (!ok) {
        s_last_error = "generate_aero_table failed";
        return -1;
    }
    return 0;
}

const char* aero_solver_last_error(void) {
    return s_last_error.c_str();
}
