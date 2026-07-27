#pragma once

#include "aero/cfd/real_fwd.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

// CEA standard gas constant, J/(mol*K)
inline constexpr Real R_UNIV = Real(8.31451);

// NASA-9 polynomial: up to 3 temperature intervals, 9 coefficients each
inline constexpr int NASA9_NCOEF = 9;
inline constexpr int NASA9_NINTV = 3;

// Transport (CEA log form): up to 3 intervals, 4 coefficients each
inline constexpr int CEA4_NCOEF = 4;
inline constexpr int CEA4_NINTV = 3;

struct SpeciesRecord {
    std::string name;
    Real M;                            // molar mass [kg/kmol]
    Real T_break[NASA9_NINTV - 1];     // break points between intervals
    int n_intervals;                   // 1, 2, or 3
    Real coeffs[NASA9_NINTV][NASA9_NCOEF]; // [interval][a1..a7, b1, b2]

    bool has_cp_data() const { return n_intervals > 0; }
    Real R_specific() const { return R_UNIV / M * Real(1000); } // [J/(kg*K)]
};

struct TransportRecord {
    std::string name;
    Real M{Real(0)};               // molar mass [kg/kmol]; set from ThermoDb after load
    int n_intervals;
    Real T_min[CEA4_NINTV];       // low end of each temperature interval
    Real T_max[CEA4_NINTV];       // high end of each temperature interval
    Real mu_coeffs[CEA4_NINTV][CEA4_NCOEF];     // ln(mu) = A*ln(T) + B/T + C/T^2 + D
    Real kappa_coeffs[CEA4_NINTV][CEA4_NCOEF];  // ln(lambda) = A*ln(T) + B/T + C/T^2 + D
};

// ─── ThermoDb: load + query thermodynamic data ─────────────────────

class ThermoDb {
public:
    bool load(const std::string& path, std::string* error = nullptr);

    int species_count() const { return static_cast<int>(species_.size()); }
    const SpeciesRecord& get_species(int idx) const { return species_[idx]; }

    int find_species(const std::string& name) const;

    // Extract a subset of species (by name). Returns vector of indices into
    // the internal database. Names not found are skipped.
    std::vector<int> select_species(const std::vector<std::string>& names) const;

private:
    std::vector<SpeciesRecord> species_;
    bool parse_thermo_inp(const std::string& path, std::string* error);
};

// ─── TransportDb: load + query transport data ──────────────────────

class TransportDb {
public:
    bool load(const std::string& path, std::string* error = nullptr);

    int record_count() const { return static_cast<int>(records_.size()); }
    const TransportRecord& get_record(int idx) const { return records_[idx]; }

    int find_species(const std::string& name) const;

    // Evaluate viscosity [Pa*s] using CEA log formula
    static Real evaluate_mu(const TransportRecord& rec, Real T);

    // Set molecular weights from matching ThermoDb species (by name)
    // Species names not found in either database are skipped.
    void set_molecular_weights(const ThermoDb& thdb, const std::vector<std::string>& names);

private:
    std::vector<TransportRecord> records_;
    bool parse_trans_inp(const std::string& path, std::string* error);
};

// ─── SpeciesConfig: species subset + model selection ───────────────

struct SpeciesConfig {
    std::vector<std::string> species_list;
    std::string chemistry_model;   // "frozen", "equilibrium", "finite_rate"
    std::string transport_model;   // "sutherland", "cea_log", "constant"
    std::string thermo_db_path;
    std::string trans_db_path;

    bool load(const std::string& config_path, std::string* error = nullptr);
    bool load_yaml(const std::string& config_path, std::string* error = nullptr);
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
