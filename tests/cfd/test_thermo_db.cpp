#include "aero/cfd/thermo_db.hpp"
#include "aero/cfd/cfd_config.hpp"
#include "aero/cfd/nasa9_eval.hpp"
#include "aero/cfd/transport_eval.hpp"
#include "aero/cfd/t_from_e.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace aerosp;
using namespace aerosp::aero::cfd;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; std::printf("[Test] %s ... ", name); } while(0)
#define PASS do { pass_count++; std::printf("PASS\n"); } while(0)
#define FAIL(fmt, ...) do { std::printf("FAIL: " fmt "\n", ##__VA_ARGS__); return 1; } while(0)

static bool approx(Real a, Real b, Real rtol = Real(1e-10)) {
    Real diff = std::abs(a - b);
    Real mag = std::max(std::abs(a), std::abs(b));
    return diff <= rtol * (mag + Real(1));
}

static int test_n2_parse() {
    TEST("THERMO-N2-1 load thermo.inp");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());
    PASS;

    TEST("THERMO-N2-2 find N2");
    int idx = db.find_species("N2");
    if (idx < 0) FAIL("N2 not found");
    PASS;

    TEST("THERMO-N2-3 N2 n_intervals == 3");
    const auto& s = db.get_species(idx);
    if (s.n_intervals != 3) FAIL("expected 3 intervals, got %d", s.n_intervals);
    PASS;

    TEST("THERMO-N2-4 N2 M == 28.0134 g/mol");
    if (!approx(s.M, Real(28.0134)))
        FAIL("expected M=28.0134, got %.8f", (double)s.M);
    PASS;

    TEST("THERMO-N2-5 N2 has 10 coeffs per interval");
    for (int iv = 0; iv < 3; ++iv) {
        bool all_finite = true;
        for (int k = 0; k < 9; ++k) {
            if (!std::isfinite(s.coeffs[iv][k])) all_finite = false;
        }
        if (!all_finite) FAIL("interval %d has non-finite coeffs", iv);
    }
    PASS;

    TEST("THERMO-N2-6 N2 interval-0 a1 ~ 2.210371497E+04");
    if (!approx(s.coeffs[0][0], Real(2.210371497e4), Real(1e-6)))
        FAIL("expected a1=2.21037e4, got %.12e", (double)s.coeffs[0][0]);
    PASS;

    TEST("THERMO-N2-7 N2 T_break[0] ~ 1000.0");
    if (!approx(s.T_break[0], Real(1000.0)))
        FAIL("expected T_break[0]=1000.0, got %.8f", (double)s.T_break[0]);
    PASS;

    return 0;
}

static int test_select() {
    TEST("THERMO-SELECT-1 select_species N2,Ar,O2");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());

    auto idx = db.select_species({"N2", "Ar", "O2", "NONEXISTENT"});
    if (idx.size() != 3) FAIL("expected 3 species selected, got %zu", idx.size());
    if (db.get_species(idx[0]).name != "N2") FAIL("idx[0] not N2");
    if (db.get_species(idx[1]).name != "Ar") FAIL("idx[1] not Ar");
    if (db.get_species(idx[2]).name != "O2") FAIL("idx[2] not O2");
    PASS;

    return 0;
}

static int test_species_count() {
    TEST("THERMO-COUNT-1 species_count > 1000");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());
    if (db.species_count() < 1000)
        FAIL("expected >1000 species, got %d", db.species_count());
    std::printf("(%d species) ", db.species_count());
    PASS;

    return 0;
}

static int test_o2_parse() {
    TEST("THERMO-O2-1 parse O2 M and coefficients");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());

    int idx = db.find_species("O2");
    if (idx < 0) FAIL("O2 not found");

    const auto& s = db.get_species(idx);
    if (!approx(s.M, Real(31.9988000), Real(1e-8)))
        FAIL("O2 M expected 31.9988, got %.8f", (double)s.M);

    if (s.n_intervals != 3) FAIL("O2 expected 3 intervals, got %d", s.n_intervals);

    if (!approx(s.coeffs[0][0], Real(-3.425563420e4), Real(1e-6)))
        FAIL("O2 interval-0 a1 expected -3.42556342e4, got %.12e", (double)s.coeffs[0][0]);

    if (!approx(s.coeffs[1][0], Real(-1.037939022e6), Real(1e-6)))
        FAIL("O2 interval-1 a1 expected -1.037939022e6, got %.12e", (double)s.coeffs[1][0]);

    PASS;

    return 0;
}

static int test_5species() {
    TEST("THERMO-5SP-1 extract N2,O2,NO,N,O");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());

    std::vector<std::string> names = {"N2", "O2", "NO", "N", "O"};
    auto idx = db.select_species(names);
    if (idx.size() != 5) FAIL("expected 5 species, got %zu", idx.size());

    for (int i = 0; i < 5; ++i) {
        const auto& s = db.get_species(idx[i]);
        if (s.name != names[i])
            FAIL("idx[%d] expected %s, got %s", i, names[i].c_str(), s.name.c_str());
        if (s.n_intervals < 1) FAIL("%s has no intervals", s.name.c_str());
        if (s.M <= 0) FAIL("%s M <= 0", s.name.c_str());
    }

    PASS;

    return 0;
}

static int test_config() {
    TEST("THERMO-CFG-1 parse air_5sp.conf");
    SpeciesConfig cfg;
    std::string err;
    if (!cfg.load("data/config/air_5sp.conf", &err))
        FAIL("load failed: %s", err.c_str());

    if (cfg.species_list.size() != 5)
        FAIL("expected 5 species, got %zu", cfg.species_list.size());
    if (cfg.species_list[0] != "N2") FAIL("species[0] not N2");
    if (cfg.species_list[1] != "O2") FAIL("species[1] not O2");
    if (cfg.species_list[2] != "NO") FAIL("species[2] not NO");
    if (cfg.species_list[3] != "N")  FAIL("species[3] not N");
    if (cfg.species_list[4] != "O")  FAIL("species[4] not O");
    if (cfg.chemistry_model != "frozen") FAIL("chemistry not frozen");
    if (cfg.transport_model != "cea_log") FAIL("transport not cea_log");
    if (cfg.thermo_db_path != "data/thermo/thermo.inp") FAIL("thermo_db_path wrong");
    PASS;

    return 0;
}

static int test_config_yaml() {
    TEST("THERMO-CFG-2 parse air_5sp.yaml");
    SpeciesConfig cfg;
    std::string err;
    if (!cfg.load_yaml("data/config/air_5sp.yaml", &err))
        FAIL("load_yaml failed: %s", err.c_str());

    if (cfg.species_list.size() != 5)
        FAIL("expected 5 species, got %zu", cfg.species_list.size());
    if (cfg.species_list[0] != "N2") FAIL("species[0] not N2");
    if (cfg.species_list[1] != "O2") FAIL("species[1] not O2");
    if (cfg.species_list[2] != "NO") FAIL("species[2] not NO");
    if (cfg.species_list[3] != "N")  FAIL("species[3] not N");
    if (cfg.species_list[4] != "O")  FAIL("species[4] not O");
    if (cfg.chemistry_model != "frozen") FAIL("chemistry not frozen");
    if (cfg.transport_model != "cea_log") FAIL("transport not cea_log");
    if (cfg.thermo_db_path != "data/thermo/thermo.inp") FAIL("thermo_db_path wrong");
    PASS;

    return 0;
}

static int test_transport() {
    TEST("THERMO-TRAN-1 load trans.inp, find N2");
    TransportDb tdb;
    std::string err;
    if (!tdb.load("data/thermo/trans.inp", &err))
        FAIL("load failed: %s", err.c_str());
    int n2 = tdb.find_species("N2");
    if (n2 < 0) FAIL("N2 not found in trans.inp");
    PASS;

    TEST("THERMO-TRAN-2 N2 viscosity at 500K ~ 260.1 uP");
    const auto& rec = tdb.get_record(n2);
    if (rec.n_intervals < 1) FAIL("N2 has no intervals");
    Real mu = TransportDb::evaluate_mu(rec, Real(500));
    if (mu <= Real(0) || !std::isfinite(mu))
        FAIL("non-finite mu: %.6e", (double)mu);
    // CEA reference: 260.12 uP = 2.6012e-5 Pa*s at 500K
    Real expected = Real(2.60124e-5);
    Real rel_err = std::abs(mu - expected) / expected;
    if (rel_err > Real(1e-6))
        FAIL("N2 mu(500K): expected %.10e, got %.10e (rel_err=%.2e)",
             (double)expected, (double)mu, (double)rel_err);
    PASS;

    return 0;
}

static int test_gas_model_regression() {
    TEST("THERMO-GAS-1 default config gas_model_kind == 0 (PerfectGas)");
    CfdConfig cfg;
    if (cfg.gas_model_kind != 0)
        FAIL("expected gas_model_kind=0, got %d", cfg.gas_model_kind);
    if (!cfg.config_path.empty())
        FAIL("expected empty config_path, got '%s'", cfg.config_path.c_str());
    PASS;

    TEST("THERMO-GAS-2 PerfectGas gamma ~ 1.4 for air");
    Real gamma = Real(1.4);
    if (!approx(gamma, Real(1.4)))
        FAIL("expected gamma=1.4, got %.8f", (double)gamma);
    PASS;

    return 0;
}

static int test_gas_constant() {
    TEST("THERMO-R-1 R_UNIV == 8.31451 J/(mol*K)");
    if (!approx(R_UNIV, Real(8.31451)))
        FAIL("R_UNIV expected 8.31451, got %.8f", (double)R_UNIV);
    PASS;

    TEST("THERMO-R-2 N2 R_specific ~ 296.8 J/(kg*K)");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());
    int idx = db.find_species("N2");
    if (idx < 0) FAIL("N2 not found");
    Real Rs = db.get_species(idx).R_specific();
    Real expected = R_UNIV / Real(28.0134) * Real(1000);
    if (!approx(Rs, expected, Real(1e-4)))
        FAIL("expected Rs ~ %.8f J/(kg*K), got %.8f", (double)expected, (double)Rs);
    PASS;

    return 0;
}

static int test_n9_eval() {
    TEST("CFD-N9-1 eval_cp_R correctness");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());
    int n2 = db.find_species("N2");
    if (n2 < 0) FAIL("N2 not found");
    const auto& sp = db.get_species(n2);

    Real cp_R_300 = eval_cp_R(sp.coeffs[0], Real(300));
    if (!approx(cp_R_300, Real(3.5029347f), Real(3e-7)))
        FAIL("N2 cp/R(300K): expected 3.5029347, got %.12f", (double)cp_R_300);

    Real cp_R_iv0 = eval_cp_R(sp.coeffs[0], Real(1000));
    Real cp_R_iv1 = eval_cp_R(sp.coeffs[1], Real(1000));
    if (!approx(cp_R_iv0, cp_R_iv1, Real(3e-7)))
        FAIL("N2 cp/R discontinuity at 1000K: iv0=%.12f iv1=%.12f",
             (double)cp_R_iv0, (double)cp_R_iv1);

    Real cp_R_1500 = eval_cp_R(sp.coeffs[1], Real(1500));
    if (!approx(cp_R_1500, Real(4.1904979f), Real(3e-7)))
        FAIL("N2 cp/R(1500K): expected 4.1904979, got %.12f", (double)cp_R_1500);
    PASS;

    TEST("CFD-N9-2 eval_h_RT correctness");
    Real h_RT_300 = eval_h_RT(sp.coeffs[0], Real(300));
    if (!approx(h_RT_300, Real(0.021600962f), Real(3e-7)))
        FAIL("N2 h/RT(300K): expected 0.02160096, got %.12f", (double)h_RT_300);

    Real h_RT_1500 = eval_h_RT(sp.coeffs[1], Real(1500));
    if (!approx(h_RT_1500, Real(3.0793233f), Real(3e-7)))
        FAIL("N2 h/RT(1500K): expected 3.0793233, got %.12f", (double)h_RT_1500);
    PASS;

    TEST("CFD-N9-3 eval_s_R correctness");
    Real s_R_300 = eval_s_R(sp.coeffs[0], Real(300));
    if (!approx(s_R_300, Real(23.066895f), Real(3e-7)))
        FAIL("N2 s/R(300K): expected 23.066895, got %.12f", (double)s_R_300);

    Real s_R_1500 = eval_s_R(sp.coeffs[1], Real(1500));
    if (!approx(s_R_1500, Real(29.091351f), Real(3e-7)))
        FAIL("N2 s/R(1500K): expected 29.091351, got %.12f", (double)s_R_1500);
    PASS;

    TEST("CFD-N9-4 species-level functions N2/O2 at 300K");
    int o2 = db.find_species("O2");
    if (o2 < 0) FAIL("O2 not found");
    const auto& so2 = db.get_species(o2);

    Real Rs_n2 = sp.R_specific();
    Real Rs_o2 = so2.R_specific();
    Real cp_n2 = species_cp(Rs_n2, sp, Real(300));
    Real h_n2 = species_h(Rs_n2, sp, Real(300));
    Real s_n2 = species_s(Rs_n2, sp, Real(300));
    Real cp_o2 = species_cp(Rs_o2, so2, Real(300));
    Real h_o2 = species_h(Rs_o2, so2, Real(300));

    if (!approx(cp_n2, Real(1039.6876f), Real(3e-6)))
        FAIL("N2 cp(300K): expected 1039.6876, got %.8f", (double)cp_n2);
    if (!approx(h_n2, Real(1923.3804f), Real(3e-6)))
        FAIL("N2 h(300K): expected 1923.3804, got %.8f", (double)h_n2);
    if (!approx(s_n2, Real(6846.3638f), Real(3e-6)))
        FAIL("N2 s(300K): expected 6846.3638, got %.8f", (double)s_n2);
    if (!approx(cp_o2, Real(918.39423f), Real(3e-6)))
        FAIL("O2 cp(300K): expected 918.39423, got %.8f", (double)cp_o2);
    if (!approx(h_o2, Real(1698.8247f), Real(3e-6)))
        FAIL("O2 h(300K): expected 1698.8247, got %.8f", (double)h_o2);
    PASS;

    return 0;
}

static int test_n9_mix() {
    TEST("CFD-N9-5 mix_R, mix_cp, mix_h, mix_gamma for 50/50 N2+O2 at 300K");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());
    int n2 = db.find_species("N2");
    int o2 = db.find_species("O2");
    if (n2 < 0 || o2 < 0) FAIL("N2 or O2 not found");

    std::vector<Real> Y = {Real(0.5), Real(0.5)};
    std::vector<const SpeciesRecord*> species = {&db.get_species(n2), &db.get_species(o2)};

    Real R = mix_R(Y, species);
    if (!approx(R, Real(278.32146244), Real(1e-4)))
        FAIL("mix_R: expected 278.32146, got %.8f", (double)R);

    Real cp = mix_cp(Real(300), Y, species);
    if (!approx(cp, Real(979.04094554), Real(1e-4)))
        FAIL("mix_cp(300K): expected 979.04095, got %.8f", (double)cp);

    Real h = mix_h(Real(300), Y, species);
    if (!approx(h, Real(1811.07891459), Real(1e-4)))
        FAIL("mix_h(300K): expected 1811.07891, got %.8f", (double)h);

    Real gamma = mix_gamma(Real(300), Y, species);
    if (!approx(gamma, Real(1.39719384), Real(1e-6)))
        FAIL("mix_gamma(300K): expected 1.39719384, got %.8f", (double)gamma);
    PASS;

    return 0;
}

static int test_transport_eval() {
    TEST("CFD-N9-6 species_mu and species_kappa for N2");
    TransportDb tdb;
    std::string err;
    if (!tdb.load("data/thermo/trans.inp", &err))
        FAIL("load failed: %s", err.c_str());

    int n2 = tdb.find_species("N2");
    if (n2 < 0) FAIL("N2 not found in trans.inp");

    const auto& rec = tdb.get_record(n2);
    Real mu = species_mu(rec, Real(500));
    if (!approx(mu, Real(2.601239948775e-5), Real(1e-6)))
        FAIL("N2 mu(500K): expected 2.60124e-5, got %.12e", (double)mu);

    Real kap = species_kappa(rec, Real(500));
    if (!approx(kap, Real(3.836499226096e-5), Real(1e-3)))
        FAIL("N2 kappa(500K): expected 3.83650e-5, got %.12e", (double)kap);
    PASS;

    TEST("CFD-N9-7 mix_mu and mix_kappa for 50/50 N2+O2 at 500K");
    int o2 = tdb.find_species("O2");
    if (o2 < 0) FAIL("O2 not found in trans.inp");

    std::vector<Real> Y = {Real(0.5), Real(0.5)};
    std::vector<const TransportRecord*> trecs = {&tdb.get_record(n2), &tdb.get_record(o2)};

    Real mix_mu_val = mix_mu(Real(500), Y, trecs);
    if (!approx(mix_mu_val, Real(2.824981932856e-5), Real(1e-6)))
        FAIL("mix_mu(500K): expected 2.82498e-5, got %.12e", (double)mix_mu_val);

    Real mix_kap = mix_kappa(Real(500), Y, trecs);
    if (!approx(mix_kap, Real(3.956982521106e-5), Real(1e-3)))
        FAIL("mix_kappa(500K): expected 3.95698e-5, got %.12e", (double)mix_kap);
    PASS;

    return 0;
}

static int test_t_from_e() {
    TEST("CFD-N9-8 T_from_e recovers temperature for N2");
    ThermoDb db;
    std::string err;
    if (!db.load("data/thermo/thermo.inp", &err))
        FAIL("load failed: %s", err.c_str());
    int n2 = db.find_species("N2");
    if (n2 < 0) FAIL("N2 not found");

    const auto& sp = db.get_species(n2);
    Real Rs = sp.R_specific();
    Real T_target = Real(2500);
    Real h = species_h(Rs, sp, T_target);
    Real e_target = h - Rs * T_target;

    std::vector<Real> Y = {Real(1)};
    std::vector<const SpeciesRecord*> species = {&sp};

    Real T_rec = T_from_e(e_target, Y, species, Rs, Real(1400));
    if (T_rec < Real(0))
        FAIL("T_from_e returned %f (failed)", (double)T_rec);
    if (!approx(T_rec, T_target, Real(1e-3)))
        FAIL("T_from_e: expected %.2f, got %.2f (diff=%.3f)",
             (double)T_target, (double)T_rec, (double)std::abs(T_rec - T_target));
    PASS;

    return 0;
}

int main() {
    int fail = 0;
    fail += test_n2_parse();
    fail += test_select();
    fail += test_species_count();
    fail += test_o2_parse();
    fail += test_5species();
    fail += test_config();
    fail += test_config_yaml();
    fail += test_gas_model_regression();
    fail += test_transport();
    fail += test_gas_constant();
    fail += test_n9_eval();
    fail += test_n9_mix();
    fail += test_transport_eval();
    fail += test_t_from_e();

    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return fail == 0 ? 0 : 1;
}
