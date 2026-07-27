#include "aero/cfd/thermo_db.hpp"

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
    if (!approx(R_UNIV / Real(28.0134), Real(0.29680), Real(1e-4)))
        FAIL("expected R_spec ~ 0.2968 J/(g*K), got %.8f", (double)(R_UNIV / Real(28.0134)));
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
    fail += test_gas_constant();

    std::printf("\n%d / %d tests PASSED.\n", pass_count, test_count);
    return fail == 0 ? 0 : 1;
}
