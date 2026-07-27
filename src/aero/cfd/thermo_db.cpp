#include "aero/cfd/thermo_db.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef AEROSP_HAS_YAML_CPP
#  include <yaml-cpp/yaml.h>
#endif

namespace aerosp {
namespace aero {
namespace cfd {

namespace {

static Real parse_fortran_float(const char* field, int len) {
    char buf[64];
    int j = 0;
    for (int i = 0; i < len && field[i] != '\0'; ++i) {
        char c = field[i];
        if (c == 'D' || c == 'd') c = 'E';
        if (j < (int)sizeof(buf) - 2) buf[j++] = c;
    }
    buf[j] = '\0';
    if (j == 0) return Real(0);
    char* end = nullptr;
    double val = std::strtod(buf, &end);
    return static_cast<Real>(val);
}

static Real parse_fortran_float(const char* s) {
    return parse_fortran_float(s, (int)std::strlen(s));
}

static Real col_float(const char* line, int start, int end) {
    if (end <= start) return Real(0);
    int e = end;
    while (e > start && (line[e - 1] == ' ' || line[e - 1] == '\t')) --e;
    if (e <= start) return Real(0);
    return parse_fortran_float(line + start, e - start);
}

static int col_int(const char* line, int start, int end) {
    char buf[32];
    int j = 0;
    for (int i = start; i < end && line[i] != '\0'; ++i) {
        if (line[i] >= '0' && line[i] <= '9') buf[j++] = line[i];
    }
    buf[j] = '\0';
    if (j == 0) return 0;
    return std::atoi(buf);
}

static std::string col_str(const char* line, int start, int end) {
    int s = start, e = end;
    while (s < e && line[s] == ' ') ++s;
    while (e > s && line[e - 1] == ' ') --e;
    if (e <= s) return {};
    return std::string(line + s, (size_t)(e - s));
}

static void read_5_coeffs(const char* line, Real out[5]) {
    out[0] = col_float(line, 0, 16);
    out[1] = col_float(line, 16, 32);
    out[2] = col_float(line, 32, 48);
    out[3] = col_float(line, 48, 64);
    out[4] = col_float(line, 64, 80);
}

static void read_cont_coeffs(const char* line, Real out[4]) {
    out[0] = col_float(line, 0, 16);
    out[1] = col_float(line, 16, 32);
    out[2] = col_float(line, 48, 64);
    out[3] = col_float(line, 64, 80);
}

} // anonymous namespace

bool ThermoDb::load(const std::string& path, std::string* error) {
    species_.clear();
    return parse_thermo_inp(path, error);
}

int ThermoDb::find_species(const std::string& name) const {
    for (int i = 0; i < (int)species_.size(); ++i) {
        if (species_[i].name == name) return i;
    }
    return -1;
}

std::vector<int> ThermoDb::select_species(const std::vector<std::string>& names) const {
    std::vector<int> indices;
    for (const auto& n : names) {
        int idx = find_species(n);
        if (idx >= 0) indices.push_back(idx);
    }
    return indices;
}

bool ThermoDb::parse_thermo_inp(const std::string& path, std::string* error) {
    std::vector<std::string> lines;
    {
        FILE* fp = std::fopen(path.c_str(), "r");
        if (!fp) {
            if (error) *error = "cannot open " + path;
            return false;
        }
        char buf[4096];
        while (std::fgets(buf, (int)sizeof(buf), fp)) {
            size_t len = std::strlen(buf);
            while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = '\0';
            if (len > 0) lines.emplace_back(buf, len);
        }
        std::fclose(fp);
    }
    if (lines.empty()) {
        if (error) *error = "empty file " + path;
        return false;
    }

    size_t li = 0;
    while (li < lines.size()) {
        std::string trimmed = lines[li];
        trimmed.erase(0, trimmed.find_first_not_of(" \t\r"));
        if (trimmed == "thermo" || trimmed == "THERMO") break;
        ++li;
    }
    if (li >= lines.size()) {
        li = 0;
        while (li < lines.size() && (lines[li].empty() || lines[li][0] == '!'))
            ++li;
    }

    while (li < lines.size() && (lines[li].empty() || lines[li][0] == '!'))
        ++li;
    ++li;

    while (li < lines.size()) {
        while (li < lines.size() && (lines[li].empty() || lines[li][0] == '!'))
            ++li;
        if (li >= lines.size()) break;

        std::string upper = lines[li];
        for (auto& c : upper) c = (char)std::toupper((unsigned char)c);
        if (upper.find("END") == 0 || upper.find("END REACTANTS") == 0)
            break;

        std::string line1 = lines[li].size() < 80 ? lines[li] + std::string(80 - lines[li].size(), ' ') : lines[li];
        std::string name = col_str(line1.c_str(), 0, 16);
        if (name.empty()) { ++li; continue; }
        ++li;
        if (li >= lines.size()) break;

        std::string line2 = lines[li].size() < 80 ? lines[li] + std::string(80 - lines[li].size(), ' ') : lines[li];
        ++li;
        const char* m = line2.c_str();

        int n_intervals = 0;
        {
            std::string ns(m, 2);
            for (char c : ns) {
                if (c >= '1' && c <= '9') { n_intervals = (int)(c - '0'); break; }
                if (c >= '0' && c <= '0') { n_intervals = 0; break; }
            }
        }

        Real M = col_float(m, 52, 65);
        Real h_f = col_float(m, 65, 80);

        if (n_intervals <= 0) {
            if (li < lines.size()) ++li;
            continue;
        }
        if (n_intervals > NASA9_NINTV) n_intervals = NASA9_NINTV;

        SpeciesRecord rec;
        rec.name = name;
        rec.M = M;
        rec.n_intervals = 0;

        for (int iv = 0; iv < n_intervals; ++iv) {
            if (li + 2 >= lines.size()) break;

            std::string tline = lines[li].size() < 80 ? lines[li] + std::string(80 - lines[li].size(), ' ') : lines[li];
            ++li;
            const char* t = tline.c_str();

            Real T_low = col_float(t, 0, 12);
            Real T_high = col_float(t, 12, 22);

            if (iv == 0) {
                rec.T_break[0] = T_high;
            } else if (iv == 1) {
                rec.T_break[1] = T_high;
            }

            std::string c1 = lines[li].size() < 80 ? lines[li] + std::string(80 - lines[li].size(), ' ') : lines[li];
            ++li;
            Real a[5];
            read_5_coeffs(c1.c_str(), a);

            std::string c2 = lines[li].size() < 80 ? lines[li] + std::string(80 - lines[li].size(), ' ') : lines[li];
            ++li;
            Real cont[4];
            read_cont_coeffs(c2.c_str(), cont);

            rec.coeffs[iv][0] = a[0]; rec.coeffs[iv][1] = a[1];
            rec.coeffs[iv][2] = a[2]; rec.coeffs[iv][3] = a[3];
            rec.coeffs[iv][4] = a[4]; rec.coeffs[iv][5] = cont[0];
            rec.coeffs[iv][6] = cont[1];
            rec.coeffs[iv][7] = cont[2];
            rec.coeffs[iv][8] = cont[3];

            ++rec.n_intervals;
        }

        if (rec.n_intervals > 0)
            species_.push_back(std::move(rec));
    }

    return species_count() > 0;
}

// ─── TransportDb ──────────────────────────────────────────────────

bool TransportDb::load(const std::string& path, std::string* error) {
    records_.clear();
    return parse_trans_inp(path, error);
}

int TransportDb::find_species(const std::string& name) const {
    for (int i = 0; i < (int)records_.size(); ++i) {
        if (records_[i].name == name) return i;
    }
    return -1;
}

bool TransportDb::parse_trans_inp(const std::string& path, std::string* error) {
    std::vector<std::string> lines;
    {
        FILE* fp = std::fopen(path.c_str(), "r");
        if (!fp) {
            if (error) *error = "cannot open " + path;
            return false;
        }
        char buf[4096];
        while (std::fgets(buf, (int)sizeof(buf), fp)) {
            size_t len = std::strlen(buf);
            while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = '\0';
            lines.emplace_back(buf, len);
        }
        std::fclose(fp);
    }
    if (lines.empty()) {
        if (error) *error = "empty file " + path;
        return false;
    }

    size_t li = 0;
    if (li < lines.size()) ++li;

    while (li < lines.size()) {
        while (li < lines.size() && lines[li].empty()) { ++li; }
        if (li >= lines.size()) break;
        if (lines[li].size() < 16) { ++li; continue; }

        std::string line = lines[li];

        std::string sp1 = col_str(line.c_str(), 0, 16);
        if (sp1.empty()) { ++li; continue; }

        std::string sp2 = col_str(line.c_str(), 16, 32);
        std::string full_name = sp2.empty() ? sp1 : sp1 + " " + sp2;

        TransportRecord rec;
        rec.name = full_name;
        rec.n_intervals = 0;

        ++li;

        int v_count = 0;
        int c_count = 0;

        while (li < lines.size()) {
            const std::string& vcline = lines[li];
            if (vcline.empty() || vcline[0] != ' ') break;

            if (vcline.size() < 76) { ++li; continue; }
            if (vcline[1] != 'V' && vcline[1] != 'C') { ++li; continue; }

            bool is_viscosity = (vcline[1] == 'V');

            float fields[6];
            int nf = 0;
            const char* p = vcline.c_str() + 2;
            while (*p && nf < 6) {
                while (*p == ' ' || *p == '\t') ++p;
                if (!*p) break;
                const char* start = p;
                while (*p && *p != ' ' && *p != '\t') {
                    ++p;
                }
                std::string field_str(start, p);
                for (auto& c : field_str) if (c == 'D' || c == 'd') c = 'E';
                if (!field_str.empty() && (field_str.back() == 'E' || field_str.back() == 'e')) {
                    while (*p == ' ' || *p == '\t') ++p;
                    const char* sign_start = p;
                    if (*p == '+' || *p == '-') ++p;
                    while (*p >= '0' && *p <= '9') ++p;
                    if (p > sign_start) field_str += std::string(sign_start, p);
                }
                fields[nf++] = (float)std::atof(field_str.c_str());
            }

            if (nf >= 6) {
                Real T_low = fields[0], T_high = fields[1];
                Real A = fields[2], B = fields[3];
                Real C = fields[4], D = fields[5];

                if (is_viscosity) {
                    if (v_count < CEA4_NINTV) {
                        int iv = v_count;
                        rec.T_min[iv] = T_low;
                        rec.T_max[iv] = T_high;
                        rec.mu_coeffs[iv][0] = A;
                        rec.mu_coeffs[iv][1] = B;
                        rec.mu_coeffs[iv][2] = C;
                        rec.mu_coeffs[iv][3] = D;
                        ++v_count;
                    }
                } else {
                    if (c_count < CEA4_NINTV) {
                        int iv = c_count;
                        rec.T_min[iv] = T_low;
                        rec.T_max[iv] = T_high;
                        rec.kappa_coeffs[iv][0] = A;
                        rec.kappa_coeffs[iv][1] = B;
                        rec.kappa_coeffs[iv][2] = C;
                        rec.kappa_coeffs[iv][3] = D;
                        ++c_count;
                    }
                }
            }

            ++li;
        }

        rec.n_intervals = (std::max)(v_count, c_count);

        if (rec.n_intervals > 0)
            records_.push_back(std::move(rec));
    }

    return record_count() > 0;
}

// ─── TransportDb viscosity evaluation ─────────────────────────────

Real TransportDb::evaluate_mu(const TransportRecord& rec, Real T) {
    if (rec.n_intervals < 1) return Real(0);

    int iv = 0;
    for (int i = 0; i < rec.n_intervals - 1; ++i) {
        if (T >= rec.T_max[i]) iv = i + 1;
    }

    const Real* c = rec.mu_coeffs[iv];
    Real ln_mu = c[0] * std::log(T) + c[1] / T + c[2] / (T * T) + c[3];
    return std::exp(ln_mu) * Real(1e-7);  // micropoise → Pa*s
}

void TransportDb::set_molecular_weights(const ThermoDb& thdb,
                                         const std::vector<std::string>& names) {
    for (const auto& name : names) {
        int ti = find_species(name);
        if (ti < 0) continue;
        int si = thdb.find_species(name);
        if (si < 0) continue;
        records_[ti].M = thdb.get_species(si).M;
    }
}

// ─── SpeciesConfig key=value parser ───────────────────────────────

bool SpeciesConfig::load(const std::string& config_path, std::string* error) {
    FILE* fp = std::fopen(config_path.c_str(), "r");
    if (!fp) {
        if (error) *error = "cannot open " + config_path;
        return false;
    }

    char line[256];
    while (std::fgets(line, (int)sizeof(line), fp)) {
        size_t len = std::strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0 || line[0] == '#') continue;

        const char* eq = std::strchr(line, '=');
        if (!eq) continue;

        std::string key(line, (size_t)(eq - line));
        while (!key.empty() && key.back() == ' ') key.pop_back();

        std::string value(eq + 1);
        while (!value.empty() && value[0] == ' ') value.erase(value.begin());

        if (key == "thermo_db") {
            thermo_db_path = value;
        } else if (key == "trans_db") {
            trans_db_path = value;
        } else if (key == "species") {
            species_list.clear();
            const char* p = value.c_str();
            while (*p) {
                while (*p == ' ') ++p;
                if (!*p) break;
                const char* start = p;
                while (*p && *p != ' ') ++p;
                species_list.push_back(std::string(start, p));
            }
        } else if (key == "chemistry") {
            chemistry_model = value;
        } else if (key == "transport") {
            transport_model = value;
        }
    }

    std::fclose(fp);
    return !species_list.empty() || !thermo_db_path.empty();
}

bool SpeciesConfig::load_yaml(const std::string& config_path, std::string* error) {
#ifdef AEROSP_HAS_YAML_CPP
    try {
        YAML::Node root = YAML::LoadFile(config_path);

        if (root["thermo_db"])
            thermo_db_path = root["thermo_db"].as<std::string>();

        if (root["trans_db"])
            trans_db_path = root["trans_db"].as<std::string>();

        if (root["species"] && root["species"].IsSequence()) {
            for (const auto& s : root["species"])
                species_list.push_back(s.as<std::string>());
        }

        if (root["chemistry"])
            chemistry_model = root["chemistry"].as<std::string>();

        if (root["transport"])
            transport_model = root["transport"].as<std::string>();

        return !species_list.empty() || !thermo_db_path.empty();
    } catch (const std::exception& e) {
        if (error) *error = e.what();
        return false;
    }
#else
    (void)config_path;
    if (error) *error = "yaml-cpp not available (AEROSP_HAS_YAML_CPP=0)";
    return false;
#endif
}

} // namespace cfd
} // namespace aero
} // namespace aerosp
