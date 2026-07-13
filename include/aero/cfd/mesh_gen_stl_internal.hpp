#pragma once

#include "aero/cfd/real.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {
namespace stl_internal {

struct Vec3 { Real x, y, z; };
Vec3 operator+(Vec3 a, Vec3 b);
Vec3 operator-(Vec3 a, Vec3 b);
Vec3 operator*(Vec3 a, Real s);
Vec3 operator/(Vec3 a, Real s);
Real dot(Vec3 a, Vec3 b);
Real norm(Vec3 a);
Vec3 cross(Vec3 a, Vec3 b);
Vec3 normalize(Vec3 a);

Real volume_tet_signed(Vec3 a, Vec3 b, Vec3 c, Vec3 d);
void hex_to_6_tets(Vec3 hex_v[8], Vec3 tet_v[6][4]);

struct Tri { Vec3 v[3]; Vec3 n; };
struct StlHeader { char data[80]; int num_tri; };
int detect_stl_format(const std::string& path, StlHeader& hdr_out);
std::vector<Tri> parse_stl(const std::string& path, std::string* error);

} // namespace stl_internal
} // namespace cfd
} // namespace aero
} // namespace aerosp
