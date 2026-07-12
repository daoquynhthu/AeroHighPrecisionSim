#include "aero/panel/aero_solver.hpp"
#include "aero/engineering/aero_skin_friction.hpp"
#include "aero/engineering/engineering_aero.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>

namespace aerosp {
namespace aero {
namespace panel {
using namespace aerosp::aero::eng;

    // --- Single-point kernel ---
    __global__ void compute_forces_moments_kernel(const Triangle* triangles, int num_triangles,
                                          float mach, float alpha_rad, float beta_rad,
                                          float3 moment_ref_point, float gamma,
                                          float T_ref, float rho_ref, float mu_ref,
                                          float3* forces_out, float3* moments_out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_triangles) return;

        Triangle tri = triangles[idx];

        float ca = cosf(alpha_rad);
        float sa = sinf(alpha_rad);
        float cb = cosf(beta_rad);
        float sb = sinf(beta_rad);

        float3 flow_dir = make_float3(-ca * cb, -sb, -sa * cb);
        float cos_theta = dot(tri.normal, flow_dir);
        float gamma_eff = gamma_effective(mach);

        // 1. Newtonian pressure coefficient
        float Cp = 0.0f;
        float mach_sq = mach * mach;

        if (cos_theta < 0.0f) {
            float Cp_max = (gamma_eff + 3.0f) / (gamma_eff + 1.0f);
            float Cp_stag = Cp_max * (1.0f - 1.0f / (gamma_eff * mach_sq));
            Cp = Cp_stag * cos_theta * cos_theta;
        } else {
            Cp = -2.0f / (gamma_eff * mach_sq);
        }

        // 2. Viscous interaction + skin friction (same local conditions)
        float sin_theta = sqrtf(fmaxf(1.0f - cos_theta * cos_theta, 0.0f));
        float M_local = mach * sin_theta;
        float a_fs = sqrtf(gamma_eff * 287.058f * T_ref);
        float V_local = a_fs * M_local;
        float running_length = fmaxf(tri.body_axis_x + 0.05f, 0.01f);
        float Re_x = rho_ref * V_local * running_length / mu_ref;
        Cp = Cp + viscous_interaction_dCp(M_local, Re_x, gamma_eff);
        float Cf = van_driest_II_Cf_adiabatic(M_local, Re_x, gamma_eff);

        // 3. Total force = pressure + friction
        float3 force_term = tri.normal * (-Cp * tri.area);
        if (Cf > 0.0f) {
            float3 tan_dir = surface_flow_direction(tri.normal, flow_dir);
            force_term = force_term + tan_dir * (Cf * tri.area);
        }
        forces_out[idx] = force_term;

        float3 r = make_float3(tri.center.x - moment_ref_point.x,
                               tri.center.y - moment_ref_point.y,
                               tri.center.z - moment_ref_point.z);
        float3 moment_term = cross(r, force_term);
        moments_out[idx] = moment_term;
    }

    // --- Device-side engineering estimate ---
    __device__ void compute_engineering(
        float mach, float alpha_rad, float beta_rad,
        const AeroGeometry& geo,
        float& CX, float& CY, float& CZ,
        float& Cl, float& Cm, float& Cn,
        float& CD, float& CL)
    {
        float ca = cosf(alpha_rad);
        float sa = sinf(alpha_rad);
        float cb = cosf(beta_rad);
        float sb = sinf(beta_rad);

        float Cf = 0.074f / powf(1.0e7f, 0.2f);
        float CD_skin = Cf * geo.wet_area / geo.ref_area;

        float CD_base = 0.0f;
        if (geo.base_area > 0.0f) {
            float Ab_ratio = geo.base_area / geo.ref_area;
            float m2 = mach * mach;
            CD_base = (mach < 1.0f) ? 0.12f * Ab_ratio / (m2 + 0.1f)
                                    : 0.20f * Ab_ratio / m2;
        }

        float CD_wave = 0.0f;
        if (mach > 1.0f) {
            float beta = sqrtf(mach * mach - 1.0f);
            float ratio = 1.0f / (beta * geo.nose_fineness);
            CD_wave = (1.0f - cosf(2.0f * atanf(ratio))) * 0.5f;
            if (mach < 1.2f) {
                float peak = 0.15f * (1.0f + cosf(3.14159265f * (mach - 1.0f) / 0.4f));
                if (peak > CD_wave) CD_wave = peak;
            }
        }

        float AR = geo.ref_span * geo.ref_span / geo.ref_area;
        float CL_alpha_0 = 2.0f * 3.14159265f * AR / (2.0f + sqrtf(AR * AR + 4.0f));

        if (mach < 0.8f) {
            float beta_pg = sqrtf(fmaxf(1.0f - mach * mach, 0.01f));
            float CL_alpha = CL_alpha_0 / beta_pg;
            CL = CL_alpha * sinf(alpha_rad) * cosf(alpha_rad);
        } else if (mach < 1.2f) {
            float beta_sub = sqrtf(fmaxf(1.0f - 0.64f, 0.01f));
            float CL_sub = (CL_alpha_0 / beta_sub) * sinf(alpha_rad) * cosf(alpha_rad);
            float CL_sup = (4.0f / sqrtf(fmaxf(mach * mach - 1.0f, 0.01f)))
                          * sinf(alpha_rad) * cosf(alpha_rad);
            float t = (mach - 0.8f) / 0.4f;
            CL = (1.0f - t) * CL_sub + t * CL_sup;
        } else {
            float beta = sqrtf(fmaxf(mach * mach - 1.0f, 0.01f));
            float CL_alpha = 4.0f / beta;
            CL = CL_alpha * sinf(alpha_rad) * cosf(alpha_rad);
        }

        if (mach > 5.0f) {
            float CL_newt = 2.0f * sinf(alpha_rad) * sinf(alpha_rad) * cosf(alpha_rad);
            float t = fminf(1.0f, (mach - 5.0f) / 3.0f);
            CL = (1.0f - t) * CL + t * CL_newt;
        }

        float e = 0.8f;
        float CD_ind = CL * CL / (3.14159265f * e * AR);
        CD = CD_skin + CD_base + CD_wave + CD_ind;

        float CY_beta = -CL_alpha_0 * 0.5f;
        CY = CY_beta * sinf(beta_rad);

        float Fsx = -CD;
        float Fsz = -CL;

        CX = Fsx * ca * cb - CY * sb + Fsz * sa * cb;
        CZ = -Fsx * sa + Fsz * ca;

        float static_margin = 0.05f;
        Cm = -static_margin * CL;
        Cn = CY * static_margin;
        Cl = 0.0f;

        if (fabsf(beta_rad) < 1e-8f) {
            CY = 0.0f;
            Cn = 0.0f;
        }
    }

    // --- Batch kernel: one block per condition ---
    __global__ void batch_compute_kernel(
        const Triangle* triangles, int num_triangles,
        const BatchCondition* conditions, int num_conditions,
        float ref_area, float ref_length, float ref_span,
        float gamma,
        const AeroGeometry* eng_geo,
        BatchResult* results)
    {
        int cond = blockIdx.x;
        if (cond >= num_conditions) return;

        BatchCondition bc = conditions[cond];
        float mach = bc.mach;
        float alpha_rad = bc.alpha_deg * 3.14159265f / 180.0f;
        float beta_rad  = bc.beta_deg  * 3.14159265f / 180.0f;
            float3 mrp = make_float3(bc.com_x, bc.com_y, bc.com_z);

        float CX=0, CY=0, CZ=0, Cl=0, Cm=0, Cn=0, CL=0, CD=0;

        if (mach >= 5.0f) {
            // --- Newtonian panel method with shared memory reduction ---
            __shared__ float smem[6 * 256];
            float* sfx = smem;
            float* sfy = smem + 256;
            float* sfz = smem + 256*2;
            float* smx = smem + 256*3;
            float* smy = smem + 256*4;
            float* smz = smem + 256*5;

            int tid = threadIdx.x;
            float3 lf = {0,0,0}, lm = {0,0,0};
            float ca = cosf(alpha_rad), sa = sinf(alpha_rad);
            float cb = cosf(beta_rad),  sb = sinf(beta_rad);
            float3 flow_dir = make_float3(-ca*cb, -sb, -sa*cb);
            float mach_sq = mach * mach;
            float gamma_eff = gamma_effective(mach);
            float Cp_max_stag = ((gamma_eff + 3.0f) / (gamma_eff + 1.0f))
                               * (1.0f - 1.0f / (gamma_eff * mach_sq));

            for (int i = tid; i < num_triangles; i += blockDim.x) {
                Triangle tri = triangles[i];
                float ct = dot(tri.normal, flow_dir);
                float Cp;
                if (ct < 0.0f) {
                    Cp = Cp_max_stag * ct * ct;
                } else {
                    Cp = -2.0f / (gamma_eff * mach_sq);
                }

                // Viscous interaction + skin friction (same local conditions)
                float sin_theta = sqrtf(fmaxf(1.0f - ct * ct, 0.0f));
                float M_local = mach * sin_theta;
                float a_fs = sqrtf(gamma_eff * 287.058f * bc.T_ref);
                float V_local = a_fs * M_local;
                float running_length = fmaxf(tri.body_axis_x + 0.05f, 0.01f);
                float Re_x = bc.rho_ref * V_local * running_length / bc.mu_ref;

                Cp = Cp + viscous_interaction_dCp(M_local, Re_x, gamma_eff);
                float3 f = tri.normal * (-Cp * tri.area);

                float Cf = van_driest_II_Cf_adiabatic(M_local, Re_x, gamma_eff);
                if (Cf > 0.0f) {
                    float3 tan_dir = surface_flow_direction(tri.normal, flow_dir);
                    f = f + tan_dir * (Cf * tri.area);
                }

                lf = lf + f;
                float3 r = make_float3(tri.center.x - mrp.x,
                                       tri.center.y - mrp.y,
                                       tri.center.z - mrp.z);
                lm = lm + cross(r, f);
            }

            sfx[tid] = lf.x; sfy[tid] = lf.y; sfz[tid] = lf.z;
            smx[tid] = lm.x; smy[tid] = lm.y; smz[tid] = lm.z;
            __syncthreads();

            for (int s = blockDim.x/2; s > 0; s >>= 1) {
                if (tid < s) {
                    sfx[tid] += sfx[tid+s]; sfy[tid] += sfy[tid+s]; sfz[tid] += sfz[tid+s];
                    smx[tid] += smx[tid+s]; smy[tid] += smy[tid+s]; smz[tid] += smz[tid+s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                float3 tf = {sfx[0], sfy[0], sfz[0]};
                float3 tm = {smx[0], smy[0], smz[0]};
                CX = tf.x / ref_area;
                CY = tf.y / ref_area;
                CZ = tf.z / ref_area;
                Cl = tm.x / (ref_area * ref_span);
                Cm = tm.y / (ref_area * ref_length);
                Cn = tm.z / (ref_area * ref_span);

                if (fabsf(bc.beta_deg) < 1e-6f) {
                    CY = 0.0f;
                    Cn = 0.0f;
                }

                // Base drag correction (turbulent base pressure correlation)
                CX = CX + base_drag_CX_correction(mach, gamma_eff,
                     eng_geo->base_area, ref_area);

                float Fsx = CX * ca * cb + CY * sb + CZ * sa * cb;
                float Fsz = -CX * sa + CZ * ca;
                CD = -Fsx;
                CL = -Fsz;
            }
        } else {
            // --- Engineering estimate (per-block, thread 0) ---
            if (threadIdx.x == 0) {
                float e_CX, e_CY, e_CZ, e_Cl, e_Cm, e_Cn, e_CD, e_CL;
                compute_engineering(mach, alpha_rad, beta_rad, *eng_geo,
                                    e_CX, e_CY, e_CZ, e_Cl, e_Cm, e_Cn, e_CD, e_CL);

                // Engineering result (CPU-side blend handles Mach 4-6 transition)
                CX = e_CX; CY = e_CY; CZ = e_CZ;
                Cl = e_Cl; Cm = e_Cm; Cn = e_Cn;
                CL = e_CL; CD = e_CD;
            }
        }

        if (threadIdx.x == 0) {
            results[cond] = {CX, CY, CZ, Cl, Cm, Cn, CL, CD};
        }
    }

    // Constructor / Destructor
    AeroSolver::AeroSolver() {}
    AeroSolver::~AeroSolver() {
        if (d_triangles) cudaFree(d_triangles);
        if (d_forces) cudaFree(d_forces);
        if (d_moments) cudaFree(d_moments);
    }

    bool AeroSolver::load_mesh(const std::vector<Triangle>& mesh, float ref_area_in, float ref_length_in, float ref_span_in) {
        this->ref_area = ref_area_in;
        this->ref_length = ref_length_in;
        this->ref_span = ref_span_in;

        if (mesh.empty()) return false;

        if (d_triangles) cudaFree(d_triangles);
        if (d_forces) cudaFree(d_forces);
        if (d_moments) cudaFree(d_moments);

        num_triangles = mesh.size();

        // Ensure body_axis_x is set (running length from nose tip along body axis)
        std::vector<Triangle> h_triangles = mesh;
        for (auto& t : h_triangles) {
            t.body_axis_x = t.center.x;
        }

        cudaMalloc(&d_triangles, num_triangles * sizeof(Triangle));
        cudaMalloc(&d_forces, num_triangles * sizeof(float3));
        cudaMalloc(&d_moments, num_triangles * sizeof(float3));

        cudaMemcpy(d_triangles, h_triangles.data(), num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
        return true;
    }

    bool AeroSolver::load_model(const std::string& stl_path, float ref_area_in, float ref_length_in, float ref_span_in) {
        std::vector<Triangle> host_triangles = parse_stl(stl_path);
        if (host_triangles.empty()) return false;

        if (load_mesh(host_triangles, ref_area_in, ref_length_in, ref_span_in)) {
             std::cout << "Loaded STL Model: " << stl_path << " (" << num_triangles << " triangles) to GPU." << std::endl;
             return true;
        }
        return false;
    }

    AeroCoefficients AeroSolver::compute_coefficients(float mach, float alpha_deg, float beta_deg) {
        float alpha_rad = alpha_deg * 3.14159265359f / 180.0f;
        float beta_rad = beta_deg * 3.14159265359f / 180.0f;

        // Reference freestream conditions (ISA at 30 km for hypersonic cruise)
        float T_ref, p_ref, rho_ref;
        isa_atmosphere(30000.0f, T_ref, p_ref, rho_ref);
        float mu_ref = sutherland_viscosity(T_ref);

        int threadsPerBlock = 256;
        int blocksPerGrid = (num_triangles + threadsPerBlock - 1) / threadsPerBlock;

        compute_forces_moments_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_triangles, num_triangles, mach, alpha_rad, beta_rad,
            moment_ref_point, gamma, T_ref, rho_ref, mu_ref, d_forces, d_moments
        );
        cudaDeviceSynchronize();

        std::vector<float3> h_forces(num_triangles);
        std::vector<float3> h_moments(num_triangles);

        cudaMemcpy(h_forces.data(), d_forces, num_triangles * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_moments.data(), d_moments, num_triangles * sizeof(float3), cudaMemcpyDeviceToHost);

        float3 total_force_term = {0,0,0};
        float3 total_moment_term = {0,0,0};

        for(int i=0; i<num_triangles; ++i) {
            total_force_term = total_force_term + h_forces[i];
            total_moment_term = total_moment_term + h_moments[i];
        }

        AeroCoefficients coeffs;

        coeffs.CX = total_force_term.x / ref_area;
        coeffs.CY = total_force_term.y / ref_area;
        coeffs.CZ = total_force_term.z / ref_area;

        coeffs.Cl = total_moment_term.x / (ref_area * ref_span);
        coeffs.Cm = total_moment_term.y / (ref_area * ref_length);
        coeffs.Cn = total_moment_term.z / (ref_area * ref_span);

        if (std::abs(beta_deg) < 1e-6f) {
            coeffs.CY = 0.0f;
            coeffs.Cn = 0.0f;
        }

        // Base drag correction: replace Newtonian base Cp (-2/(γM²)) with
        // correlated base pressure (p_base/p_∞ = 0.18 + 0.10/M² for turbulent)
        if (mach >= 0.1f) {
            coeffs.CX = coeffs.CX + base_drag_CX_correction(mach,
                gamma_effective(mach), base_area, ref_area);
        }

        float ca = cosf(alpha_rad);
        float sa = sinf(alpha_rad);
        float cb = cosf(beta_rad);
        float sb = sinf(beta_rad);

        float Fsx = coeffs.CX * ca * cb + coeffs.CY * sb + coeffs.CZ * sa * cb;
        float Fsz = -coeffs.CX * sa + coeffs.CZ * ca;

        coeffs.CD = -Fsx;
        coeffs.CL = -Fsz;

        if (std::abs(coeffs.CD) > 1e-6f)
            coeffs.L_D = coeffs.CL / coeffs.CD;
        else
            coeffs.L_D = 0.0f;

        return coeffs;
    }

    // --- Batch compute: all conditions in a single GPU pass ---
    std::vector<BatchResult> AeroSolver::compute_batch(
        const std::vector<BatchCondition>& conditions,
        const AeroGeometry& eng_geo)
    {
        int n = static_cast<int>(conditions.size());
        if (n == 0) return {};

        // Fill in freestream conditions (ISA reference altitude per Mach)
        std::vector<BatchCondition> conds = conditions;
        for (int i = 0; i < n; ++i) {
            float alt = 30000.0f;
            if (conds[i].mach < 1.5f)      alt = 5000.0f;
            else if (conds[i].mach < 4.0f) alt = 15000.0f;
            else if (conds[i].mach < 8.0f) alt = 25000.0f;
            else if (conds[i].mach < 15.0f) alt = 40000.0f;
            else if (conds[i].mach < 20.0f) alt = 55000.0f;
            else                            alt = 65000.0f;
            float T, p, rho;
            isa_atmosphere(alt, T, p, rho);
            conds[i].T_ref = T;
            conds[i].rho_ref = rho;
            conds[i].mu_ref = sutherland_viscosity(T);
        }

        BatchCondition* d_cond = nullptr;
        BatchResult* d_res = nullptr;
        AeroGeometry* d_geo = nullptr;

        cudaMalloc(&d_cond, n * sizeof(BatchCondition));
        cudaMalloc(&d_res,  n * sizeof(BatchResult));
        cudaMalloc(&d_geo,  sizeof(AeroGeometry));

        cudaMemcpy(d_cond, conds.data(), n * sizeof(BatchCondition), cudaMemcpyHostToDevice);
        cudaMemcpy(d_geo,  &eng_geo, sizeof(AeroGeometry), cudaMemcpyHostToDevice);

        batch_compute_kernel<<<n, 256>>>(
            d_triangles, num_triangles,
            d_cond, n,
            ref_area, ref_length, ref_span,
            gamma,
            d_geo,
            d_res
        );
        cudaDeviceSynchronize();

        std::vector<BatchResult> results(n);
        cudaMemcpy(results.data(), d_res, n * sizeof(BatchResult), cudaMemcpyDeviceToHost);

        cudaFree(d_cond);
        cudaFree(d_res);
        cudaFree(d_geo);

        // CPU-side blending for transition region (Mach 4.0-6.0)
        // Smoothly blends engineering estimate and GPU Newtonian panel results.
        for (int i = 0; i < n; ++i) {
            double mach = static_cast<double>(conditions[i].mach);
            if (mach >= 4.0 && mach <= 6.0) {
                double alpha_rad = conditions[i].alpha_deg * 3.141592653589793 / 180.0;
                double beta_rad  = conditions[i].beta_deg  * 3.141592653589793 / 180.0;
                auto eng = compute_engineering_coeffs(eng_geo, mach, alpha_rad, beta_rad);

                if (mach >= 5.0) {
                    // Blend: t=0 (M=5) → 50% eng + 50% GPU, t=1 (M=6) → 100% GPU
                    double t = (mach - 4.0) / 2.0;
                    t = std::max(0.0, std::min(1.0, t));
                    auto blend = [t](double a, double b) { return (1.0 - t) * a + t * b; };
                    results[i].CX = static_cast<float>(blend(eng.CX, results[i].CX));
                    results[i].CY = static_cast<float>(blend(eng.CY, results[i].CY));
                    results[i].CZ = static_cast<float>(blend(eng.CZ, results[i].CZ));
                    results[i].Cl = static_cast<float>(blend(eng.Cl, results[i].Cl));
                    results[i].Cm = static_cast<float>(blend(eng.Cm, results[i].Cm));
                    results[i].Cn = static_cast<float>(blend(eng.Cn, results[i].Cn));
                    results[i].CL = static_cast<float>(blend(eng.CL, results[i].CL));
                    results[i].CD = static_cast<float>(blend(eng.CD, results[i].CD));
                } else {
                    // Mach 4-5: override GPU device engineering with CPU engineering
                    results[i].CX = static_cast<float>(eng.CX);
                    results[i].CY = static_cast<float>(eng.CY);
                    results[i].CZ = static_cast<float>(eng.CZ);
                    results[i].Cl = static_cast<float>(eng.Cl);
                    results[i].Cm = static_cast<float>(eng.Cm);
                    results[i].Cn = static_cast<float>(eng.Cn);
                    results[i].CL = static_cast<float>(eng.CL);
                    results[i].CD = static_cast<float>(eng.CD);
                }
            }
        }

        return results;
    }

    std::vector<Triangle> AeroSolver::parse_stl(const std::string& path) {
        std::vector<Triangle> triangles;
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open STL file: " << path << std::endl;
            return triangles;
        }

        file.seekg(80);

        uint32_t count;
        file.read(reinterpret_cast<char*>(&count), 4);

        for (uint32_t i = 0; i < count; ++i) {
            Triangle t;
            float normal[3];
            float v0[3], v1[3], v2[3];
            uint16_t attr;

            file.read(reinterpret_cast<char*>(normal), 12);
            file.read(reinterpret_cast<char*>(v0), 12);
            file.read(reinterpret_cast<char*>(v1), 12);
            file.read(reinterpret_cast<char*>(v2), 12);
            file.read(reinterpret_cast<char*>(&attr), 2);

            t.normal = make_float3(normal[0], normal[1], normal[2]);
            t.v0 = make_float3(v0[0], v0[1], v0[2]);
            t.v1 = make_float3(v1[0], v1[1], v1[2]);
            t.v2 = make_float3(v2[0], v2[1], v2[2]);

            float3 e1 = t.v1 - t.v0;
            float3 e2 = t.v2 - t.v0;
            float3 cp = cross(e1, e2);
            t.area = 0.5f * sqrtf(dot(cp, cp));
            t.center = (t.v0 + t.v1 + t.v2) * (1.0f/3.0f);

            float len_cp = sqrtf(dot(cp, cp));
            if (len_cp > 1e-8f) {
                t.normal = cp * (1.0f / len_cp);
            }

            t.body_axis_x = t.center.x;

            triangles.push_back(t);
        }
        return triangles;
    }

}
}
}
