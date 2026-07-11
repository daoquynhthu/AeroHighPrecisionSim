#pragma once

#include "aero/cfd/real_fwd.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct GpuDeviceInfo {
    int device_id = -1;
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int multi_processor_count = 0;
    std::size_t total_global_mem = 0;
    int memory_clock_rate = 0;
    int memory_bus_width = 0;
    bool supports_limited_atomic64 = false;
};

struct GpuTopology {
    std::vector<GpuDeviceInfo> devices;
    std::vector<std::vector<bool>> peer_access;   // n×n, [i][j] = can i access j directly
    std::vector<std::vector<int>> nvlink_count;   // n×n, link count between i and j

    int device_count() const { return static_cast<int>(devices.size()); }

    bool has_nvlink(int i, int j) const {
        if (i < 0 || i >= device_count() || j < 0 || j >= device_count()) return false;
        return nvlink_count[i][j] > 0;
    }

    int best_peer(int i) const;
    std::vector<int> select_devices(int n) const;

    std::string bandwidth_report() const;
    std::string to_string() const;
};

GpuTopology detect_gpu_topology();

} // namespace cfd
} // namespace aero
} // namespace aerosp
