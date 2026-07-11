#pragma once

#include "aero/cfd/real_fwd.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

class GpuCommunicator {
public:
    GpuCommunicator() = default;
    ~GpuCommunicator();

    GpuCommunicator(const GpuCommunicator&) = delete;
    GpuCommunicator& operator=(const GpuCommunicator&) = delete;

    GpuCommunicator(GpuCommunicator&& other) noexcept;
    GpuCommunicator& operator=(GpuCommunicator&& other) noexcept;

    bool initialize(int* argc, char*** argv, std::string* error = nullptr);
    void finalize();

    int rank() const { return rank_; }
    int size() const { return size_; }
    bool is_mpi_mode() const { return initialized_; }

    bool assign_device(int devices_per_node, std::string* error = nullptr);

    bool send_recv_exchange(const Real* send_buf, Real* recv_buf, int count,
        int peer_rank, int tag, std::string* error = nullptr);

    bool allreduce_min(Real& value, std::string* error = nullptr);
    bool allreduce_sum(Real& value, std::string* error = nullptr);
    bool allreduce_sum(int& value, std::string* error = nullptr);
    bool allreduce_sum(Real* values, int count, std::string* error = nullptr);

    bool barrier(std::string* error = nullptr);

private:
    bool initialized_ = false;
    int rank_ = 0;
    int size_ = 1;
};

} // namespace cfd
} // namespace aero
} // namespace aerosp
