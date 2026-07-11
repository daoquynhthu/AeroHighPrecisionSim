#pragma once

#include "aero/cfd/real_fwd.hpp"
#include "aero/cfd/cfd_mesh.hpp"

#include <string>
#include <vector>

namespace aerosp {
namespace aero {
namespace cfd {

struct PartitionInfo {
    std::vector<int> partition_owner;   // [cell] → rank 0..n_ranks-1
    int n_ranks = 1;
    int my_rank = 0;

    std::vector<int> owned_cells;
    std::vector<int> ghost_cells;       // cells owned by other ranks, needed locally
    std::vector<int> ghost_owner_rank;  // [ghost_idx] → rank that owns this ghost
    std::vector<int> send_cells;        // local cells that other ranks need as ghosts
    std::vector<int> send_rank;         // [send_idx] → rank that needs this cell

    std::size_t n_owned() const { return owned_cells.size(); }
    std::size_t n_ghost() const { return ghost_cells.size(); }
};

struct GpuPartition {
    int* d_partition_owner = nullptr;  // [n_cells]
    int* d_ghost_indices = nullptr;     // [n_ghost]
    int* d_ghost_owner = nullptr;       // [n_ghost]
    int n_ghost = 0;
    int my_rank = 0;

    void release();
};

bool partition_linear(const CfdMesh& mesh, int n_ranks, PartitionInfo& info, std::string* error = nullptr);
bool upload_partition_to_device(const PartitionInfo& info, GpuPartition& gpu_part, std::string* error = nullptr);

std::vector<int> build_ghost_from_partition(const CfdMesh& mesh, const PartitionInfo& info);

} // namespace cfd
} // namespace aero
} // namespace aerosp
