// Copied and trimmed from the Hopper forward params path in
// include/min_fa3_varlen_params.h. DCP_MEGA fields describe the persistent
// packed-varlen DCP queues and node-local IPC views.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "min_fa3_varlen_params.h"

namespace min_fa3_varlen_demo::dcp_mega {

enum AttentionKind : int32_t {
    kChunk = 0,
    kHistory = 1,
};

enum PhaseTimestampIndex : int32_t {
    kKernelStartTimestamp = 0,
    kQAllGatherDoneTimestamp = 1,
    kAttentionDoneTimestamp = 2,
    kHistoryCombineDoneTimestamp = 3,
    kPublishDoneTimestamp = 4,
    kReceiveDoneTimestamp = 5,
    kFinalCombineDoneTimestamp = 6,
    kKernelDoneTimestamp = 7,
    kPhaseTimestampCount = 8,
};

struct AttentionWorkDesc {
    int32_t kind;
    int32_t batch_idx;
    int32_t m_block;
    int32_t head;
    int32_t split_idx;
    int32_t q_dependency_begin;
    int32_t q_dependency_count;
    int32_t completion_id;
};

struct QTaskDesc {
    int32_t src_rank;
    // The fixed 16-token communication layout requires this to be zero.
    int32_t local_head;
    int32_t token_begin;
    int32_t valid_rows;
};

struct PublishWorkDesc {
    int32_t dst_rank;
    // 16-token tile flattened over CommHeads.
    int32_t vector_begin;
    int32_t valid_vectors;
    int32_t dependency_begin;
    int32_t dependency_count;
    int32_t pack_gqa;
    int32_t combine_task_count;
    int32_t reserved1;
};

struct HistoryCombineWorkDesc {
    int32_t publish_id;
    int32_t vector_begin;
    int32_t valid_vectors;
    int32_t dependency_begin;
    int32_t dependency_count;
    int32_t batch_idx;
    int32_t actual_splits;
    int32_t reserved;
};

struct FinalWorkDesc {
    // Final compute subtask flattened over CommHeads.
    int32_t vector_begin;
    int32_t valid_vectors;
    int32_t dependency_begin;
    int32_t dependency_count;
    // Parent 16-token communication tile shared by sibling final subtasks.
    int32_t parent_token_block;
    int32_t reserved1;
    int32_t reserved2;
    int32_t reserved3;
};

struct MetadataHeader {
    int32_t version;
    int32_t attention_count;
    int32_t q_task_count;
    int32_t q_dependency_count;
    int32_t publish_count;
    int32_t publish_dependency_count;
    int32_t final_count;
    int32_t final_dependency_count;
    int32_t chunk_attention_count;
    int32_t history_attention_count;
    int32_t total_q;
    int32_t total_vectors;
    int32_t effective_num_splits;
    int32_t chunk_num_splits;
    int32_t history_num_splits;
    int32_t pack_gqa;
    int32_t split;
    int32_t block_n;
    int32_t batch_size;
    int32_t pre_phase;
    int32_t post_phase;
    int32_t attention_offset;
    int32_t q_tasks_offset;
    int32_t q_dependencies_offset;
    int32_t publish_offset;
    int32_t publish_dependencies_offset;
    int32_t final_offset;
    int32_t final_dependencies_offset;
    int32_t chunk_splits_offset;
    int32_t history_splits_offset;
    int32_t used_ints;
    int32_t reserved;
    // Retains the metadata v2 slot; fixed-layout images use marker value 1.
    int32_t reserved_v2_communication_layout;
    int32_t token_block_count;
    int32_t q_ready_count;
    int32_t receive_count;
    int32_t tile_ready_count;
    int32_t dcp_size;
    int32_t history_combine_count;
    int32_t history_combine_offset;
};

static_assert(sizeof(AttentionWorkDesc) == 8 * sizeof(int32_t));
static_assert(sizeof(QTaskDesc) == 4 * sizeof(int32_t));
static_assert(sizeof(PublishWorkDesc) == 8 * sizeof(int32_t));
static_assert(sizeof(HistoryCombineWorkDesc) == 8 * sizeof(int32_t));
static_assert(sizeof(FinalWorkDesc) == 8 * sizeof(int32_t));
static_assert(sizeof(MetadataHeader) == 40 * sizeof(int32_t));
static_assert(sizeof(uint64_t) == 8 && alignof(uint64_t) >= 8);
static_assert(std::is_trivially_copyable_v<AttentionWorkDesc>);
static_assert(std::is_trivially_copyable_v<MetadataHeader>);

struct DCPMega_fwd_params {
    // DCP_MEGA: the two bundles retain the copied Flash_fwd_params layout and
    // differ only in Q/K/V, mask, head count, and epilogue destinations.
    Flash_fwd_params chunk{};
    Flash_fwd_params history{};

    void const* ipc_q_ptrs[8]{};
    void* ipc_history_send_o_ptrs[8]{};
    float* ipc_history_send_lse_ptrs[8]{};
    int32_t* ipc_tile_ready_ptrs[8]{};
    int32_t* ipc_barrier_ptrs[8]{};

    void* q_group_ptr = nullptr;
    void* history_receive_o_ptr = nullptr;
    float* history_receive_lse_ptr = nullptr;
    void* final_o_ptr = nullptr;
    float* final_lse_ptr = nullptr;
    int64_t chunk_lse_head_stride = 0;
    int64_t history_lse_head_stride = 0;
    int64_t final_lse_head_stride = 0;

    int32_t const* metadata = nullptr;
    MetadataHeader metadata_header{};
    bool dynamic_metadata = false;
    int32_t* q_ready = nullptr;
    int32_t* attention_done = nullptr;
    int32_t* publish_ready = nullptr;
    int32_t* receive_ready = nullptr;
    int32_t* queue_state = nullptr;
    uint64_t* phase_timestamps = nullptr;
    int32_t const* graph_post_phase = nullptr;

    int dcp_size = 0;
    int dcp_rank = 0;
    int hq_local = 0;
    int tile_ready_phase = 0;
    int ipc_q_token_capacity = 0;
    int ipc_vector_capacity = 0;
    int ipc_token_block_capacity = 0;
    int device = 0;
    int num_sms = 0;
    int num_comm_sm = 0;
    bool return_lse = false;
    // Optional reusable benchmark events. The typed launcher records them
    // immediately around the kernel command after all host-side preparation.
    cudaEvent_t timing_start = nullptr;
    cudaEvent_t timing_end = nullptr;
};

static_assert(alignof(DCPMega_fwd_params) >= alignof(void*));

void run_dcp_mega_barrier(
    DCPMega_fwd_params const& params,
    int phase,
    cudaStream_t stream);

void advance_dcp_mega_graph_phase(
    int32_t* graph_post_phase,
    cudaStream_t stream);

void run_dcp_mega_graph_barrier(
    DCPMega_fwd_params const& params,
    int phase_offset,
    cudaStream_t stream);

void run_dcp_mega_varlen_fwd(
    DCPMega_fwd_params& params,
    cudaStream_t stream);

}  // namespace min_fa3_varlen_demo::dcp_mega
