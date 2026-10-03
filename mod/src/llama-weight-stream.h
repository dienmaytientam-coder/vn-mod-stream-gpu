// SPDX-License-Identifier: MIT
// Copyright (c) 2026 dienmaytientam-coder

#pragma once

#include "ggml-cpp.h"
#include "llama.h"
#include <functional>

#include <array>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

enum class llama_weight_stream_capability {
    STREAMABLE,
    MUST_RESIDENT,
    UNKNOWN,
    STATE_NOT_WEIGHT,
};

// Profile data has no runtime handles or source addresses.
struct llama_weight_stream_tensor_profile {
    std::string name;
    int block_index = -1;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, GGML_MAX_DIMS> ne = {};
    std::array<size_t, GGML_MAX_DIMS> nb = {};
    size_t logical_bytes = 0;
    size_t stored_bytes = 0;
    size_t allocation_bytes = 0;
    size_t alignment = 0;
    bool cuda_weight = false;
    llama_weight_stream_capability capability = llama_weight_stream_capability::UNKNOWN;
};

struct llama_weight_stream_block_profile {
    int block_index = -1;
    std::vector<llama_weight_stream_tensor_profile> tensors;
    size_t streamable_bytes = 0;
    size_t resident_bytes = 0;
    size_t unknown_bytes = 0;
};

struct llama_weight_stream_model_profile {
    uint32_t schema_version = 1;
    uint32_t profile_version = 2;
    std::string architecture;
    std::string model_fingerprint;
    std::vector<llama_weight_stream_block_profile> blocks;
    std::vector<llama_weight_stream_tensor_profile> non_block_tensors;
    std::vector<int> demand_order;
};

struct llama_weight_stream_tensor_layout {
    llama_weight_stream_tensor_profile profile;
    size_t offset = 0;
};

struct llama_weight_stream_block_layout {
    int block_index = -1;
    size_t alignment = 0;
    size_t slot_bytes = 0;
    std::vector<llama_weight_stream_tensor_layout> tensors;
};

llama_weight_stream_tensor_profile llama_weight_stream_describe(const ggml_tensor & tensor, int block_index, llama_weight_stream_capability capability);
llama_weight_stream_block_profile llama_weight_stream_profile_block(int block_index, std::vector<llama_weight_stream_tensor_profile> tensors);
void llama_weight_stream_validate_profile(const llama_weight_stream_model_profile & model);
llama_weight_stream_block_layout llama_weight_stream_pack(const llama_weight_stream_block_profile & block, ggml_backend_buffer_type_t buft);
void llama_weight_stream_validate(const llama_weight_stream_block_layout & block, size_t capacity);

// stage() borrows sources; persist() drops them after the initial pack.
struct llama_weight_stream_source {
    std::string name;
    const void * data = nullptr;
    size_t bytes = 0;
};

// Backend must outlive the transfer. One staging buffer, slot and completion event.
class llama_weight_stream_transfer {
public:
    llama_weight_stream_transfer(ggml_backend_t backend, const llama_weight_stream_block_layout & block, const std::vector<llama_weight_stream_source> & sources, ggml_backend_buffer_t shared_slot = nullptr);
    ~llama_weight_stream_transfer();

    llama_weight_stream_transfer(const llama_weight_stream_transfer &) = delete;
    llama_weight_stream_transfer & operator=(const llama_weight_stream_transfer &) = delete;

    void stage();
    void persist();
    ggml_backend_buffer_t host_buffer() const;
    void verify() const;
    void upload();
    void synchronize();
    void bind_slot(ggml_backend_buffer_t buffer);
    ggml_backend_event_t completion_event() const;
    bool transfer_pending() const;

    bool pinned() const;
    const void * staging_data() const;
    size_t staging_capacity() const;
    size_t slot_capacity() const;
    // Tensor views are valid only while this transfer is alive.
    const ggml_tensor * tensor(size_t index) const;
    const llama_weight_stream_block_layout & block() const;

private:
    ggml_backend_t backend;
    llama_weight_stream_block_layout layout;
    std::vector<llama_weight_stream_source> sources;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr staging;
    ggml_backend_buffer_ptr slot;
    ggml_backend_buffer_t slot_buffer = nullptr;
    ggml_backend_event_ptr event;
    std::vector<ggml_tensor *> destinations;
    ggml_tensor * packed = nullptr;
    bool persistent = false;
    bool staged = false;
    bool pending = false;
};

struct llama_weight_stream_pager_plan {
    std::vector<llama_weight_stream_block_layout> blocks;
    size_t slot_count = 0;
    size_t slot_capacity = 0;
    size_t removed_weight_bytes = 0;
};

llama_weight_stream_pager_plan llama_weight_stream_make_pager_plan(const llama_weight_stream_model_profile & profile, const std::vector<int> & blocks, size_t slots, ggml_backend_buffer_type_t buft, size_t capacity = 0);
void llama_weight_stream_validate_pager_plan(const llama_weight_stream_pager_plan & plan);

struct llama_weight_stream_options {
    std::string mode = "off";
    uint64_t gpu_budget = 0;
    uint64_t reserve = 2048ULL * 1024 * 1024;
    uint64_t runtime_cuda_bytes = 0;
    uint64_t external_cuda_bytes = 0;
    std::string runtime_key;
    bool cache = true;
    std::string cache_dir;
};

std::string llama_weight_stream_default_cache_dir();
uint64_t llama_weight_stream_resolve_gpu_budget(uint64_t requested, uint64_t total_vram, uint64_t reserve);

struct llama_weight_stream_model_identity {
    std::string canonical_path;
    uint64_t file_size = 0;
    int64_t mtime_ns = 0;
};

struct llama_weight_stream_hardware {
    std::string identity;
    int compute_capability = 0;
    uint64_t total_vram = 0;
};

struct llama_weight_stream_auto_plan_result {
    uint32_t schema_version = 1;
    uint32_t planner_version = 2;
    std::string fingerprint;
    std::string hardware_key;
    llama_weight_stream_hardware hardware;
    std::string plan_key;
    llama_weight_stream_options options;
    llama_weight_stream_pager_plan pager;
    uint64_t full_resident_bytes = 0;
    uint64_t slot_bytes = 0;
    uint64_t extra_cuda_bytes = 0;
    uint64_t net_saving = 0;
    uint64_t expected_bytes = 0;
    size_t candidate_count = 0;
    std::vector<int> demand_order;
};

uint64_t llama_weight_stream_parse_size(const std::string & value, bool allow_zero = false);
llama_weight_stream_model_identity llama_weight_stream_identify(const std::string & path);
std::string llama_weight_stream_profile_json(const llama_weight_stream_model_profile & profile, const llama_weight_stream_model_identity & identity);
llama_weight_stream_model_profile llama_weight_stream_profile_from_json(const std::string & data, llama_weight_stream_model_identity & identity);
std::string llama_weight_stream_plan_json(const llama_weight_stream_auto_plan_result & plan);
llama_weight_stream_auto_plan_result llama_weight_stream_plan_from_json(const std::string & data);
std::string llama_weight_stream_profile_path(const llama_weight_stream_options & options, const llama_weight_stream_model_identity & identity);
std::string llama_weight_stream_plan_key(const llama_weight_stream_model_profile & profile, const llama_weight_stream_hardware & hardware, const llama_weight_stream_options & options);
llama_weight_stream_auto_plan_result llama_weight_stream_auto_plan(const llama_weight_stream_model_profile & profile, const llama_weight_stream_hardware & hardware, const llama_weight_stream_options & options, ggml_backend_buffer_type_t buft);
llama_weight_stream_model_profile llama_weight_stream_cached_profile(const llama_weight_stream_options & options, const llama_weight_stream_model_identity & identity, const std::function<llama_weight_stream_model_profile()> & profiler);
llama_weight_stream_auto_plan_result llama_weight_stream_cached_plan(const llama_weight_stream_options & options, const llama_weight_stream_model_profile & profile, const llama_weight_stream_hardware & hardware, const std::function<llama_weight_stream_auto_plan_result(const llama_weight_stream_model_profile &)> & planner);
void llama_weight_stream_validate_descriptor(const llama_weight_stream_tensor_profile & expected, const llama_weight_stream_tensor_profile & actual);
llama_model * llama_weight_stream_load_auto(const char * path, llama_model_params params, llama_context_params context_params, const llama_weight_stream_options & options, const llama_context_params * draft_params = nullptr, bool draft_backend_sampling = false);

enum class llama_weight_stream_slot_phase { EMPTY, LOADING, READY, IN_USE, REUSABLE };

struct llama_weight_stream_slot_state {
    size_t id = 0;
    size_t capacity = 0;
    int block = -1;
    uint64_t generation = 0;
    llama_weight_stream_slot_phase phase = llama_weight_stream_slot_phase::EMPTY;
    size_t loads = 0;
    size_t reuses = 0;
    size_t evictions = 0;
    void begin_load(int next_block, size_t bytes);
    void finish_load();
    void validate(int expected_block, uint64_t expected_generation) const;
    void begin_use(int expected_block, uint64_t expected_generation);
    void complete(int expected_block, uint64_t expected_generation, bool gpu_complete);
};

struct llama_weight_stream_ticket {
    size_t slot = 0;
    int block = -1;
    uint64_t generation = 0;
};

class llama_weight_stream_demand_queue {
public:
    explicit llama_weight_stream_demand_queue(std::vector<int> sequence);
    int next(const std::unordered_set<int> & excluded) const;
    void consume(int block);
    uint64_t position() const;
private:
    std::vector<int> sequence;
    uint64_t cursor = 0;
};

struct llama_weight_stream_pending_prefetch {
    bool active = false;
    llama_weight_stream_ticket ticket;
    uint64_t demand = 0;
    size_t bytes = 0;
    size_t transfer = 0;
};

struct llama_weight_stream_transfer_trace {
    size_t evaluation = 0;
    llama_weight_stream_ticket ticket;
    size_t bytes = 0;
    uint64_t demand = 0;
    int64_t launch_us = 0;
    int64_t ready_us = 0;
    int64_t needed_us = 0;
    int64_t wait_us = 0;
    double start_ms = 0;
    double end_ms = 0;
    bool consumed = false;
    bool cold = false;
};

struct llama_weight_stream_pager_stats {
    size_t evaluations = 0;
    size_t loads = 0;
    size_t reuses = 0;
    size_t evictions = 0;
    size_t hits = 0;
    size_t misses = 0;
    size_t upload_bytes = 0;
    size_t initial_host_copy_bytes = 0;
    size_t runtime_host_copy_bytes = 0;
    size_t verified_blocks = 0;
    size_t completion_waits = 0;
    size_t prefetched_hits = 0;
    size_t demand_misses = 0;
    int64_t exposed_wait_us = 0;
    int64_t upload_us = 0;
    int64_t completion_us = 0;
};

struct llama_weight_stream_slot_trace {
    size_t evaluation;
    size_t slot;
    int block;
    uint64_t generation;
    llama_weight_stream_slot_phase phase;
    std::string action;
    size_t bytes;
};

class llama_weight_stream_pager {
public:
    llama_weight_stream_pager(ggml_backend_t backend, const llama_weight_stream_pager_plan & plan, const std::vector<std::vector<llama_weight_stream_source>> & sources, bool prefetch = false);
    ~llama_weight_stream_pager();
    void begin_evaluation();
    llama_weight_stream_ticket enter(int block, ggml_backend_t compute_backend = nullptr);
    void complete(const llama_weight_stream_ticket & ticket, ggml_backend_t compute_backend);
    const ggml_tensor * checked_tensor(const llama_weight_stream_ticket & ticket, const std::string & name) const;
    const llama_weight_stream_transfer * block_store(int block) const;
    const std::vector<llama_weight_stream_slot_state> & slot_states() const;
    const llama_weight_stream_pager_stats & statistics() const;
    const std::vector<llama_weight_stream_slot_trace> & history() const;
    const llama_weight_stream_pager_plan & plan() const;
    bool prefetching() const;
    const std::vector<llama_weight_stream_pending_prefetch> & pending_prefetches() const;
    void validate_pending(const llama_weight_stream_ticket & ticket) const;
    void finish_timeline();
    const std::vector<llama_weight_stream_transfer_trace> & transfer_history() const;
private:
    ggml_backend_t backend;
    llama_weight_stream_pager_plan layout;
    bool asynchronous;
    llama_weight_stream_demand_queue demands;
    ggml_backend_event_t (*new_timed)(ggml_backend_dev_t) = nullptr;
    bool (*event_query)(ggml_backend_event_t) = nullptr;
    float (*elapsed_ms)(ggml_backend_event_t, ggml_backend_event_t) = nullptr;
    ggml_backend_event_ptr origin;
    std::vector<llama_weight_stream_pending_prefetch> pending;
    std::vector<llama_weight_stream_transfer_trace> transfers;
    std::vector<std::pair<ggml_backend_event_ptr, ggml_backend_event_ptr>> timing;
    std::unordered_map<int, size_t> bindings;
    void launch_prefetch(size_t slot, int block, bool cold = false);
    void finish_prefetch(size_t slot, ggml_backend_t compute_backend, bool consumer);
    size_t binding(int block) const;
    std::vector<ggml_backend_buffer_ptr> buffers;
    std::vector<ggml_backend_event_ptr> completion_events;
    std::vector<std::unique_ptr<llama_weight_stream_transfer>> stores;
    std::vector<llama_weight_stream_slot_state> states;
    std::unordered_set<int> verified;
    std::unordered_set<ggml_backend_t> compute_backends;
    llama_weight_stream_pager_stats stats;
    std::vector<llama_weight_stream_slot_trace> trace;
    size_t index(int block) const;
    llama_weight_stream_ticket ensure(int block, bool force);
    void record(size_t slot, const std::string & action, size_t bytes = 0);
};

// Internal experiment configuration. It is scoped to model construction on one thread.
struct llama_weight_stream_config {
    bool stream = false;
    bool prefetch = false;
    bool cached_descriptors = false;
    bool embedded_mtp = false;
    uint64_t mtp_weight_cuda_bytes = 0;
    uint64_t mtp_runtime_cuda_bytes = 0;
    llama_weight_stream_auto_plan_result auto_plan;
    int target_block = -1;
    std::vector<int> target_blocks;
    size_t slot_count = 0;
    llama_weight_stream_model_profile discovered;
};

class llama_weight_stream_config_scope {
public:
    explicit llama_weight_stream_config_scope(const llama_weight_stream_config & config);
    ~llama_weight_stream_config_scope();
    llama_weight_stream_config_scope(const llama_weight_stream_config_scope &) = delete;
    llama_weight_stream_config_scope & operator=(const llama_weight_stream_config_scope &) = delete;
private:
    const llama_weight_stream_config * previous;
};

const llama_weight_stream_config * llama_weight_stream_current_config();
llama_weight_stream_block_profile llama_weight_stream_plan(const llama_weight_stream_model_profile & profile, int block);
ggml_tensor * llama_weight_stream_resolve(bool enabled, bool ready, ggml_tensor * original, ggml_tensor * replacement);
void llama_weight_stream_require_cuda(ggml_backend_t backend);

struct llama_weight_stream_stats {
    size_t evaluations = 0;
    size_t uploads = 0;
    size_t upload_bytes = 0;
    size_t event_waits = 0;
    size_t initial_host_copy_bytes = 0;
    size_t runtime_host_copy_bytes = 0;
    int64_t upload_us = 0;
    int64_t wait_us = 0;
    size_t cuda_nodes = 0;
    size_t cpu_nodes = 0;
    size_t cuda_transformer_nodes = 0;
    size_t cpu_transformer_nodes = 0;
    bool byte_verified = false;
    std::unordered_map<int, size_t> per_block_cuda;
    std::unordered_map<int, size_t> per_block_cpu;
};

struct llama_weight_stream_node_trace {
    size_t evaluation;
    std::string name;
    std::string operation;
    std::string backend;
    int block = -1;
};

class llama_weight_stream_runtime {
public:
    explicit llama_weight_stream_runtime(const llama_weight_stream_config & config);
    void register_tensor(ggml_tensor * tensor, int block);
    void begin_graph();
    void observe_projection(ggml_tensor * tensor);
    void finish_graph(ggml_cgraph * graph);
    void tag_node(ggml_tensor * tensor, int block);
    llama_weight_stream_model_profile profile(const std::string & architecture) const;
    std::vector<std::string> selected_names() const;
    // Returns the temporary loader buffers which can now be released.
    std::vector<ggml_backend_buffer_t> initialize(ggml_backend_dev_t device, const std::string & architecture);
    void begin_evaluation(ggml_backend_sched_t sched);
    ggml_tensor * resolve(ggml_tensor * tensor);
    void check_backends(ggml_cgraph * graph, ggml_backend_sched_t sched);
    bool active() const;
    bool paging() const;
    const llama_weight_stream_pager * pager() const;
    static bool evaluation_callback(ggml_tensor * tensor, bool ask, void * runtime);
    int target_block() const;
    const llama_weight_stream_transfer * store() const;
    const llama_weight_stream_stats & statistics() const;
    const std::vector<llama_weight_stream_node_trace> & backend_trace() const;
    int64_t profile_time_us() const;
    bool embedded_mtp() const;
    void check_context_memory(uint64_t bytes, bool draft = false) const;
    void report_statistics(const char * label = "vn-wstream");
private:
    struct entry {
        ggml_tensor * tensor;
        llama_weight_stream_tensor_profile metadata;
        bool supported = false;
        bool unsafe = false;
    };
    llama_weight_stream_config config;
    std::vector<entry> entries;
    std::unordered_map<ggml_tensor *, size_t> registry;
    std::unordered_set<ggml_tensor *> graph_projections;
    std::vector<int> discovered_order;
    std::unordered_map<ggml_tensor *, int> node_blocks;
    std::unordered_map<ggml_tensor *, ggml_tensor *> replacements;
    ggml_backend_ptr backend;
    std::unique_ptr<llama_weight_stream_transfer> persistent;
    llama_weight_stream_stats stats;
    std::vector<llama_weight_stream_node_trace> trace;
    std::unique_ptr<llama_weight_stream_pager> paging_store;
    ggml_backend_sched_t evaluation_sched = nullptr;
    std::unordered_map<ggml_tensor *, int> consumers;
    std::unordered_map<int, ggml_tensor *> first_consumers;
    std::unordered_map<int, ggml_tensor *> last_consumers;
    std::unordered_set<ggml_tensor *> demand_boundaries;
    std::unordered_map<int, llama_weight_stream_ticket> tickets;
    bool ready = false;
    mutable int64_t profiling_us = 0;
    void upload();
    std::vector<ggml_backend_buffer_t> initialize_pager(ggml_backend_dev_t device);
};
