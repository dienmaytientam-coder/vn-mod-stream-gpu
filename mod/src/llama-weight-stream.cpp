#include "llama-weight-stream.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <climits>
#include <map>
#include <stdexcept>

static size_t weight_stream_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::invalid_argument("weight stream size overflow");
    }
    return a + b;
}

static size_t weight_stream_mul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::invalid_argument("weight stream size overflow");
    }
    return a * b;
}

static size_t weight_stream_align(size_t size, size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("invalid weight stream alignment");
    }
    return weight_stream_add(size, alignment - 1) & ~(alignment - 1);
}

static void weight_stream_validate_tensor(const llama_weight_stream_tensor_profile & tensor) {
    if (tensor.name.empty() || tensor.logical_bytes == 0 || tensor.type < 0 || tensor.type >= GGML_TYPE_COUNT) {
        throw std::invalid_argument("invalid weight stream tensor profile");
    }
    const int64_t block_size = ggml_blck_size(tensor.type);
    const size_t type_size = ggml_type_size(tensor.type);
    if (block_size <= 0 || type_size == 0 || tensor.ne[0] <= 0 || tensor.ne[0] % block_size != 0 || tensor.nb[0] != type_size) {
        throw std::invalid_argument("invalid weight stream tensor type or row");
    }
    size_t bytes = weight_stream_mul(size_t(tensor.ne[0] / block_size), type_size);
    for (int d = 1; d < GGML_MAX_DIMS; ++d) {
        if (tensor.ne[d] <= 0 || tensor.nb[d] != bytes) {
            throw std::invalid_argument("weight stream requires a contiguous layout");
        }
        bytes = weight_stream_mul(bytes, size_t(tensor.ne[d]));
    }
    if (bytes != tensor.logical_bytes) {
        throw std::invalid_argument("weight stream byte size does not match layout");
    }
    switch (tensor.capability) {
        case llama_weight_stream_capability::STREAMABLE:
        case llama_weight_stream_capability::MUST_RESIDENT:
        case llama_weight_stream_capability::UNKNOWN:
            if (tensor.stored_bytes != bytes) {
                throw std::invalid_argument("weight stream stored span does not match layout");
            }
            break;
        case llama_weight_stream_capability::STATE_NOT_WEIGHT:
            if (tensor.stored_bytes != 0) {
                throw std::invalid_argument("runtime state is not a stored weight");
            }
            break;
        default:
            throw std::invalid_argument("invalid weight stream capability");
    }
}

static ggml_tensor weight_stream_metadata(const llama_weight_stream_tensor_profile & tensor) {
    ggml_tensor result = {};
    result.type = tensor.type;
    std::copy(tensor.ne.begin(), tensor.ne.end(), result.ne);
    std::copy(tensor.nb.begin(), tensor.nb.end(), result.nb);
    return result;
}

llama_weight_stream_tensor_profile llama_weight_stream_describe(const ggml_tensor & tensor, int block_index, llama_weight_stream_capability capability) {
    llama_weight_stream_tensor_profile result;
    result.name = ggml_get_name(&tensor);
    result.block_index = block_index;
    result.type = tensor.type;
    std::copy(tensor.ne, tensor.ne + GGML_MAX_DIMS, result.ne.begin());
    std::copy(tensor.nb, tensor.nb + GGML_MAX_DIMS, result.nb.begin());
    result.logical_bytes = ggml_nbytes(&tensor);
    result.stored_bytes = capability == llama_weight_stream_capability::STATE_NOT_WEIGHT ? 0 : result.logical_bytes;
    result.capability = capability;
    weight_stream_validate_tensor(result);
    return result;
}

llama_weight_stream_block_profile llama_weight_stream_profile_block(int block_index, std::vector<llama_weight_stream_tensor_profile> tensors) {
    if (block_index < 0 || tensors.empty()) {
        throw std::invalid_argument("invalid or empty weight stream block profile");
    }
    llama_weight_stream_block_profile result;
    result.block_index = block_index;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & tensor = tensors[i];
        weight_stream_validate_tensor(tensor);
        if (tensor.block_index != block_index) {
            throw std::invalid_argument("weight stream block identity mismatch");
        }
        for (size_t j = 0; j < i; ++j) {
            if (tensor.name == tensors[j].name) {
                throw std::invalid_argument("duplicate weight stream tensor identity");
            }
        }
        switch (tensor.capability) {
            case llama_weight_stream_capability::STREAMABLE:
                result.streamable_bytes = weight_stream_add(result.streamable_bytes, tensor.stored_bytes);
                break;
            case llama_weight_stream_capability::MUST_RESIDENT:
                result.resident_bytes = weight_stream_add(result.resident_bytes, tensor.stored_bytes);
                break;
            case llama_weight_stream_capability::UNKNOWN:
                result.unknown_bytes = weight_stream_add(result.unknown_bytes, tensor.stored_bytes);
                break;
            case llama_weight_stream_capability::STATE_NOT_WEIGHT:
                break;
        }
    }
    result.tensors = std::move(tensors);
    return result;
}

void llama_weight_stream_validate_profile(const llama_weight_stream_model_profile & model) {
    if (model.schema_version == 0 || model.profile_version == 0 || model.architecture.empty() || (model.blocks.empty() && model.non_block_tensors.empty())) {
        throw std::invalid_argument("invalid or empty model weight profile");
    }
    std::vector<std::string> names;
    for (size_t i = 0; i < model.blocks.size(); ++i) {
        const auto & block = model.blocks[i];
        const auto checked = llama_weight_stream_profile_block(block.block_index, block.tensors);
        if (checked.streamable_bytes != block.streamable_bytes || checked.resident_bytes != block.resident_bytes || checked.unknown_bytes != block.unknown_bytes) {
            throw std::invalid_argument("inconsistent weight stream block totals");
        }
        for (size_t j = 0; j < i; ++j) {
            if (block.block_index == model.blocks[j].block_index) {
                throw std::invalid_argument("duplicate weight stream block identity");
            }
        }
        for (const auto & tensor : block.tensors) {
            names.push_back(tensor.name);
        }
    }
    for (const auto & tensor : model.non_block_tensors) {
        weight_stream_validate_tensor(tensor);
        if (tensor.block_index != -1) {
            throw std::invalid_argument("non-block tensor has a block identity");
        }
        names.push_back(tensor.name);
    }
    std::sort(names.begin(), names.end());
    if (std::adjacent_find(names.begin(), names.end()) != names.end()) {
        throw std::invalid_argument("duplicate model tensor identity");
    }
}

llama_weight_stream_block_layout llama_weight_stream_pack(const llama_weight_stream_block_profile & block, ggml_backend_buffer_type_t buft) {
    if (buft == nullptr) {
        throw std::invalid_argument("weight stream buffer type is missing");
    }
    const auto checked = llama_weight_stream_profile_block(block.block_index, block.tensors);
    llama_weight_stream_block_layout result;
    result.block_index = checked.block_index;
    result.alignment = ggml_backend_buft_get_alignment(buft);
    for (const auto & tensor : checked.tensors) {
        if (tensor.capability == llama_weight_stream_capability::STREAMABLE) {
            result.tensors.push_back({tensor, 0});
        }
    }
    std::sort(result.tensors.begin(), result.tensors.end(), [](const llama_weight_stream_tensor_layout & a, const llama_weight_stream_tensor_layout & b) { return a.profile.name < b.profile.name; });
    for (auto & tensor : result.tensors) {
        const ggml_tensor metadata = weight_stream_metadata(tensor.profile);
        tensor.profile.allocation_bytes = ggml_backend_buft_get_alloc_size(buft, &metadata);
        tensor.profile.alignment = result.alignment;
        tensor.offset = weight_stream_align(result.slot_bytes, result.alignment);
        result.slot_bytes = weight_stream_add(tensor.offset, tensor.profile.allocation_bytes);
    }
    result.slot_bytes = weight_stream_align(result.slot_bytes, result.alignment);
    llama_weight_stream_validate(result, ggml_backend_buft_get_max_size(buft));
    return result;
}

void llama_weight_stream_validate(const llama_weight_stream_block_layout & block, size_t capacity) {
    if (block.block_index < 0 || block.tensors.empty() || block.slot_bytes == 0 || block.slot_bytes > capacity || weight_stream_align(block.slot_bytes, block.alignment) != block.slot_bytes) {
        throw std::invalid_argument("invalid weight stream block layout or capacity");
    }
    for (size_t i = 0; i < block.tensors.size(); ++i) {
        const auto & tensor = block.tensors[i];
        weight_stream_validate_tensor(tensor.profile);
        if (tensor.profile.capability != llama_weight_stream_capability::STREAMABLE || tensor.profile.block_index != block.block_index || tensor.profile.alignment != block.alignment || tensor.profile.allocation_bytes < tensor.profile.logical_bytes || tensor.offset % block.alignment != 0 || weight_stream_add(tensor.offset, tensor.profile.allocation_bytes) > block.slot_bytes) {
            throw std::invalid_argument("invalid weight stream tensor extent or capability");
        }
        for (size_t j = 0; j < i; ++j) {
            const auto & other = block.tensors[j];
            if (tensor.profile.name == other.profile.name || (tensor.offset < other.offset + other.profile.allocation_bytes && other.offset < tensor.offset + tensor.profile.allocation_bytes)) {
                throw std::invalid_argument("duplicate or overlapping weight stream tensor");
            }
        }
    }
}

llama_weight_stream_transfer::llama_weight_stream_transfer(ggml_backend_t backend, const llama_weight_stream_block_layout & block, const std::vector<llama_weight_stream_source> & host_sources, ggml_backend_buffer_t shared_slot) : backend(backend), layout(block) {
    if (backend == nullptr) {
        throw std::invalid_argument("weight stream backend is missing");
    }
    const ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);
    if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0 || props.type != GGML_BACKEND_DEVICE_TYPE_GPU || !props.caps.async || !props.caps.host_buffer || !props.caps.events) {
        throw std::runtime_error("weight stream requires CUDA async, pinned host buffers and events");
    }
    const auto buft = ggml_backend_get_default_buffer_type(backend);
    llama_weight_stream_validate(layout, ggml_backend_buft_get_max_size(buft));
    if (layout.alignment != ggml_backend_buft_get_alignment(buft) || host_sources.size() != layout.tensors.size()) {
        throw std::invalid_argument("weight stream backend layout or source count mismatch");
    }
    for (const auto & tensor : layout.tensors) {
        const ggml_tensor metadata = weight_stream_metadata(tensor.profile);
        if (tensor.profile.allocation_bytes != ggml_backend_buft_get_alloc_size(buft, &metadata)) {
            throw std::invalid_argument("weight stream allocation does not match backend");
        }
        const llama_weight_stream_source * match = nullptr;
        for (const auto & source : host_sources) {
            if (source.name == tensor.profile.name) {
                if (match != nullptr) {
                    throw std::invalid_argument("duplicate weight stream source");
                }
                match = &source;
            }
        }
        if (match == nullptr || match->data == nullptr || match->bytes != tensor.profile.stored_bytes) {
            throw std::invalid_argument("missing or invalid weight stream source span");
        }
        sources.push_back(*match);
    }
    const auto host_buft = ggml_backend_dev_host_buffer_type(device);
    if (host_buft == nullptr) {
        throw std::runtime_error("weight stream pinned buffer type is missing");
    }
    staging.reset(ggml_backend_buft_alloc_buffer(host_buft, layout.slot_bytes));
    if (!staging || ggml_backend_buffer_get_type(staging.get()) != host_buft || !ggml_backend_buffer_is_host(staging.get())) {
        throw std::runtime_error("weight stream pinned staging allocation failed or fell back to pageable memory");
    }
    if (ggml_backend_buffer_get_base(staging.get()) == nullptr || staging_capacity() < layout.slot_bytes) {
        throw std::runtime_error("weight stream staging capacity is insufficient");
    }
    std::memset(ggml_backend_buffer_get_base(staging.get()), 0, layout.slot_bytes);
    if (shared_slot) {
        if (ggml_backend_buffer_get_type(shared_slot) != buft) { throw std::invalid_argument("shared slot buffer type differs from transfer backend"); }
        slot_buffer = shared_slot;
    } else {
        slot.reset(ggml_backend_buft_alloc_buffer(buft, layout.slot_bytes));
        slot_buffer = slot.get();
    }
    if (!slot_buffer || slot_capacity() < layout.slot_bytes) {
        throw std::runtime_error("weight stream CUDA slot allocation failed");
    }
    ggml_backend_buffer_set_usage(slot_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    context.reset(ggml_init({weight_stream_mul(ggml_tensor_overhead(), layout.tensors.size() + 1), nullptr, true}));
    if (!context) {
        throw std::runtime_error("weight stream tensor context allocation failed");
    }
    destinations.reserve(layout.tensors.size());
    for (const auto & descriptor : layout.tensors) {
        ggml_tensor * tensor = ggml_new_tensor(context.get(), descriptor.profile.type, GGML_MAX_DIMS, descriptor.profile.ne.data());
        ggml_set_name(tensor, descriptor.profile.name.c_str());
        void * address = static_cast<char *>(ggml_backend_buffer_get_base(slot_buffer)) + descriptor.offset;
        if (ggml_backend_tensor_alloc(slot_buffer, tensor, address) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("weight stream CUDA tensor allocation failed");
        }
        destinations.push_back(tensor);
    }
    packed = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, layout.slot_bytes);
    if (ggml_backend_tensor_alloc(slot_buffer, packed, ggml_backend_buffer_get_base(slot_buffer)) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("weight stream packed view allocation failed");
    }
    event.reset(ggml_backend_event_new(device));
    if (!event) {
        throw std::runtime_error("weight stream backend event allocation failed");
    }
}

llama_weight_stream_transfer::~llama_weight_stream_transfer() {
    // Keep queued transfers complete before buffers and descriptors are freed.
    ggml_backend_synchronize(backend);
}

void llama_weight_stream_transfer::stage() {
    if (pending || persistent) {
        throw std::logic_error("weight stream store is pending or immutable");
    }
    auto * data = static_cast<char *>(ggml_backend_buffer_get_base(staging.get()));
    for (size_t i = 0; i < layout.tensors.size(); ++i) {
        std::memcpy(data + layout.tensors[i].offset, sources[i].data, sources[i].bytes);
    }
    staged = true;
}

void llama_weight_stream_transfer::upload() {
    if (pending || !staged) {
        throw std::logic_error("weight stream upload requires an idle staged snapshot");
    }
    pending = true;
    staged = persistent;
    const auto * data = static_cast<const char *>(ggml_backend_buffer_get_base(staging.get()));
    ggml_backend_tensor_set_async(backend, packed, data, 0, layout.slot_bytes);
    ggml_backend_event_record(event.get(), backend);
}

void llama_weight_stream_transfer::synchronize() {
    if (pending) {
        ggml_backend_event_synchronize(event.get());
        pending = false;
    }
}

void llama_weight_stream_transfer::bind_slot(ggml_backend_buffer_t buffer) {
    if (!buffer || ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(slot_buffer) || ggml_backend_buffer_get_size(buffer) < layout.slot_bytes) {
        throw std::invalid_argument("invalid rebound transfer slot");
    }
    if (pending && buffer != slot_buffer) { throw std::invalid_argument("cannot move a pending transfer"); }
    slot_buffer = buffer;
    for (size_t i = 0; i < destinations.size(); ++i) {
        destinations[i]->buffer = buffer;
        destinations[i]->data = static_cast<char *>(ggml_backend_buffer_get_base(buffer)) + layout.tensors[i].offset;
    }
    packed->buffer = buffer;
    packed->data = ggml_backend_buffer_get_base(buffer);
}
ggml_backend_event_t llama_weight_stream_transfer::completion_event() const { return event.get(); }
bool llama_weight_stream_transfer::transfer_pending() const { return pending; }

bool llama_weight_stream_transfer::pinned() const {
    return ggml_backend_buffer_get_type(staging.get()) == ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
}

const void * llama_weight_stream_transfer::staging_data() const { return ggml_backend_buffer_get_base(staging.get()); }
size_t llama_weight_stream_transfer::staging_capacity() const { return ggml_backend_buffer_get_size(staging.get()); }
size_t llama_weight_stream_transfer::slot_capacity() const { return ggml_backend_buffer_get_size(slot_buffer); }
const ggml_tensor * llama_weight_stream_transfer::tensor(size_t index) const { return destinations.at(index); }
const llama_weight_stream_block_layout & llama_weight_stream_transfer::block() const { return layout; }

void llama_weight_stream_transfer::persist() {
    stage();
    persistent = true;
    sources.clear();
}

ggml_backend_buffer_t llama_weight_stream_transfer::host_buffer() const { return staging.get(); }

void llama_weight_stream_transfer::verify() const {
    if (pending) { throw std::logic_error("verification requires completed upload"); }
    std::vector<unsigned char> copy(layout.slot_bytes);
    ggml_backend_tensor_get(packed, copy.data(), 0, copy.size());
    if (std::memcmp(copy.data(), staging_data(), copy.size()) != 0) {
        throw std::runtime_error("weight stream byte verification failed");
    }
}

static thread_local const llama_weight_stream_config * weight_stream_config = nullptr;

llama_weight_stream_config_scope::llama_weight_stream_config_scope(const llama_weight_stream_config & config) : previous(weight_stream_config) {
    weight_stream_config = &config;
}
llama_weight_stream_config_scope::~llama_weight_stream_config_scope() { weight_stream_config = previous; }
const llama_weight_stream_config * llama_weight_stream_current_config() { return weight_stream_config; }

llama_weight_stream_block_profile llama_weight_stream_plan(const llama_weight_stream_model_profile & profile, int block) {
    llama_weight_stream_validate_profile(profile);
    for (const auto & candidate : profile.blocks) {
        if (candidate.block_index == block && candidate.streamable_bytes > 0) { return candidate; }
    }
    throw std::invalid_argument("requested block has no proven streamable weights");
}

ggml_tensor * llama_weight_stream_resolve(bool enabled, bool ready, ggml_tensor * original, ggml_tensor * replacement) {
    if (!enabled || replacement == nullptr) { return original; }
    if (!ready) { throw std::invalid_argument("weight stream slot is not ready"); }
    return replacement;
}

void llama_weight_stream_require_cuda(ggml_backend_t backend) {
    if (backend == nullptr || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend))), "CUDA") != 0 || ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw std::invalid_argument("streamed transformer compute must use CUDA");
    }
}

llama_weight_stream_runtime::llama_weight_stream_runtime(const llama_weight_stream_config & config) : config(config) {
    if (config.stream) {
        if (config.target_blocks.empty()) { llama_weight_stream_plan(config.discovered, config.target_block); }
        else {
            if (config.slot_count != 2 || config.target_blocks.size() <= config.slot_count) { throw std::invalid_argument("pager experiment requires at least three blocks and exactly two slots"); }
            std::unordered_set<int> blocks;
            for (int block : config.target_blocks) {
                if (!blocks.insert(block).second) { throw std::invalid_argument("duplicate selected runtime block"); }
                llama_weight_stream_plan(config.discovered, block);
            }
        }
    }
}

void llama_weight_stream_runtime::register_tensor(ggml_tensor * tensor, int block) {
    if (tensor == nullptr || registry.count(tensor)) { return; }
    if (config.cached_descriptors) {
        for (const auto & layout : config.auto_plan.pager.blocks) {
            for (const auto & descriptor : layout.tensors) {
                if (descriptor.profile.name != ggml_get_name(tensor)) { continue; }
                llama_weight_stream_validate_descriptor(descriptor.profile, llama_weight_stream_describe(*tensor, block, llama_weight_stream_capability::STREAMABLE));
                registry.emplace(tensor, entries.size());
                entries.push_back({tensor, descriptor.profile});
                return;
            }
        }
        return;
    }
    const auto start = ggml_time_us();
    registry.emplace(tensor, entries.size());
    entries.push_back({tensor, llama_weight_stream_describe(*tensor, block, llama_weight_stream_capability::UNKNOWN)});
    profiling_us += ggml_time_us() - start;
}

void llama_weight_stream_runtime::begin_graph() {
    graph_projections.clear();
    node_blocks.clear();
    consumers.clear();
    first_consumers.clear();
    last_consumers.clear();
    demand_boundaries.clear();
}

void llama_weight_stream_runtime::observe_projection(ggml_tensor * tensor) {
    if (registry.count(tensor)) { graph_projections.insert(tensor); }
}

void llama_weight_stream_runtime::finish_graph(ggml_cgraph * graph) {
    const auto start = ggml_time_us();
    std::vector<int> observed_order;
    std::unordered_set<int> seen_blocks;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * source = node->src[j];
            for (const auto & remap : replacements) {
                if (source == remap.second) { source = remap.first; break; }
            }
            auto found = registry.find(source);
            if (found == registry.end()) { continue; }
            auto & weight = entries[found->second];
            const bool supported = j == 0 && node->op == GGML_OP_MUL_MAT && graph_projections.count(source);
            if (!config.cached_descriptors) {
                weight.supported |= supported; weight.unsafe |= !supported;
                const int block = weight.metadata.block_index;
                if (supported && block >= 0 && seen_blocks.insert(block).second) { observed_order.push_back(block); }
            }
            if (paging() && supported && replacements.count(source)) {
                const int block = weight.metadata.block_index;
                consumers.emplace(node, block);
                first_consumers.emplace(block, node);
                last_consumers[block] = node;
            }
            if (config.stream && replacements.count(source) && !supported) {
                throw std::runtime_error("selected weight has an unsupported graph consumer");
            }
        }
    }
    if (!config.cached_descriptors && !observed_order.empty()) { discovered_order = observed_order; }
    if (config.stream) {
        for (const auto & remap : replacements) {
            if (!graph_projections.count(remap.first)) { throw std::runtime_error("selected weight bypassed common remap hook"); }
        }
    }
    if (paging()) {
        std::vector<int> order;
        std::unordered_map<int, int> first_positions, last_positions;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            auto consumer = consumers.find(node);
            if (consumer == consumers.end()) { continue; }
            if (first_consumers.at(consumer->second) == node) { order.push_back(consumer->second); first_positions[consumer->second] = i; }
            last_positions[consumer->second] = i;
        }
        if (order != config.target_blocks) { throw std::runtime_error("selected graph consumer order differs from pager plan"); }
        for (size_t i = 1; i < order.size(); ++i) {
            if (last_positions.at(order[i-1]) >= first_positions.at(order[i])) { throw std::runtime_error("pager does not support interleaved block consumers"); }
        }
        if (config.prefetch) {
            for (int block : order) {
                const int position = first_positions.at(block);
                if (position > 0) { demand_boundaries.insert(ggml_graph_node(graph, position - 1)); }
            }
        }
    }
    profiling_us += ggml_time_us() - start;
}

void llama_weight_stream_runtime::tag_node(ggml_tensor * tensor, int block) {
    if (tensor == nullptr || tensor->op == GGML_OP_NONE || node_blocks.count(tensor)) { return; }
    node_blocks.emplace(tensor, block);
    for (auto * source : tensor->src) { tag_node(source, block); }
}

llama_weight_stream_model_profile llama_weight_stream_runtime::profile(const std::string & architecture) const {
    if (config.cached_descriptors) { return config.discovered; }
    const auto start = ggml_time_us();
    llama_weight_stream_model_profile result;
    result.architecture = architecture;
    std::map<int, std::vector<llama_weight_stream_tensor_profile>> blocks;
    std::vector<llama_weight_stream_tensor_profile> ordered;
    for (const auto & entry : entries) {
        auto metadata = entry.metadata;
        metadata.capability = entry.unsafe ? llama_weight_stream_capability::MUST_RESIDENT : entry.supported ? llama_weight_stream_capability::STREAMABLE : llama_weight_stream_capability::UNKNOWN;
        if (entry.tensor->buffer) {
            auto buft = ggml_backend_buffer_get_type(entry.tensor->buffer);
            metadata.allocation_bytes = ggml_backend_buft_get_alloc_size(buft, entry.tensor);
            metadata.alignment = ggml_backend_buft_get_alignment(buft);
            metadata.cuda_weight = !ggml_backend_buffer_is_host(entry.tensor->buffer) && std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_buft_get_device(buft)))) == "CUDA";
        }
        ordered.push_back(metadata);
        if (metadata.block_index < 0) { result.non_block_tensors.push_back(metadata); }
        else { blocks[metadata.block_index].push_back(metadata); }
    }
    for (auto & block : blocks) { result.blocks.push_back(llama_weight_stream_profile_block(block.first, std::move(block.second))); }
    result.demand_order = discovered_order;
    std::sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) { return a.name < b.name; });
    uint64_t hash = 14695981039346656037ULL;
    auto hash_text = [&](const std::string & text) { for (unsigned char byte : text) { hash = (hash ^ byte) * 1099511628211ULL; } };
    hash_text(architecture);
    for (const auto & t : ordered) {
        hash_text(t.name + ":" + std::to_string(t.block_index) + ":" + std::to_string(t.type));
        for (int d = 0; d < GGML_MAX_DIMS; ++d) { hash_text(":" + std::to_string(t.ne[d]) + ":" + std::to_string(t.nb[d])); }
    }
    result.model_fingerprint = "descriptor-fnv1a64:" + std::to_string(hash);
    llama_weight_stream_validate_profile(result);
    profiling_us += ggml_time_us() - start;
    return result;
}

std::vector<std::string> llama_weight_stream_runtime::selected_names() const {
    std::vector<std::string> result;
    if (!config.stream) { return result; }
    const auto blocks = config.target_blocks.empty() ? std::vector<int>{config.target_block} : config.target_blocks;
    for (int block : blocks) {
        for (const auto & tensor : llama_weight_stream_plan(config.discovered, block).tensors) {
            if (tensor.capability == llama_weight_stream_capability::STREAMABLE) { result.push_back(tensor.name); }
        }
    }
    return result;
}

std::vector<ggml_backend_buffer_t> llama_weight_stream_runtime::initialize(ggml_backend_dev_t device, const std::string & architecture) {
    if (!config.stream) { return {}; }
    if (!config.cached_descriptors && profile(architecture).model_fingerprint != config.discovered.model_fingerprint) { throw std::runtime_error("discovery metadata fingerprint differs on reload"); }
    backend.reset(ggml_backend_dev_init(device, nullptr));
    llama_weight_stream_require_cuda(backend.get());
    if (!config.target_blocks.empty()) { return initialize_pager(device); }
    const auto layout = llama_weight_stream_pack(llama_weight_stream_plan(config.discovered, config.target_block), ggml_backend_get_default_buffer_type(backend.get()));
    std::vector<llama_weight_stream_source> sources;
    std::vector<ggml_tensor *> originals;
    std::vector<ggml_backend_buffer_t> temporary;
    for (const auto & descriptor : layout.tensors) {
        ggml_tensor * original = nullptr;
        for (const auto & entry : entries) { if (entry.metadata.name == descriptor.profile.name) { original = entry.tensor; } }
        if (!original || !original->buffer || ggml_backend_buffer_get_type(original->buffer) != ggml_backend_dev_host_buffer_type(device)) {
            throw std::runtime_error("selected original is not exclusively pinned host resident");
        }
        if (!ggml_backend_buffer_is_host(original->buffer)) { throw std::runtime_error("duplicate CUDA original detected"); }
        auto actual = llama_weight_stream_describe(*original, config.target_block, llama_weight_stream_capability::STREAMABLE);
        if (actual.type != descriptor.profile.type || actual.ne != descriptor.profile.ne || actual.nb != descriptor.profile.nb || actual.stored_bytes != descriptor.profile.stored_bytes) {
            throw std::runtime_error("selected weight metadata differs on reload");
        }
        originals.push_back(original);
        sources.push_back({actual.name, original->data, actual.stored_bytes});
        if (std::find(temporary.begin(), temporary.end(), original->buffer) == temporary.end()) { temporary.push_back(original->buffer); }
    }
    for (const auto & entry : entries) {
        if (std::find(temporary.begin(), temporary.end(), entry.tensor->buffer) != temporary.end() && std::find(originals.begin(), originals.end(), entry.tensor) == originals.end()) {
            throw std::runtime_error("selected loader host buffer contains unselected weights");
        }
    }
    persistent = std::make_unique<llama_weight_stream_transfer>(backend.get(), layout, sources);
    persistent->persist();
    for (size_t i = 0; i < originals.size(); ++i) {
        stats.initial_host_copy_bytes += layout.tensors[i].profile.stored_bytes;
        originals[i]->buffer = persistent->host_buffer();
        originals[i]->data = static_cast<char *>(const_cast<void *>(persistent->staging_data())) + layout.tensors[i].offset;
        replacements.emplace(originals[i], const_cast<ggml_tensor *>(persistent->tensor(i)));
    }
    upload();
    persistent->verify();
    stats.byte_verified = true;
    return temporary;
}

void llama_weight_stream_runtime::upload() {
    ready = false;
    const auto start = ggml_time_us();
    persistent->upload();
    const auto wait_start = ggml_time_us();
    persistent->synchronize();
    const auto end = ggml_time_us();
    stats.upload_us += end - start;
    stats.wait_us += end - wait_start;
    stats.uploads++;
    stats.event_waits++;
    stats.upload_bytes += persistent->block().slot_bytes;
    ready = true;
}

void llama_weight_stream_runtime::begin_evaluation(ggml_backend_sched_t sched) {
    if (active()) {
        if (paging() && evaluation_sched && evaluation_sched != sched) { throw std::invalid_argument("pager experiment supports one inference context per model"); }
        ggml_backend_sched_synchronize(sched);
        evaluation_sched = sched;
        tickets.clear();
        if (paging()) { paging_store->begin_evaluation(); } else { upload(); }
        stats.evaluations++;
    } else if (config.cached_descriptors) {
        stats.evaluations++;
    }
}

ggml_tensor * llama_weight_stream_runtime::resolve(ggml_tensor * tensor) {
    auto found = replacements.find(tensor);
    // Pager descriptors are checked against residency at execution, not graph creation.
    if (paging() && found != replacements.end()) { return found->second; }
    return llama_weight_stream_resolve(active(), ready, tensor, found == replacements.end() ? nullptr : found->second);
}

void llama_weight_stream_runtime::check_backends(ggml_cgraph * graph, ggml_backend_sched_t sched) {
    if (!active() && !config.cached_descriptors) { return; }
    if (paging()) {
        size_t weight_consumers = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_MUL_MAT) { continue; }
            for (const auto & replacement : replacements) {
                if (node->src[0] == replacement.second) {
                    if (!consumers.count(node)) { throw std::runtime_error("pager callback map belongs to a different graph"); }
                    weight_consumers++;
                }
            }
        }
        if (weight_consumers != consumers.size()) { throw std::runtime_error("pager consumer coverage mismatch"); }
    }
    std::unordered_set<ggml_backend_t> validated_backends;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        auto * node = ggml_graph_node(graph, i);
        auto block = node_blocks.find(node);
        if (block == node_blocks.end() || block->second < 0) { continue; }
        switch (node->op) {
            case GGML_OP_NONE: case GGML_OP_VIEW: case GGML_OP_RESHAPE: case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE: continue;
            default: break;
        }
        auto * assigned = ggml_backend_sched_get_tensor_backend(sched, node);
        const bool selected = paging() ? std::find(config.target_blocks.begin(), config.target_blocks.end(), block->second) != config.target_blocks.end() : block->second == config.target_block;
        if (selected) { trace.push_back({stats.evaluations, ggml_get_name(node), ggml_op_name(node->op), assigned ? ggml_backend_name(assigned) : "UNASSIGNED", block->second}); }
        try {
            if (validated_backends.insert(assigned).second) { llama_weight_stream_require_cuda(assigned); }
            stats.cuda_transformer_nodes++;
            if (selected) { stats.cuda_nodes++; stats.per_block_cuda[block->second]++; }
        } catch (const std::invalid_argument &) {
            stats.cpu_transformer_nodes++;
            if (selected) { stats.cpu_nodes++; stats.per_block_cpu[block->second]++; }
            throw std::runtime_error("non-CUDA transformer node: block=" + std::to_string(block->second) + " name=" + ggml_get_name(node) + " op=" + ggml_op_name(node->op) + " backend=" + (assigned ? ggml_backend_name(assigned) : "UNASSIGNED"));
        }
    }
}

int llama_weight_stream_runtime::target_block() const { return config.target_blocks.empty() ? config.target_block : config.target_blocks.front(); }

bool llama_weight_stream_runtime::active() const { return config.stream && (persistent != nullptr || paging_store != nullptr); }
bool llama_weight_stream_runtime::paging() const { return paging_store != nullptr; }
const llama_weight_stream_pager * llama_weight_stream_runtime::pager() const { return paging_store.get(); }
const llama_weight_stream_transfer * llama_weight_stream_runtime::store() const { return persistent.get(); }
const llama_weight_stream_stats & llama_weight_stream_runtime::statistics() const { return stats; }
const std::vector<llama_weight_stream_node_trace> & llama_weight_stream_runtime::backend_trace() const { return trace; }
int64_t llama_weight_stream_runtime::profile_time_us() const { return profiling_us; }
bool llama_weight_stream_runtime::embedded_mtp() const { return config.embedded_mtp; }
void llama_weight_stream_runtime::check_context_memory(uint64_t bytes, bool draft) const {
    if (!config.cached_descriptors || config.auto_plan.plan_key.empty()) { return; }
    const auto expected = draft ? config.mtp_runtime_cuda_bytes : config.auto_plan.options.runtime_cuda_bytes - config.mtp_runtime_cuda_bytes;
    fprintf(stderr, "vn-wstream: actual %scontext/compute CUDA=%llu estimated=%llu\n", draft ? "MTP " : "", (unsigned long long) bytes, (unsigned long long) expected);
    if (bytes > expected) { throw std::runtime_error("runtime CUDA allocation exceeds auto estimate"); }
}
void llama_weight_stream_runtime::report_statistics(const char * label) {
    if (!config.cached_descriptors || config.auto_plan.plan_key.empty()) { return; }
    fprintf(stderr, "%s: evaluations=%zu CPU_transformer=%zu CUDA_transformer=%zu\n", label, stats.evaluations, stats.cpu_transformer_nodes, stats.cuda_transformer_nodes);
    if (!paging_store) { return; }
    try {
        paging_store->finish_timeline();
        const auto & telemetry = paging_store->statistics();
        double h2d_ms = 0;
        for (const auto & row : paging_store->transfer_history()) {
            if (row.consumed) { h2d_ms += row.end_ms - row.start_ms; }
            if (getenv("VN_WSTREAM_TRACE")) {
                fprintf(stderr, "vn-wstream-transfer: eval=%zu demand=%llu bytes=%zu consumed=%d launch_us=%lld needed_us=%lld wait_us=%lld start_ms=%.6f end_ms=%.6f\n",
                    row.evaluation, (unsigned long long) row.demand, row.bytes, row.consumed, (long long) row.launch_us, (long long) row.needed_us, (long long) row.wait_us, row.start_ms, row.end_ms);
            }
        }
        fprintf(stderr, "vn-wstream: loads=%zu verified=%zu slots=%zu pinned=%zu host_memcpy=%zu H2D_bytes=%zu H2D_ms=%.3f exposed_wait_ms=%.3f demand_misses=%zu\n",
            telemetry.loads, telemetry.verified_blocks, paging_store->plan().slot_count, telemetry.initial_host_copy_bytes, telemetry.runtime_host_copy_bytes, telemetry.upload_bytes, h2d_ms, telemetry.exposed_wait_us / 1000.0, telemetry.demand_misses);
    } catch (const std::exception & e) {
        fprintf(stderr, "vn-wstream: final telemetry failed: %s\n", e.what());
    }
}

llama_weight_stream_pager_plan llama_weight_stream_make_pager_plan(const llama_weight_stream_model_profile & profile, const std::vector<int> & blocks, size_t slots, ggml_backend_buffer_type_t buft, size_t capacity) {
    if (slots == 0 || blocks.empty() || slots > blocks.size()) { throw std::invalid_argument("invalid pager block/slot count"); }
    llama_weight_stream_pager_plan plan;
    plan.slot_count = slots;
    std::unordered_set<int> identities;
    for (int block : blocks) {
        if (!identities.insert(block).second) { throw std::invalid_argument("duplicate pager block"); }
        auto layout = llama_weight_stream_pack(llama_weight_stream_plan(profile, block), buft);
        plan.slot_capacity = std::max(plan.slot_capacity, layout.slot_bytes);
        for (const auto & tensor : layout.tensors) { plan.removed_weight_bytes = weight_stream_add(plan.removed_weight_bytes, tensor.profile.allocation_bytes); }
        plan.blocks.push_back(std::move(layout));
    }
    if (capacity) { plan.slot_capacity = capacity; }
    llama_weight_stream_validate_pager_plan(plan);
    return plan;
}

void llama_weight_stream_validate_pager_plan(const llama_weight_stream_pager_plan & plan) {
    if (plan.slot_count == 0 || plan.blocks.empty() || plan.slot_count > plan.blocks.size() || plan.slot_capacity == 0) {
        throw std::invalid_argument("invalid pager capacity/count");
    }
    weight_stream_mul(plan.slot_count, plan.slot_capacity);
    size_t removed = 0;
    std::unordered_set<int> blocks;
    for (const auto & block : plan.blocks) {
        llama_weight_stream_validate(block, plan.slot_capacity);
        if (!blocks.insert(block.block_index).second || plan.slot_capacity % block.alignment != 0) { throw std::invalid_argument("duplicate block or unaligned slot"); }
        for (const auto & tensor : block.tensors) { removed = weight_stream_add(removed, tensor.profile.allocation_bytes); }
    }
    if (removed != plan.removed_weight_bytes) { throw std::invalid_argument("pager allocation total mismatch"); }
}

void llama_weight_stream_slot_state::begin_load(int next_block, size_t bytes) {
    if ((phase != llama_weight_stream_slot_phase::EMPTY && phase != llama_weight_stream_slot_phase::REUSABLE) || next_block < 0 || bytes == 0 || bytes > capacity || generation == UINT64_MAX) {
        throw std::invalid_argument("illegal slot overwrite, capacity or generation");
    }
    if (block >= 0) { reuses++; if (block != next_block) { evictions++; } }
    block = next_block;
    generation++;
    loads++;
    phase = llama_weight_stream_slot_phase::LOADING;
}
void llama_weight_stream_slot_state::finish_load() {
    if (phase != llama_weight_stream_slot_phase::LOADING) { throw std::invalid_argument("slot was not loading"); }
    phase = llama_weight_stream_slot_phase::READY;
}
void llama_weight_stream_slot_state::validate(int expected_block, uint64_t expected_generation) const {
    if ((phase != llama_weight_stream_slot_phase::READY && phase != llama_weight_stream_slot_phase::IN_USE) || block != expected_block || generation != expected_generation) {
        throw std::invalid_argument("stale slot block, generation or readiness");
    }
}
void llama_weight_stream_slot_state::begin_use(int expected_block, uint64_t expected_generation) {
    validate(expected_block, expected_generation);
    if (phase != llama_weight_stream_slot_phase::READY) { throw std::invalid_argument("slot already in use"); }
    phase = llama_weight_stream_slot_phase::IN_USE;
}
void llama_weight_stream_slot_state::complete(int expected_block, uint64_t expected_generation, bool gpu_complete) {
    validate(expected_block, expected_generation);
    if (phase != llama_weight_stream_slot_phase::IN_USE || !gpu_complete) { throw std::invalid_argument("slot consumer has not completed"); }
    phase = llama_weight_stream_slot_phase::REUSABLE;
}

llama_weight_stream_demand_queue::llama_weight_stream_demand_queue(std::vector<int> blocks) : sequence(std::move(blocks)) {
    std::unordered_set<int> unique;
    for (int block : sequence) {
        if (block < 0 || !unique.insert(block).second) { throw std::invalid_argument("invalid future demand sequence"); }
    }
    if (sequence.empty()) { throw std::invalid_argument("empty future demand sequence"); }
}
int llama_weight_stream_demand_queue::next(const std::unordered_set<int> & excluded) const {
    for (size_t i = 0; i < sequence.size(); ++i) {
        int block = sequence[(cursor + i) % sequence.size()];
        if (!excluded.count(block)) { return block; }
    }
    return -1;
}
void llama_weight_stream_demand_queue::consume(int block) {
    if (block != sequence[cursor % sequence.size()] || cursor == UINT64_MAX) { throw std::invalid_argument("out of order future demand"); }
    cursor++;
}
uint64_t llama_weight_stream_demand_queue::position() const { return cursor; }
static std::vector<int> weight_stream_sequence(const llama_weight_stream_pager_plan & plan) {
    std::vector<int> blocks;
    for (const auto & block : plan.blocks) { blocks.push_back(block.block_index); }
    return blocks;
}

llama_weight_stream_pager::llama_weight_stream_pager(ggml_backend_t backend, const llama_weight_stream_pager_plan & plan, const std::vector<std::vector<llama_weight_stream_source>> & sources, bool prefetch) : backend(backend), layout(plan), asynchronous(prefetch), demands(weight_stream_sequence(plan)) {
    llama_weight_stream_require_cuda(backend);
    llama_weight_stream_validate_pager_plan(layout);
    if (sources.size() != layout.blocks.size()) { throw std::invalid_argument("pager source block count mismatch"); }
    for (size_t i = 0; i < layout.slot_count; ++i) {
        ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), layout.slot_capacity));
        if (!buffer || ggml_backend_buffer_get_size(buffer.get()) != layout.slot_capacity) { throw std::runtime_error("pager slot allocation differs from plan capacity"); }
        buffers.push_back(std::move(buffer));
        ggml_backend_event_ptr event(ggml_backend_event_new(ggml_backend_get_device(backend)));
        if (!event) { throw std::runtime_error("pager completion event allocation failed"); }
        completion_events.push_back(std::move(event));
        llama_weight_stream_slot_state state;
        state.id = i; state.capacity = layout.slot_capacity;
        states.push_back(state);
        record(i, "EMPTY");
    }
    for (size_t i = 0; i < layout.blocks.size(); ++i) {
        auto store = std::make_unique<llama_weight_stream_transfer>(backend, layout.blocks[i], sources[i], buffers[i % layout.slot_count].get());
        store->persist();
        for (const auto & tensor : layout.blocks[i].tensors) { stats.initial_host_copy_bytes += tensor.profile.stored_bytes; }
        stores.push_back(std::move(store));
    }
    if (asynchronous) {
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        new_timed = (decltype(new_timed)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_new_timed");
        event_query = (decltype(event_query)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_query");
        elapsed_ms = (decltype(elapsed_ms)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_elapsed_ms");
        if (!new_timed || !event_query || !elapsed_ms) { throw std::runtime_error("CUDA prefetch event procedures unavailable"); }
        origin.reset(new_timed(ggml_backend_get_device(backend)));
        if (!origin) { throw std::runtime_error("prefetch origin event allocation failed"); }
        ggml_backend_event_record(origin.get(), backend);
        ggml_backend_event_synchronize(origin.get());
        pending.resize(layout.slot_count);
        for (size_t i = 0; i < layout.blocks.size(); ++i) { bindings[layout.blocks[i].block_index] = i % layout.slot_count; }
        for (size_t i = 0; i < layout.slot_count; ++i) {
            launch_prefetch(i, layout.blocks[i].block_index, true);
            finish_prefetch(i, nullptr, false);
        }
    } else {
        for (size_t i = 0; i < layout.slot_count; ++i) { ensure(layout.blocks[i].block_index, true); }
    }
}

size_t llama_weight_stream_pager::index(int block) const {
    for (size_t i = 0; i < layout.blocks.size(); ++i) { if (layout.blocks[i].block_index == block) { return i; } }
    throw std::invalid_argument("block not in pager plan");
}
void llama_weight_stream_pager::record(size_t slot, const std::string & action, size_t bytes) {
    const auto & state = states.at(slot);
    trace.push_back({stats.evaluations, slot, state.block, state.generation, state.phase, action, bytes});
}
llama_weight_stream_ticket llama_weight_stream_pager::ensure(int block, bool force) {
    const size_t block_index = index(block);
    const size_t slot = block_index % layout.slot_count;
    auto & state = states.at(slot);
    if (!force && state.block == block && state.phase == llama_weight_stream_slot_phase::READY) {
        stats.hits++;
        record(slot, "HIT");
        return {slot, block, state.generation};
    }
    stats.misses++;
    const auto old_reuses = state.reuses, old_evictions = state.evictions;
    if (state.phase == llama_weight_stream_slot_phase::REUSABLE && state.block != block) { record(slot, "EVICT"); }
    state.begin_load(block, layout.blocks[block_index].slot_bytes);
    stats.reuses += state.reuses - old_reuses;
    stats.evictions += state.evictions - old_evictions;
    record(slot, "LOADING", layout.blocks[block_index].slot_bytes);
    const auto start = ggml_time_us();
    stores[block_index]->upload();
    stores[block_index]->synchronize();
    stats.upload_us += ggml_time_us() - start;
    stats.loads++;
    stats.upload_bytes += layout.blocks[block_index].slot_bytes;
    state.finish_load();
    record(slot, "READY");
    if (verified.insert(block).second) {
        stores[block_index]->verify();
        stats.verified_blocks++;
        record(slot, "BYTE_VERIFIED", layout.blocks[block_index].slot_bytes);
    }
    return {slot, block, state.generation};
}
void llama_weight_stream_pager::begin_evaluation() {
    if (asynchronous) {
        if (demands.position() % layout.blocks.size() != 0) { throw std::invalid_argument("previous evaluation did not finish demands"); }
        for (const auto & state : states) {
            if (state.phase == llama_weight_stream_slot_phase::IN_USE) { throw std::invalid_argument("previous evaluation still owns a slot"); }
        }
        for (size_t i = 0; i < layout.blocks.size(); ++i) {
            const int block = layout.blocks[i].block_index;
            const size_t slot = (demands.position() + i) % layout.slot_count;
            stores[i]->bind_slot(buffers[slot].get());
            bindings[block] = slot;
        }
        stats.evaluations++;
        return;
    }
    if (stats.evaluations > 0) {
        for (const auto & state : states) {
            if (state.phase != llama_weight_stream_slot_phase::REUSABLE) { throw std::invalid_argument("previous evaluation still owns a slot"); }
        }
    }
    stats.evaluations++;
    if (stats.evaluations > 1) {
        if (asynchronous) {
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        new_timed = (decltype(new_timed)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_new_timed");
        event_query = (decltype(event_query)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_query");
        elapsed_ms = (decltype(elapsed_ms)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_event_elapsed_ms");
        if (!new_timed || !event_query || !elapsed_ms) { throw std::runtime_error("CUDA prefetch event procedures unavailable"); }
        origin.reset(new_timed(ggml_backend_get_device(backend)));
        if (!origin) { throw std::runtime_error("prefetch origin event allocation failed"); }
        ggml_backend_event_record(origin.get(), backend);
        ggml_backend_event_synchronize(origin.get());
        pending.resize(layout.slot_count);
        for (size_t i = 0; i < layout.blocks.size(); ++i) { bindings[layout.blocks[i].block_index] = i % layout.slot_count; }
        for (size_t i = 0; i < layout.slot_count; ++i) {
            launch_prefetch(i, layout.blocks[i].block_index, true);
            finish_prefetch(i, nullptr, false);
        }
    } else {
        for (size_t i = 0; i < layout.slot_count; ++i) { ensure(layout.blocks[i].block_index, true); }
    }
    }
}
llama_weight_stream_ticket llama_weight_stream_pager::enter(int block, ggml_backend_t compute_backend) {
    if (asynchronous) {
        if (demands.next({}) != block) { throw std::invalid_argument("consumer differs from demand sequence"); }
        size_t slot = binding(block);
        bool found = false;
        for (const auto & state : states) {
            if (state.block == block && (state.phase == llama_weight_stream_slot_phase::READY || state.phase == llama_weight_stream_slot_phase::LOADING)) {
                slot = state.id; found = true; break;
            }
        }
        if (!found) { stats.demand_misses++; launch_prefetch(slot, block); }
        else { stats.prefetched_hits++; stats.hits++; }
        auto & state = states.at(slot);
        auto ticket = llama_weight_stream_ticket{slot, block, state.generation};
        finish_prefetch(slot, compute_backend, true);
        demands.consume(block);
        state.begin_use(block, ticket.generation);
        record(slot, "IN_USE");
        return ticket;
    }
    auto ticket = ensure(block, false);
    states[ticket.slot].begin_use(ticket.block, ticket.generation);
    record(ticket.slot, "IN_USE");
    return ticket;
}
void llama_weight_stream_pager::complete(const llama_weight_stream_ticket & ticket, ggml_backend_t compute_backend) {
    auto & state = states.at(ticket.slot);
    state.validate(ticket.block, ticket.generation);
    if (compute_backends.insert(compute_backend).second) { llama_weight_stream_require_cuda(compute_backend); }
    record(ticket.slot, "LAST_CONSUMER_COMPLETE");
    const auto start = ggml_time_us();
    ggml_backend_event_record(completion_events[ticket.slot].get(), compute_backend);
    ggml_backend_event_synchronize(completion_events[ticket.slot].get());
    stats.completion_us += ggml_time_us() - start;
    stats.completion_waits++;
    state.complete(ticket.block, ticket.generation, true);
    record(ticket.slot, "REUSABLE");
    if (asynchronous) {
        std::unordered_set<int> reserved;
        for (const auto & slot : states) {
            if (slot.phase == llama_weight_stream_slot_phase::LOADING || slot.phase == llama_weight_stream_slot_phase::READY || slot.phase == llama_weight_stream_slot_phase::IN_USE) { reserved.insert(slot.block); }
        }
        const int next = demands.next(reserved);
        if (next >= 0) { launch_prefetch(ticket.slot, next); }
    }
}
const ggml_tensor * llama_weight_stream_pager::checked_tensor(const llama_weight_stream_ticket & ticket, const std::string & name) const {
    states.at(ticket.slot).validate(ticket.block, ticket.generation);
    const auto * store = block_store(ticket.block);
    if ((asynchronous ? binding(ticket.block) : index(ticket.block) % layout.slot_count) != ticket.slot || store->tensor(0)->buffer != buffers.at(ticket.slot).get()) { throw std::invalid_argument("tensor resolved to wrong slot"); }
    for (size_t i = 0; i < store->block().tensors.size(); ++i) {
        if (store->block().tensors[i].profile.name == name) { return store->tensor(i); }
    }
    throw std::invalid_argument("tensor not selected in requested block");
}
const llama_weight_stream_transfer * llama_weight_stream_pager::block_store(int block) const { return stores.at(index(block)).get(); }
const std::vector<llama_weight_stream_slot_state> & llama_weight_stream_pager::slot_states() const { return states; }
const llama_weight_stream_pager_stats & llama_weight_stream_pager::statistics() const { return stats; }
const std::vector<llama_weight_stream_slot_trace> & llama_weight_stream_pager::history() const { return trace; }
const llama_weight_stream_pager_plan & llama_weight_stream_pager::plan() const { return layout; }

std::vector<ggml_backend_buffer_t> llama_weight_stream_runtime::initialize_pager(ggml_backend_dev_t device) {
    const auto plan = config.cached_descriptors ? config.auto_plan.pager : llama_weight_stream_make_pager_plan(config.discovered, config.target_blocks, config.slot_count, ggml_backend_get_default_buffer_type(backend.get()));
    std::vector<std::vector<llama_weight_stream_source>> sources;
    std::vector<std::vector<ggml_tensor *>> originals;
    std::vector<ggml_backend_buffer_t> temporary;
    std::unordered_set<ggml_tensor *> selected;
    for (const auto & block : plan.blocks) {
        sources.emplace_back(); originals.emplace_back();
        for (const auto & descriptor : block.tensors) {
            ggml_tensor * original = nullptr;
            for (const auto & entry : entries) {
                if (entry.metadata.name == descriptor.profile.name && entry.metadata.block_index == block.block_index) { original = entry.tensor; }
            }
            if (!original || !original->buffer || ggml_backend_buffer_get_type(original->buffer) != ggml_backend_dev_host_buffer_type(device) || !ggml_backend_buffer_is_host(original->buffer)) {
                throw std::runtime_error("pager original is not exclusively pinned host resident");
            }
            const auto actual = llama_weight_stream_describe(*original, block.block_index, llama_weight_stream_capability::STREAMABLE);
            if (actual.type != descriptor.profile.type || actual.ne != descriptor.profile.ne || actual.nb != descriptor.profile.nb || actual.stored_bytes != descriptor.profile.stored_bytes) {
                throw std::runtime_error("pager descriptor differs from discovery");
            }
            sources.back().push_back({actual.name, original->data, actual.stored_bytes});
            originals.back().push_back(original); selected.insert(original);
            if (std::find(temporary.begin(), temporary.end(), original->buffer) == temporary.end()) { temporary.push_back(original->buffer); }
        }
    }
    for (const auto & entry : entries) {
        if (std::find(temporary.begin(), temporary.end(), entry.tensor->buffer) != temporary.end() && !selected.count(entry.tensor)) {
            throw std::runtime_error("pager loader host buffer also contains unselected weights");
        }
    }
    paging_store = std::make_unique<llama_weight_stream_pager>(backend.get(), plan, sources, config.prefetch);
    for (size_t i = 0; i < plan.blocks.size(); ++i) {
        const auto * store = paging_store->block_store(plan.blocks[i].block_index);
        for (size_t j = 0; j < originals[i].size(); ++j) {
            auto * original = originals[i][j];
            original->buffer = store->host_buffer();
            original->data = static_cast<char *>(const_cast<void *>(store->staging_data())) + plan.blocks[i].tensors[j].offset;
            replacements.emplace(original, const_cast<ggml_tensor *>(store->tensor(j)));
        }
    }
    return temporary;
}

bool llama_weight_stream_runtime::evaluation_callback(ggml_tensor * tensor, bool ask, void * runtime) {
    auto & manager = *static_cast<llama_weight_stream_runtime *>(runtime);
    // Submit preceding work before asking for the next streamed weight.
    if (manager.demand_boundaries.count(tensor)) { return true; }
    const auto found = manager.consumers.find(tensor);
    if (found == manager.consumers.end()) { return ask ? false : true; }
    const int block = found->second;
    if (ask) {
        if (manager.first_consumers.at(block) == tensor) {
            manager.tickets[block] = manager.paging_store->enter(block, ggml_backend_sched_get_tensor_backend(manager.evaluation_sched, tensor));
        }
        const auto ticket = manager.tickets.find(block);
        if (ticket == manager.tickets.end()) { throw std::runtime_error("streamed consumer precedes its residency guard"); }
        const auto * expected = manager.paging_store->checked_tensor(ticket->second, ggml_get_name(tensor->src[0]));
        if (tensor->src[0] != expected) { throw std::runtime_error("streamed consumer uses a stale or wrong slot descriptor"); }
        return manager.last_consumers.at(block) == tensor;
    }
    if (manager.last_consumers.at(block) != tensor) { throw std::runtime_error("completion callback was not the last weight consumer"); }
    manager.paging_store->complete(manager.tickets.at(block), ggml_backend_sched_get_tensor_backend(manager.evaluation_sched, tensor));
    return true;
}

llama_weight_stream_pager::~llama_weight_stream_pager() {
    // Transfer stores and their shared buffers must outlive queued copies.
    for (auto & store : stores) { store->synchronize(); }
}
size_t llama_weight_stream_pager::binding(int block) const { return bindings.at(block); }
bool llama_weight_stream_pager::prefetching() const { return asynchronous; }
const std::vector<llama_weight_stream_pending_prefetch> & llama_weight_stream_pager::pending_prefetches() const { return pending; }
void llama_weight_stream_pager::validate_pending(const llama_weight_stream_ticket & ticket) const {
    const auto & transfer = pending.at(ticket.slot);
    const auto & state = states.at(ticket.slot);
    if (!transfer.active || state.phase != llama_weight_stream_slot_phase::LOADING || transfer.ticket.block != ticket.block || transfer.ticket.generation != ticket.generation || state.block != ticket.block || state.generation != ticket.generation) {
        throw std::invalid_argument("stale or wrong pending transfer");
    }
}
void llama_weight_stream_pager::launch_prefetch(size_t slot, int block, bool cold) {
    auto & state = states.at(slot);
    auto & active = pending.at(slot);
    if (active.active) { throw std::invalid_argument("duplicate pending prefetch"); }
    const size_t block_index = index(block);
    const auto old_reuses = state.reuses, old_evictions = state.evictions;
    if (state.phase == llama_weight_stream_slot_phase::REUSABLE && state.block != block) { record(slot, "EVICT"); }
    state.begin_load(block, layout.blocks[block_index].slot_bytes);
    stats.reuses += state.reuses - old_reuses;
    stats.evictions += state.evictions - old_evictions;
    stores[block_index]->bind_slot(buffers.at(slot).get());
    bindings[block] = slot;
    uint64_t demand = demands.position();
    const auto sequence = weight_stream_sequence(layout);
    while (sequence[demand % sequence.size()] != block) { demand++; }
    auto start = ggml_backend_event_ptr(new_timed(ggml_backend_get_device(backend)));
    auto end = ggml_backend_event_ptr(new_timed(ggml_backend_get_device(backend)));
    if (!start || !end) { throw std::runtime_error("prefetch timing event allocation failed"); }
    active = {true, {slot, block, state.generation}, demand, layout.blocks[block_index].slot_bytes, transfers.size()};
    llama_weight_stream_transfer_trace row;
    row.evaluation = stats.evaluations; row.ticket = active.ticket; row.bytes = active.bytes;
    row.demand = demand; row.launch_us = ggml_time_us(); row.cold = cold;
    transfers.push_back(row);
    timing.emplace_back(std::move(start),std::move(end));
    ggml_backend_event_record(timing.back().first.get(), backend);
    stores[block_index]->upload();
    ggml_backend_event_record(timing.back().second.get(), backend);
    stats.loads++; stats.misses++; stats.upload_bytes += active.bytes;
    record(slot, "PREFETCH_LOADING", active.bytes);
}
void llama_weight_stream_pager::finish_prefetch(size_t slot, ggml_backend_t compute_backend, bool consumer) {
    auto & active = pending.at(slot);
    auto & state = states.at(slot);
    auto & row = transfers.at(active.transfer);
    auto & store = stores.at(index(state.block));
    if (consumer) {
        if (row.consumed) { throw std::invalid_argument("prefetch consumed twice"); }
        row.needed_us = ggml_time_us(); row.consumed = true;
    }
    if (active.active) {
        validate_pending(active.ticket);
        const bool done = event_query(store->completion_event());
        const auto before = ggml_time_us();
        store->synchronize();
        const auto waited = ggml_time_us() - before;
        if (consumer && !done) { row.wait_us = waited; stats.exposed_wait_us += waited; }
        state.finish_load(); active.active = false; row.ready_us = ggml_time_us();
        record(slot, "READY");
        if (verified.insert(state.block).second) {
            store->verify(); stats.verified_blocks++; record(slot, "BYTE_VERIFIED", row.bytes);
        }
    }
    if (compute_backend) { ggml_backend_event_wait(compute_backend, store->completion_event()); }
}
void llama_weight_stream_pager::finish_timeline() {
    for (size_t i = 0; i < transfers.size(); ++i) {
        ggml_backend_event_synchronize(timing[i].second.get());
        transfers[i].start_ms = elapsed_ms(origin.get(),timing[i].first.get());
        transfers[i].end_ms = elapsed_ms(origin.get(),timing[i].second.get());
    }
}
const std::vector<llama_weight_stream_transfer_trace> & llama_weight_stream_pager::transfer_history() const { return transfers; }
