// SPDX-License-Identifier: MIT
// Copyright (c) 2026 dienmaytientam-coder

#include "llama-weight-stream.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-cpp.h"
#include "llama-impl.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

using wstream_json = nlohmann::ordered_json;

static uint64_t ws_add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) { throw std::invalid_argument("weight stream size overflow"); }
    return a + b;
}
static uint64_t ws_mul(uint64_t a, uint64_t b) {
    if (a && b > UINT64_MAX / a) { throw std::invalid_argument("weight stream size overflow"); }
    return a * b;
}
static uint64_t ws_align(uint64_t n, uint64_t a) {
    if (!a || (a & (a - 1))) { throw std::invalid_argument("invalid allocation alignment"); }
    return ws_add(n, a - 1) & ~(a - 1);
}
static std::string ws_hash(const std::string & value) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : value) { hash = (hash ^ c) * 1099511628211ULL; }
    return std::to_string(hash);
}
uint64_t llama_weight_stream_parse_size(const std::string & value, bool allow_zero) {
    if (value.empty()) { throw std::invalid_argument("empty memory size"); }
    size_t end = value.size();
    uint64_t unit = 1024ULL * 1024;
    const char suffix = value.back();
    if (suffix == 'M' || suffix == 'G') {
        --end;
        if (suffix == 'G') { unit *= 1024; }
    }
    uint64_t whole = 0, fraction = 0, scale = 1;
    bool dot = false, digit = false, fractional_digit = false;
    for (size_t i = 0; i < end; ++i) {
        const char c = value[i];
        if (c == '.' && !dot && digit && end != value.size()) { dot = true; continue; }
        if (c < '0' || c > '9') { throw std::invalid_argument("invalid memory size: " + value); }
        digit = true;
        if (dot) {
            scale = ws_mul(scale, 10);
            fraction = ws_add(ws_mul(fraction, 10), c - '0');
            fractional_digit = true;
        } else { whole = ws_add(ws_mul(whole, 10), c - '0'); }
    }
    if (!digit || (dot && !fractional_digit)) { throw std::invalid_argument("invalid memory size"); }
    // Integer arithmetic gives a locale-independent floor to whole bytes.
    const uint64_t bytes = ws_add(ws_mul(whole, unit), ws_mul(fraction, unit) / scale);
    if ((!bytes && !allow_zero) || bytes > SIZE_MAX) { throw std::invalid_argument("memory size out of range"); }
    return bytes;
}

llama_weight_stream_model_identity llama_weight_stream_identify(const std::string & path) {
    llama_weight_stream_model_identity id;
    id.canonical_path = std::filesystem::canonical(path).string();
    struct stat info;
    if (stat(id.canonical_path.c_str(), &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0) { throw std::invalid_argument("cannot stat weight stream model"); }
    id.file_size = info.st_size;
#if defined(__APPLE__)
    id.mtime_ns = int64_t(info.st_mtimespec.tv_sec) * 1000000000 + info.st_mtimespec.tv_nsec;
#elif defined(_WIN32)
    id.mtime_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::filesystem::last_write_time(path).time_since_epoch()).count();
#else
    id.mtime_ns = int64_t(info.st_mtim.tv_sec) * 1000000000 + info.st_mtim.tv_nsec;
#endif
    return id;
}
static wstream_json ws_identity(const llama_weight_stream_model_identity & id) {
    return {{"canonical_path", id.canonical_path}, {"file_size", id.file_size}, {"mtime_ns", id.mtime_ns}};
}
static uint64_t ws_uint(const wstream_json & j) {
    if (!j.is_number_integer() || (j.is_number_integer() && !j.is_number_unsigned() && j.get<int64_t>() < 0)) { throw std::invalid_argument("invalid unsigned cache integer"); }
    const auto n = j.get<uint64_t>();
    if (n > SIZE_MAX) { throw std::invalid_argument("cache integer overflow"); }
    return n;
}
static int ws_int(const wstream_json & j) {
    if (!j.is_number_integer() || (j.is_number_unsigned() && j.get<uint64_t>() > INT_MAX)) { throw std::invalid_argument("invalid cache integer"); }
    const auto n = j.get<int64_t>();
    if (n < INT_MIN || n > INT_MAX) { throw std::invalid_argument("cache integer overflow"); }
    return int(n);
}
static wstream_json ws_tensor(const llama_weight_stream_tensor_profile & t) {
    return {{"name",t.name},{"block",t.block_index},{"type",int(t.type)},{"ne",t.ne},{"nb",t.nb},
        {"logical_bytes",t.logical_bytes},{"stored_bytes",t.stored_bytes},{"allocation_bytes",t.allocation_bytes},
        {"alignment",t.alignment},{"cuda_weight",t.cuda_weight},{"capability",int(t.capability)}};
}
static llama_weight_stream_tensor_profile ws_tensor(const wstream_json & j) {
    llama_weight_stream_tensor_profile t;
    t.name = j.at("name").get<std::string>(); t.block_index = ws_int(j.at("block"));
    const int type = ws_int(j.at("type"));
    if (type < 0 || type >= GGML_TYPE_COUNT || ggml_type_size(ggml_type(type)) == 0) { throw std::invalid_argument("invalid cached tensor type"); }
    t.type = ggml_type(type);
    const auto & ne = j.at("ne"); const auto & nb = j.at("nb");
    if (!ne.is_array() || !nb.is_array() || ne.size() != GGML_MAX_DIMS || nb.size() != GGML_MAX_DIMS) { throw std::invalid_argument("invalid cached dimensions"); }
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        const uint64_t n = ws_uint(ne[d]);
        if (!n || n > INT64_MAX) { throw std::invalid_argument("invalid cached dimension"); }
        t.ne[d] = int64_t(n); t.nb[d] = ws_uint(nb[d]);
    }
    t.logical_bytes = ws_uint(j.at("logical_bytes")); t.stored_bytes = ws_uint(j.at("stored_bytes"));
    t.allocation_bytes = ws_uint(j.at("allocation_bytes")); t.alignment = ws_uint(j.at("alignment"));
    t.cuda_weight = j.at("cuda_weight").get<bool>();
    t.capability = llama_weight_stream_capability(ws_int(j.at("capability")));
    if (t.cuda_weight && (t.allocation_bytes < t.logical_bytes || !t.alignment || (t.alignment & (t.alignment - 1)))) { throw std::invalid_argument("invalid cached CUDA allocation"); }
    return t;
}
std::string llama_weight_stream_profile_json(const llama_weight_stream_model_profile & p, const llama_weight_stream_model_identity & id) {
    llama_weight_stream_validate_profile(p);
    wstream_json j = {{"schema_version",1},{"profile_version",2},{"model_identity",ws_identity(id)},
        {"architecture",p.architecture},{"descriptor_fingerprint",p.model_fingerprint},{"block_count",p.blocks.size()},
        {"demand_order",p.demand_order},{"blocks",wstream_json::array()},{"non_block_tensors",wstream_json::array()}};
    size_t count = p.non_block_tensors.size();
    for (const auto & b : p.blocks) {
        wstream_json block = {{"block",b.block_index},{"tensors",wstream_json::array()}};
        for (const auto & t : b.tensors) { block["tensors"].push_back(ws_tensor(t)); }
        count += b.tensors.size(); j["blocks"].push_back(block);
    }
    for (const auto & t : p.non_block_tensors) { j["non_block_tensors"].push_back(ws_tensor(t)); }
    j["tensor_count"] = count;
    return j.dump(2) + "\n";
}
llama_weight_stream_model_profile llama_weight_stream_profile_from_json(const std::string & data, llama_weight_stream_model_identity & id) {
    const auto j = wstream_json::parse(data);
    if (ws_uint(j.at("schema_version")) != 1 || ws_uint(j.at("profile_version")) != 2) { throw std::invalid_argument("stale profile version"); }
    const auto & identity = j.at("model_identity");
    id.canonical_path = identity.at("canonical_path").get<std::string>();
    id.file_size = ws_uint(identity.at("file_size"));
    if (!identity.at("mtime_ns").is_number_integer()) { throw std::invalid_argument("invalid cached mtime"); }
    id.mtime_ns = identity.at("mtime_ns").get<int64_t>();
    llama_weight_stream_model_profile p;
    p.architecture = j.at("architecture").get<std::string>(); p.model_fingerprint = j.at("descriptor_fingerprint").get<std::string>();
    if (id.canonical_path.empty() || !id.file_size || p.model_fingerprint.empty()) { throw std::invalid_argument("missing cache identity"); }
    if (!j.at("blocks").is_array() || !j.at("non_block_tensors").is_array() || !j.at("demand_order").is_array()) { throw std::invalid_argument("invalid profile arrays"); }
    for (const auto & b : j.at("blocks")) {
        std::vector<llama_weight_stream_tensor_profile> tensors;
        for (const auto & t : b.at("tensors")) { tensors.push_back(ws_tensor(t)); }
        p.blocks.push_back(llama_weight_stream_profile_block(ws_int(b.at("block")), std::move(tensors)));
    }
    for (const auto & t : j.at("non_block_tensors")) { p.non_block_tensors.push_back(ws_tensor(t)); }
    for (const auto & b : j.at("demand_order")) { p.demand_order.push_back(ws_int(b)); }
    size_t count = p.non_block_tensors.size();
    for (const auto & b : p.blocks) { count += b.tensors.size(); }
    if (count != ws_uint(j.at("tensor_count")) || p.blocks.size() != ws_uint(j.at("block_count"))) { throw std::invalid_argument("cached profile count mismatch"); }
    llama_weight_stream_validate_profile(p);
    std::set<int> order;
    for (int b : p.demand_order) {
        bool found = false;
        for (const auto & block : p.blocks) { found |= block.block_index == b; }
        if (!found || !order.insert(b).second) { throw std::invalid_argument("invalid demand order"); }
    }
    return p;
}
static wstream_json ws_options(const llama_weight_stream_options & o) {
    return {{"mode",o.mode},{"gpu_budget",o.gpu_budget},{"reserve",o.reserve},{"runtime_cuda_bytes",o.runtime_cuda_bytes},{"external_cuda_bytes",o.external_cuda_bytes},{"runtime_key",o.runtime_key}};
}
std::string llama_weight_stream_plan_key(const llama_weight_stream_model_profile & p, const llama_weight_stream_hardware & h, const llama_weight_stream_options & o) {
    wstream_json key = {{"descriptor",p.model_fingerprint},{"profile_version",p.profile_version},
        {"planner_version",2},{"algorithm_version",2},{"hardware_identity",h.identity},
        {"compute_capability",h.compute_capability},{"total_vram",h.total_vram},{"config",ws_options(o)},
        {"slot_count",2},{"slot_policy","cyclic-block-async-v1"}};
    return ws_hash(key.dump());
}
static uint64_t ws_resident(const llama_weight_stream_model_profile & p) {
    uint64_t full = 0;
    auto add = [&](const auto & t) { if (t.cuda_weight) { full = ws_add(full, ws_align(t.allocation_bytes,t.alignment)); } };
    for (const auto & b : p.blocks) { for (const auto & t : b.tensors) { add(t); } }
    for (const auto & t : p.non_block_tensors) { add(t); }
    return full;
}
static uint64_t ws_removed(const llama_weight_stream_block_layout & b) {
    uint64_t bytes = 0;
    for (const auto & t : b.tensors) { bytes = ws_add(bytes, ws_align(t.profile.allocation_bytes,t.profile.alignment)); }
    return bytes;
}
void llama_weight_stream_validate_descriptor(const llama_weight_stream_tensor_profile & expected, const llama_weight_stream_tensor_profile & actual) {
    if (expected.name != actual.name || expected.block_index != actual.block_index || expected.type != actual.type || expected.ne != actual.ne || expected.nb != actual.nb || expected.stored_bytes != actual.stored_bytes) { throw std::invalid_argument("cached weight descriptor differs from runtime tensor"); }
}
static void ws_validate_plan(const llama_weight_stream_auto_plan_result & p) {
    if (p.schema_version != 1 || p.planner_version != 2 || p.options.mode != "auto" || p.options.gpu_budget <= p.options.reserve || p.fingerprint.empty() || p.hardware_key.empty() || p.plan_key.empty()) { throw std::invalid_argument("invalid cached plan contract"); }
    uint64_t removed = 0;
    if (!p.pager.blocks.empty()) {
        llama_weight_stream_validate_pager_plan(p.pager);
        if (p.pager.slot_count != 2 || p.pager.slot_capacity == 0) { throw std::invalid_argument("invalid cached slot policy"); }
        std::vector<int> demands;
        for (const auto & b : p.pager.blocks) { removed = ws_add(removed, ws_removed(b)); demands.push_back(b.block_index); }
        if (demands != p.demand_order) { throw std::invalid_argument("invalid cached demand order"); }
    } else if (p.pager.slot_count || p.pager.slot_capacity || !p.demand_order.empty()) { throw std::invalid_argument("invalid empty plan"); }
    const uint64_t slots = ws_mul(p.pager.slot_count,p.pager.slot_capacity);
    if (removed > p.full_resident_bytes || removed < slots || removed != p.pager.removed_weight_bytes || p.slot_bytes != slots || p.extra_cuda_bytes != ws_add(p.options.runtime_cuda_bytes,p.options.external_cuda_bytes)) { throw std::invalid_argument("cached accounting mismatch"); }
    const uint64_t expected = ws_add(p.full_resident_bytes - removed,ws_add(slots,p.extra_cuda_bytes));
    if (p.expected_bytes != expected || expected > p.options.gpu_budget - p.options.reserve || p.net_saving != removed - slots || (!p.pager.blocks.empty() && !p.net_saving)) { throw std::invalid_argument("cached budget violation"); }
}
std::string llama_weight_stream_plan_json(const llama_weight_stream_auto_plan_result & p) {
    ws_validate_plan(p);
    wstream_json j = {{"schema_version",p.schema_version},{"planner_version",p.planner_version},
        {"descriptor_fingerprint",p.fingerprint},{"hardware_key",p.hardware_key},{"hardware",{{"identity",p.hardware.identity},{"compute_capability",p.hardware.compute_capability},{"total_vram",p.hardware.total_vram}}},{"plan_key",p.plan_key},{"config",ws_options(p.options)},
        {"slot_count",p.pager.slot_count},{"slot_capacity",p.pager.slot_capacity},{"slot_bytes",p.slot_bytes},
        {"full_resident_bytes",p.full_resident_bytes},{"removed_weight_bytes",p.pager.removed_weight_bytes},
        {"extra_cuda_bytes",p.extra_cuda_bytes},{"expected_net_saving",p.net_saving},{"expected_managed_bytes",p.expected_bytes},
        {"candidate_count",p.candidate_count},{"demand_order",p.demand_order},{"prefetch_policy","cyclic-future-demand-v1"},{"blocks",wstream_json::array()}};
    for (const auto & b : p.pager.blocks) {
        wstream_json block = {{"block",b.block_index},{"alignment",b.alignment},{"packed_bytes",b.slot_bytes},{"tensors",wstream_json::array()}};
        for (const auto & t : b.tensors) { block["tensors"].push_back({{"descriptor",ws_tensor(t.profile)},{"offset",t.offset}}); }
        j["blocks"].push_back(block);
    }
    return j.dump(2) + "\n";
}
llama_weight_stream_auto_plan_result llama_weight_stream_plan_from_json(const std::string & data) {
    const auto j = wstream_json::parse(data);
    llama_weight_stream_auto_plan_result p;
    if (ws_uint(j.at("schema_version")) != 1 || ws_uint(j.at("planner_version")) != 2 || j.at("prefetch_policy") != "cyclic-future-demand-v1") { throw std::invalid_argument("stale plan version"); }
    p.fingerprint = j.at("descriptor_fingerprint").get<std::string>(); p.hardware_key = j.at("hardware_key").get<std::string>(); p.plan_key = j.at("plan_key").get<std::string>();
    p.hardware.identity = j.at("hardware").at("identity").get<std::string>();
    p.hardware.compute_capability = ws_int(j.at("hardware").at("compute_capability"));
    p.hardware.total_vram = ws_uint(j.at("hardware").at("total_vram"));
    p.options.mode = j.at("config").at("mode").get<std::string>();
    p.options.gpu_budget = ws_uint(j.at("config").at("gpu_budget")); p.options.reserve = ws_uint(j.at("config").at("reserve"));
    p.options.runtime_cuda_bytes = ws_uint(j.at("config").at("runtime_cuda_bytes"));
    p.options.external_cuda_bytes = ws_uint(j.at("config").at("external_cuda_bytes"));
    p.options.runtime_key = j.at("config").at("runtime_key").get<std::string>();
    p.pager.slot_count = ws_uint(j.at("slot_count")); p.pager.slot_capacity = ws_uint(j.at("slot_capacity")); p.slot_bytes = ws_uint(j.at("slot_bytes"));
    p.full_resident_bytes = ws_uint(j.at("full_resident_bytes")); p.pager.removed_weight_bytes = ws_uint(j.at("removed_weight_bytes"));
    p.extra_cuda_bytes = ws_uint(j.at("extra_cuda_bytes")); p.net_saving = ws_uint(j.at("expected_net_saving")); p.expected_bytes = ws_uint(j.at("expected_managed_bytes")); p.candidate_count = ws_uint(j.at("candidate_count"));
    for (const auto & b : j.at("demand_order")) { p.demand_order.push_back(ws_int(b)); }
    for (const auto & b : j.at("blocks")) {
        llama_weight_stream_block_layout block;
        block.block_index = ws_int(b.at("block")); block.alignment = ws_uint(b.at("alignment")); block.slot_bytes = ws_uint(b.at("packed_bytes"));
        for (const auto & t : b.at("tensors")) { block.tensors.push_back({ws_tensor(t.at("descriptor")), size_t(ws_uint(t.at("offset")))}); }
        p.pager.blocks.push_back(std::move(block));
    }
    ws_validate_plan(p);
    return p;
}

llama_weight_stream_auto_plan_result llama_weight_stream_auto_plan(const llama_weight_stream_model_profile & profile, const llama_weight_stream_hardware & hw, const llama_weight_stream_options & opts, ggml_backend_buffer_type_t buft) {
    llama_weight_stream_validate_profile(profile);
    if (opts.mode != "auto" || opts.gpu_budget <= opts.reserve || opts.gpu_budget > hw.total_vram || !buft || hw.identity.empty()) { throw std::invalid_argument("invalid GPU weight stream budget or hardware"); }
    llama_weight_stream_auto_plan_result best;
    best.fingerprint = profile.model_fingerprint; best.hardware_key = hw.identity; best.hardware = hw; best.plan_key = llama_weight_stream_plan_key(profile,hw,opts); best.options = opts;
    best.full_resident_bytes = ws_resident(profile); best.extra_cuda_bytes = ws_add(opts.runtime_cuda_bytes,opts.external_cuda_bytes); best.expected_bytes = ws_add(best.full_resident_bytes,best.extra_cuda_bytes);
    if (best.extra_cuda_bytes >= opts.gpu_budget - opts.reserve) { throw std::invalid_argument("runtime CUDA allocations exceed GPU budget after reserve"); }
    const uint64_t budget = opts.gpu_budget - opts.reserve - best.extra_cuda_bytes;
    std::vector<llama_weight_stream_block_layout> candidates;
    for (int index : profile.demand_order) {
        for (const auto & b : profile.blocks) {
            if (b.block_index != index) { continue; }
            auto filtered = b;
            for (auto & t : filtered.tensors) { if (!t.cuda_weight && t.capability == llama_weight_stream_capability::STREAMABLE) { t.capability = llama_weight_stream_capability::UNKNOWN; } }
            filtered = llama_weight_stream_profile_block(b.block_index, std::move(filtered.tensors));
            if (filtered.streamable_bytes) { candidates.push_back(llama_weight_stream_pack(filtered,buft)); }
        }
    }
    best.candidate_count = candidates.size();
    if (best.full_resident_bytes <= budget) { return best; }
    bool have = false;
    uint64_t best_spill = UINT64_MAX, best_window = 0;
    auto evaluate = [&](const std::vector<size_t> & selected) {
        if (selected.size() <= 2) { return; }
        auto result = best;
        result.pager = {}; result.demand_order.clear(); result.net_saving = 0;
        uint64_t spill = 0;
        auto order = selected; std::sort(order.begin(),order.end());
        for (size_t i : order) {
            result.pager.blocks.push_back(candidates[i]); result.demand_order.push_back(candidates[i].block_index);
            result.pager.slot_capacity = std::max(result.pager.slot_capacity,candidates[i].slot_bytes);
            result.pager.removed_weight_bytes = ws_add(result.pager.removed_weight_bytes,ws_removed(candidates[i]));
            for (const auto & t : candidates[i].tensors) { spill = ws_add(spill,t.profile.stored_bytes); }
        }
        result.pager.slot_count = 2; result.slot_bytes = ws_mul(2,result.pager.slot_capacity);
        if (result.pager.removed_weight_bytes <= result.slot_bytes) { return; }
        result.net_saving = result.pager.removed_weight_bytes - result.slot_bytes;
        result.expected_bytes = ws_add(result.full_resident_bytes - result.net_saving,result.extra_cuda_bytes);
        if (result.expected_bytes - result.extra_cuda_bytes > budget) { return; }
        // The smallest cyclic two-demand gap is the useful next-use window.
        uint64_t window = UINT64_MAX;
        for (size_t i = 0; i < order.size(); ++i) {
            auto position = [&](size_t candidate) { return size_t(std::find(profile.demand_order.begin(),profile.demand_order.end(),candidates[candidate].block_index) - profile.demand_order.begin()); };
            const auto a = position(order[i]), b = position(order[(i+2)%order.size()]);
            const uint64_t gap = (b + profile.demand_order.size() - a) % profile.demand_order.size();
            window = std::min(window,gap);
        }
        if (!have || spill < best_spill || (spill == best_spill && (result.slot_bytes < best.slot_bytes || (result.slot_bytes == best.slot_bytes && (window > best_window || (window == best_window && result.demand_order < best.demand_order)))))) {
            have = true; best = std::move(result); best_spill = spill; best_window = window;
        }
    };
    for (const auto & threshold : candidates) {
        const uint64_t required = ws_add(best.full_resident_bytes - budget,ws_mul(2,threshold.slot_bytes));
        std::vector<size_t> pool, selected;
        for (size_t i = 0; i < candidates.size(); ++i) { if (candidates[i].slot_bytes <= threshold.slot_bytes) { pool.push_back(i); } }
        std::sort(pool.begin(),pool.end(),[&](size_t a,size_t b) { const auto x=ws_removed(candidates[a]),y=ws_removed(candidates[b]); return x != y ? x < y : a < b; });
        uint64_t removed = 0;
        while (!pool.empty() && (removed < required || selected.size() <= 2)) {
            size_t choice = 0;
            const uint64_t remaining = removed < required ? required - removed : 0;
            // Take the smallest completing block, else accumulate small blocks.
            for (size_t i = 0; i < pool.size(); ++i) { if (ws_removed(candidates[pool[i]]) >= remaining) { choice = i; break; } }
            removed = ws_add(removed,ws_removed(candidates[pool[choice]])); selected.push_back(pool[choice]); pool.erase(pool.begin()+choice);
        }
        for (size_t i = selected.size(); i-- > 0;) {
            const auto bytes = ws_removed(candidates[selected[i]]);
            if (selected.size() > 3 && removed - bytes >= required) { removed -= bytes; selected.erase(selected.begin()+i); }
        }
        evaluate(selected);
        for (size_t i = 0; i < selected.size(); ++i) {
            for (size_t alternative : pool) { auto swapped = selected; swapped[i] = alternative; evaluate(swapped); }
        }
    }
    if (!have) { throw std::invalid_argument("weight stream budget impossible with proven weights and two slots"); }
    ws_validate_plan(best);
    return best;
}

static std::string ws_read(const std::string & path) {
    const auto size = std::filesystem::file_size(path);
    if (size > 64ULL * 1024 * 1024) { throw std::invalid_argument("metadata cache too large"); }
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open metadata cache"); }
    std::string data((std::istreambuf_iterator<char>(in)), {});
    if (in.bad()) { throw std::runtime_error("cannot read metadata cache"); }
    return data;
}
static void ws_write(const std::string & path, const std::string & data) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
#ifndef _WIN32
    std::string name = path + ".tmp.XXXXXX";
    std::vector<char> temp(name.begin(),name.end()); temp.push_back(0);
    const int fd = mkstemp(temp.data());
    if (fd < 0) { throw std::runtime_error("cannot create cache temporary file"); }
    bool open = true;
    try {
        size_t written = 0;
        while (written < data.size()) {
            const auto n = write(fd,data.data()+written,data.size()-written);
            if (n < 0 && errno == EINTR) { continue; }
            if (n <= 0) { throw std::runtime_error("cannot write cache"); }
            written += n;
        }
        if (fsync(fd)) { throw std::runtime_error("cannot flush cache"); }
        const int result = close(fd); open = false;
        if (result) { throw std::runtime_error("cannot close cache"); }
        std::filesystem::rename(temp.data(),path);
        const int dir = ::open(std::filesystem::path(path).parent_path().c_str(),O_RDONLY | O_DIRECTORY);
        if (dir >= 0) { fsync(dir); close(dir); }
    } catch (...) { if (open) { close(fd); } unlink(temp.data()); throw; }
#else
    throw std::runtime_error("atomic weight stream cache writes are unsupported on this platform");
#endif
}
std::string llama_weight_stream_profile_path(const llama_weight_stream_options & opts, const llama_weight_stream_model_identity & id) {
    // Keep one profile location per canonical path so changed stats are STALE.
    return (std::filesystem::path(opts.cache_dir)/"models"/ws_hash(id.canonical_path)/"profile.v1.json").string();
}
llama_weight_stream_model_profile llama_weight_stream_cached_profile(const llama_weight_stream_options & opts, const llama_weight_stream_model_identity & id, const std::function<llama_weight_stream_model_profile()> & profiler) {
    const auto path = llama_weight_stream_profile_path(opts,id);
    const auto start = ggml_time_us();
    if (opts.cache && std::filesystem::exists(path)) {
        try {
            llama_weight_stream_model_identity cached;
            auto profile = llama_weight_stream_profile_from_json(ws_read(path),cached);
            if (cached.canonical_path != id.canonical_path || cached.file_size != id.file_size || cached.mtime_ns != id.mtime_ns) { throw std::invalid_argument("model identity changed"); }
            fprintf(stderr, "vn-wstream: profile cache HIT\nvn-wstream: profile scan skipped\nvn-wstream: profile_lookup_ms=%.3f profiler_invocations=0\n", (ggml_time_us()-start)/1000.0);
            return profile;
        } catch (const std::exception & e) { fprintf(stderr, "vn-wstream: profile cache STALE/invalid: %s\n",e.what()); }
    }
    fprintf(stderr, "vn-wstream: profile cache MISS\nvn-wstream: profiling model ...\n");
    const auto scan = ggml_time_us(); auto profile = profiler();
    fprintf(stderr, "vn-wstream: profile_scan_ms=%.3f profiler_invocations=1\n",(ggml_time_us()-scan)/1000.0);
    if (opts.cache) {
        const auto write = ggml_time_us(); ws_write(path,llama_weight_stream_profile_json(profile,id));
        fprintf(stderr, "vn-wstream: profile cache written\nvn-wstream: profile_write_ms=%.3f\n",(ggml_time_us()-write)/1000.0);
    }
    return profile;
}
static void ws_validate_cached(const llama_weight_stream_auto_plan_result & plan, const llama_weight_stream_model_profile & p, const llama_weight_stream_hardware & h, const llama_weight_stream_options & o) {
    ws_validate_plan(plan);
    if (plan.hardware.identity != h.identity || plan.hardware.compute_capability != h.compute_capability || plan.hardware.total_vram != h.total_vram) { throw std::invalid_argument("cached hardware mismatch"); }
    if (plan.plan_key != llama_weight_stream_plan_key(p,h,o) || plan.fingerprint != p.model_fingerprint || plan.hardware_key != h.identity || ws_options(plan.options) != ws_options(o) || plan.full_resident_bytes != ws_resident(p)) { throw std::invalid_argument("cached plan input mismatch"); }
    size_t cursor = 0;
    for (const auto & block : plan.pager.blocks) {
        while (cursor < p.demand_order.size() && p.demand_order[cursor] != block.block_index) { ++cursor; }
        if (cursor++ >= p.demand_order.size()) { throw std::invalid_argument("cached plan order mismatch"); }
        const auto source = llama_weight_stream_plan(p,block.block_index);
        size_t count = 0;
        for (const auto & t : source.tensors) { if (t.cuda_weight && t.capability == llama_weight_stream_capability::STREAMABLE) { ++count; } }
        if (block.tensors.size() != count) { throw std::invalid_argument("cached selected tensor count mismatch"); }
        for (const auto & t : block.tensors) {
            bool found = false;
            for (const auto & actual : source.tensors) { if (actual.name == t.profile.name) {
                llama_weight_stream_validate_descriptor(t.profile,actual);
                if (!actual.cuda_weight || actual.capability != llama_weight_stream_capability::STREAMABLE || actual.allocation_bytes != t.profile.allocation_bytes || actual.alignment != t.profile.alignment) { throw std::invalid_argument("cached selected capability mismatch"); }
                found = true;
            } }
            if (!found) { throw std::invalid_argument("cached selected tensor missing"); }
        }
    }
}
llama_weight_stream_auto_plan_result llama_weight_stream_cached_plan(const llama_weight_stream_options & opts, const llama_weight_stream_model_profile & profile, const llama_weight_stream_hardware & hw, const std::function<llama_weight_stream_auto_plan_result(const llama_weight_stream_model_profile &)> & planner) {
    const auto key = llama_weight_stream_plan_key(profile,hw,opts);
    const auto path = (std::filesystem::path(opts.cache_dir)/"plans"/key).string()+".json";
    const auto start = ggml_time_us();
    if (opts.cache && std::filesystem::exists(path)) {
        try {
            auto plan = llama_weight_stream_plan_from_json(ws_read(path)); ws_validate_cached(plan,profile,hw,opts);
            fprintf(stderr, "vn-wstream: plan cache HIT\nvn-wstream: planner skipped\nvn-wstream: plan_lookup_ms=%.3f planner_invocations=0\n",(ggml_time_us()-start)/1000.0);
            return plan;
        } catch (const std::exception & e) { fprintf(stderr, "vn-wstream: plan cache invalid: %s\n",e.what()); }
    }
    fprintf(stderr, "vn-wstream: plan cache MISS\nvn-wstream: planner running\n");
    const auto scan = ggml_time_us(); auto plan = planner(profile);
    ws_validate_cached(plan,profile,hw,opts);
    fprintf(stderr, "vn-wstream: planner_ms=%.3f planner_invocations=1\n",(ggml_time_us()-scan)/1000.0);
    if (opts.cache) {
        const auto write = ggml_time_us(); ws_write(path,llama_weight_stream_plan_json(plan));
        fprintf(stderr, "vn-wstream: plan cache written\nvn-wstream: plan_write_ms=%.3f\n",(ggml_time_us()-write)/1000.0);
    }
    return plan;
}

llama_model * llama_weight_stream_load_auto(const char * path, llama_model_params params, llama_context_params cp, const llama_weight_stream_options & requested, const llama_context_params * draft_params, bool draft_backend_sampling) {
    auto opts = requested;
    if (opts.mode == "off") { return llama_model_load_from_file(path,params); }
    if (params.tensor_buft_overrides && !params.tensor_buft_overrides[0].pattern) { params.tensor_buft_overrides = nullptr; }
    if (params.kv_overrides && !params.kv_overrides[0].key[0]) { params.kv_overrides = nullptr; }
    if (opts.mode != "auto" || params.no_alloc || params.tensor_buft_overrides || params.kv_overrides || cp.cb_eval || cp.n_seq_max != 1 || cp.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || params.load_mtp != (draft_params != nullptr)) { throw std::invalid_argument("unsupported weight stream load configuration"); }
    if (draft_params && (draft_params->ctx_type != LLAMA_CONTEXT_TYPE_MTP || draft_params->n_seq_max != 1 || draft_params->cb_eval || !draft_params->offload_kqv || !draft_params->op_offload)) { throw std::invalid_argument("unsupported embedded MTP context"); }
    ggml_backend_dev_t device = params.devices ? params.devices[0] : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device || (params.devices && params.devices[1]) || std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device))) != "CUDA") { throw std::invalid_argument("auto weight streaming requires exactly one CUDA GPU"); }
    llama_weight_stream_hardware hw;
    ggml_backend_dev_props props; ggml_backend_dev_get_props(device,&props);
    hw.total_vram = props.memory_total;
    using hardware_fn = bool (*)(ggml_backend_dev_t,char *,size_t,int *);
    auto get_hardware = (hardware_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device),"ggml_backend_cuda_weight_stream_identity");
    char identity[128] = {};
    if (!get_hardware || !get_hardware(device,identity,sizeof(identity),&hw.compute_capability)) { throw std::invalid_argument("CUDA hardware profile is unavailable"); }
    hw.identity = identity;
    if (opts.gpu_budget <= opts.reserve || opts.gpu_budget > hw.total_vram) { throw std::invalid_argument("invalid GPU budget/reserve"); }
    params.n_gpu_layers = INT_MAX; params.load_mode = LLAMA_LOAD_MODE_NONE; params.no_host = true;
    if (!params.devices) {
        size_t gpus = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) { ++gpus; } }
        if (gpus != 1) { throw std::invalid_argument("select one device for auto streaming"); }
    }
    const auto id = llama_weight_stream_identify(path);
    bool estimated = false;
    auto discover = [&](bool scan) {
        llama_weight_stream_config discovery;
        discovery.cached_descriptors = !scan;
        discovery.embedded_mtp = params.load_mtp;
        llama_weight_stream_config_scope scope(discovery);
        auto mp = params; mp.no_alloc = true; mp.load_mtp = false;
        llama_model_ptr model(llama_model_load_from_file(path,mp));
        if (!model || !model->weight_stream) { throw std::runtime_error("metadata-only model discovery failed"); }
        llama_context_ptr context(llama_init_from_model(model.get(),cp));
        if (!context) { throw std::runtime_error("metadata-only graph discovery failed"); }
        opts.runtime_cuda_bytes = 0;
        wstream_json allocation = wstream_json::array();
        for (const auto & item : context->memory_breakdown()) {
            auto dev = ggml_backend_buft_get_device(item.first);
            if (dev == device && !ggml_backend_buft_is_host(item.first)) {
                opts.runtime_cuda_bytes = ws_add(opts.runtime_cuda_bytes,ws_add(item.second.context,item.second.compute));
                allocation.push_back({ggml_backend_buft_name(item.first),item.second.context,item.second.compute});
                fprintf(stderr, "vn-wstream: estimated KV/state=%llu compute=%llu buffer=%s\n", (unsigned long long) item.second.context, (unsigned long long) item.second.compute, ggml_backend_buft_name(item.first));
            }
        }
        opts.runtime_key = ws_hash(wstream_json{{"n_ctx",cp.n_ctx},{"n_batch",cp.n_batch},{"n_ubatch",cp.n_ubatch},{"n_rs_seq",cp.n_rs_seq},{"embedded_mtp",params.load_mtp},{"type_k",int(cp.type_k)},{"type_v",int(cp.type_v)},{"flash_attn",int(cp.flash_attn_type)},{"allocations",allocation}}.dump());
        estimated = true;
        fprintf(stderr,"vn-wstream: runtime CUDA estimate=%llu external CUDA=%llu runtime_key=%s (metadata-only)\n",(unsigned long long)opts.runtime_cuda_bytes,(unsigned long long)opts.external_cuda_bytes,opts.runtime_key.c_str());
        llama_weight_stream_model_profile result;
        if (scan) {
            result = model->weight_stream->profile(model->arch_name());
            result.model_fingerprint = "profile-fnv1a64:" + ws_hash(llama_weight_stream_profile_json(result,id));
        }
        return result;
    };
    const auto profile = llama_weight_stream_cached_profile(opts,id,[&] { return discover(true); });
    if (!estimated) { discover(false); }
    uint64_t mtp_weight_bytes = 0, mtp_runtime_bytes = 0;
    if (draft_params) {
        llama_weight_stream_config discovery;
        discovery.cached_descriptors = true;
        discovery.embedded_mtp = true;
        llama_weight_stream_config_scope scope(discovery);
        auto mp = params; mp.no_alloc = true;
        llama_model_ptr model(llama_model_load_from_file(path,mp));
        if (!model || !model->hparams.n_layer_nextn) { throw std::invalid_argument("embedded MTP weights are unavailable"); }
        uint64_t weights = 0;
        for (const auto & item : model->memory_breakdown()) {
            if (ggml_backend_buft_get_device(item.first) == device && !ggml_backend_buft_is_host(item.first)) { weights = ws_add(weights,item.second); }
        }
        const auto trunk_weights = ws_resident(profile);
        if (weights < trunk_weights) { throw std::runtime_error("embedded MTP weight accounting differs from trunk profile"); }
        mtp_weight_bytes = weights - trunk_weights;
        auto draft = *draft_params;
        draft.n_ctx = cp.n_ctx ? cp.n_ctx : model->hparams.n_ctx_train;
        draft.n_rs_seq = 0;
        llama_sampler_ptr sampler;
        llama_sampler_seq_config sampling = {};
        if (draft_backend_sampling) {
            sampler.reset(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            llama_sampler_chain_add(sampler.get(),llama_sampler_init_top_k(10));
            sampling.seq_id = 0; sampling.sampler = sampler.get();
            draft.samplers = &sampling; draft.n_samplers = 1;
        }
        llama_context_ptr context(llama_init_from_model(model.get(),draft));
        if (!context) { throw std::runtime_error("metadata-only MTP context discovery failed"); }
        wstream_json allocation = wstream_json::array();
        for (const auto & item : context->memory_breakdown()) {
            if (ggml_backend_buft_get_device(item.first) == device && !ggml_backend_buft_is_host(item.first)) {
                mtp_runtime_bytes = ws_add(mtp_runtime_bytes,ws_add(item.second.context,item.second.compute));
                allocation.push_back({ggml_backend_buft_name(item.first),item.second.context,item.second.compute});
                fprintf(stderr,"vn-wstream: estimated MTP KV/state=%llu compute=%llu buffer=%s\n",(unsigned long long)item.second.context,(unsigned long long)item.second.compute,ggml_backend_buft_name(item.first));
            }
        }
        opts.runtime_cuda_bytes = ws_add(opts.runtime_cuda_bytes,mtp_runtime_bytes);
        opts.external_cuda_bytes = ws_add(opts.external_cuda_bytes,mtp_weight_bytes);
        opts.runtime_key = ws_hash(wstream_json{{"target",opts.runtime_key},{"MTP_weights",mtp_weight_bytes},{"MTP_allocations",allocation},{"backend_sampling",draft_backend_sampling}}.dump());
        fprintf(stderr,"vn-wstream: MTP weights MUST_RESIDENT=%llu runtime STATE_NOT_WEIGHT=%llu combined runtime=%llu\n",(unsigned long long)mtp_weight_bytes,(unsigned long long)mtp_runtime_bytes,(unsigned long long)opts.runtime_cuda_bytes);
    }
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_dev_memory(device, &free_vram, &total_vram);
    if (free_vram > total_vram) { throw std::runtime_error("invalid CUDA memory estimate"); }
    // Round cache inputs without replacing the explicit runtime reserve.
    const auto other_cuda = ws_align(total_vram - free_vram, 64ULL * 1024 * 1024);
    opts.external_cuda_bytes = ws_add(opts.external_cuda_bytes, other_cuda);
    fprintf(stderr, "vn-wstream: other occupied CUDA=%llu external total=%llu\n", (unsigned long long) other_cuda, (unsigned long long) opts.external_cuda_bytes);
    auto plan = llama_weight_stream_cached_plan(opts,profile,hw,[&](const auto & p) { return llama_weight_stream_auto_plan(p,hw,opts,ggml_backend_dev_buffer_type(device)); });
    size_t tensors = profile.non_block_tensors.size();
    for (const auto & b : profile.blocks) { tensors += b.tensors.size(); }
    fprintf(stderr, "vn-wstream: architecture=%s blocks=%zu tensors=%zu candidates=%zu\nvn-wstream: gpu budget=%llu reserve=%llu effective=%llu\nvn-wstream: resident=%llu removed=%zu slots=%llu capacity=%zu expected net save=%llu managed=%llu\n",
        profile.architecture.c_str(),profile.blocks.size(),tensors,plan.candidate_count,(unsigned long long)opts.gpu_budget,(unsigned long long)opts.reserve,(unsigned long long)(opts.gpu_budget-opts.reserve),
        (unsigned long long)plan.full_resident_bytes,plan.pager.removed_weight_bytes,(unsigned long long)plan.slot_bytes,plan.pager.slot_capacity,(unsigned long long)plan.net_saving,(unsigned long long)plan.expected_bytes);
    std::string selected;
    uint64_t streamed_bytes = 0; size_t selected_tensors = 0;
    for (const auto & b : plan.pager.blocks) {
        if (!selected.empty()) { selected += ","; }
        selected += std::to_string(b.block_index);
        for (const auto & t : b.tensors) { streamed_bytes = ws_add(streamed_bytes,t.profile.stored_bytes); ++selected_tensors; }
    }
    fprintf(stderr, "vn-wstream: selected blocks=[%s] tensors=%zu streamed=%llu slot_count=%zu\n",selected.c_str(),selected_tensors,(unsigned long long)streamed_bytes,plan.pager.slot_count);
    llama_weight_stream_config config;
    config.embedded_mtp = params.load_mtp;
    config.mtp_weight_cuda_bytes = mtp_weight_bytes;
    config.mtp_runtime_cuda_bytes = mtp_runtime_bytes;
    config.stream = !plan.pager.blocks.empty(); config.prefetch = config.stream; config.cached_descriptors = true;
    config.auto_plan = plan; config.discovered = profile; config.slot_count = config.stream ? 2 : 0;
    for (const auto & b : plan.pager.blocks) { config.target_blocks.push_back(b.block_index); }
    if (config.stream) { config.target_block = config.target_blocks.front(); }
    llama_weight_stream_config_scope scope(config);
    auto * model = llama_model_load_from_file(path,params);
    if (!model) {
        // Descriptor/graph drift must not produce another cache HIT next run.
        if (opts.cache) { std::error_code ec; std::filesystem::remove(llama_weight_stream_profile_path(opts,id),ec); }
        throw std::runtime_error("auto streaming model load failed; profile invalidated");
    }
    fprintf(stderr, "vn-wstream: ready\n");
    return model;
}
