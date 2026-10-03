#include "llama-weight-stream.h"
#include "ggml-cpu.h"
#include "arg.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

using capability = llama_weight_stream_capability;

static void check(bool condition) {
    if (!condition) {
        throw std::runtime_error("contract assertion failed");
    }
}

template<typename F>
static void rejects(F operation) {
    try {
        operation();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("invalid descriptor was accepted");
}

static llama_weight_stream_tensor_profile profile(const std::string & name, int block, int64_t rows, capability cap = capability::STREAMABLE) {
    llama_weight_stream_tensor_profile result;
    result.name = name;
    result.block_index = block;
    result.ne = {8, rows, 1, 1};
    result.nb = {4, 32, size_t(32 * rows), size_t(32 * rows)};
    result.logical_bytes = size_t(32 * rows);
    result.stored_bytes = cap == capability::STATE_NOT_WEIGHT ? 0 : result.logical_bytes;
    result.capability = cap;
    return result;
}


static void gate7_contract(const llama_weight_stream_model_profile & fixture, ggml_backend_buffer_type_t buft) {
    check(llama_weight_stream_parse_size("2048") == 2048ULL * 1024 * 1024);
    check(llama_weight_stream_parse_size("2048M") == llama_weight_stream_parse_size("2G"));
    check(llama_weight_stream_parse_size("11.4G") == 12240656793ULL);
    for (const auto * value : {"", "-1", "1x", "1.2", "0", "18446744073709551616G", "1e3G"}) {
        rejects([&] { llama_weight_stream_parse_size(value); });
    }
    auto model = fixture;
    model.blocks.clear();
    for (int i = 0; i < 12; ++i) { model.blocks.push_back(llama_weight_stream_profile_block(i, {profile("weight" + std::to_string(i), i, 40)})); model.demand_order.push_back(i); }
    for (auto & block : model.blocks) { for (auto & t : block.tensors) {
        t.cuda_weight = true; t.alignment = ggml_backend_buft_get_alignment(buft); t.allocation_bytes = t.stored_bytes;
    } }
    llama_weight_stream_options opts;
    opts.mode = "auto"; opts.gpu_budget = 14000; opts.reserve = 0;
    llama_weight_stream_hardware hw;
    hw.identity = "synthetic-GPU-A"; hw.compute_capability = 860; hw.total_vram = 32768;
    auto plan = llama_weight_stream_auto_plan(model, hw, opts, buft);
    check(llama_weight_stream_plan_json(plan) == llama_weight_stream_plan_json(llama_weight_stream_auto_plan(model, hw, opts, buft)));
    check(plan.expected_bytes <= opts.gpu_budget && plan.net_saving > 0 && plan.pager.slot_count == 2);
    check(plan.slot_bytes == 2 * plan.pager.slot_capacity);
    check(plan.pager.blocks.size() == 4);
    auto runtime_options = opts;
    runtime_options.reserve = 100;
    runtime_options.runtime_cuda_bytes = 3000;
    runtime_options.external_cuda_bytes = 2000;
    runtime_options.runtime_key = "context-a";
    auto runtime_plan = llama_weight_stream_auto_plan(model, hw, runtime_options, buft);
    check(runtime_plan.pager.blocks.size() > plan.pager.blocks.size());
    check(runtime_plan.extra_cuda_bytes == 5000);
    check(runtime_plan.expected_bytes + runtime_options.reserve <= runtime_options.gpu_budget);
    check(runtime_plan.net_saving == runtime_plan.pager.removed_weight_bytes - runtime_plan.slot_bytes);
    check(llama_weight_stream_plan_json(llama_weight_stream_plan_from_json(llama_weight_stream_plan_json(runtime_plan))) == llama_weight_stream_plan_json(runtime_plan));
    auto another_context = runtime_options;
    another_context.runtime_key = "context-b";
    check(llama_weight_stream_plan_key(model, hw, runtime_options) != llama_weight_stream_plan_key(model, hw, another_context));
    runtime_options.runtime_cuda_bytes = runtime_options.gpu_budget;
    rejects([&] { llama_weight_stream_auto_plan(model, hw, runtime_options, buft); });
    check(llama_weight_stream_plan_json(llama_weight_stream_plan_from_json(llama_weight_stream_plan_json(plan))) == llama_weight_stream_plan_json(plan));
    auto protected_model = model;
    protected_model.blocks[0].tensors[0].capability = capability::UNKNOWN;
    protected_model.blocks[1].tensors[0].capability = capability::MUST_RESIDENT;
    for (int i : {0,1}) { auto & b = protected_model.blocks[i]; b = llama_weight_stream_profile_block(b.block_index,b.tensors); }
    const auto protected_plan = llama_weight_stream_auto_plan(protected_model,hw,opts,buft);
    for (const auto & b : protected_plan.pager.blocks) { check(b.block_index != 0 && b.block_index != 1); }
    auto ample = opts; ample.gpu_budget = 20000;
    check(llama_weight_stream_auto_plan(model,hw,ample,buft).pager.blocks.empty());
    opts.gpu_budget = 1;
    rejects([&] { llama_weight_stream_auto_plan(model, hw, opts, buft); });
    opts.gpu_budget = 14000;
    const auto root = std::filesystem::temp_directory_path() / ("vn-wstream-test-" + std::to_string(ggml_time_us()));
    std::filesystem::create_directories(root);
    const auto file = root / "model.gguf";
    { std::ofstream out(file); out << "synthetic"; }
    opts.cache_dir = (root / "cache").string();
    auto count_fds = [] {
        size_t count = 0;
#ifndef _WIN32
        for (const auto & entry : std::filesystem::directory_iterator("/proc/self/fd")) { (void) entry; ++count; }
#endif
        return count;
    };
    const size_t initial_fds = count_fds();
    auto id = llama_weight_stream_identify(file.string());
    const auto serialized = llama_weight_stream_profile_json(model, id);
    llama_weight_stream_model_identity loaded_id;
    check(llama_weight_stream_profile_json(llama_weight_stream_profile_from_json(serialized, loaded_id), loaded_id) == serialized);
    int scans = 0, planners = 0;
    auto scan = [&] { ++scans; return model; };
    auto planner = [&](const auto & p) { ++planners; return llama_weight_stream_auto_plan(p, hw, opts, buft); };
    auto first = llama_weight_stream_cached_profile(opts, id, scan);
    auto first_plan = llama_weight_stream_cached_plan(opts, first, hw, planner);
    auto second = llama_weight_stream_cached_profile(opts, id, scan);
    llama_weight_stream_cached_plan(opts, second, hw, planner);
    check(scans == 1 && planners == 1);
    opts.reserve = 1;
    llama_weight_stream_cached_profile(opts, id, scan);
    llama_weight_stream_cached_plan(opts, second, hw, planner);
    check(scans == 1 && planners == 2);
    hw.identity = "synthetic-GPU-B";
    llama_weight_stream_cached_plan(opts, second, hw, planner);
    check(scans == 1 && planners == 3);
    auto stale = id; ++stale.file_size;
    llama_weight_stream_cached_profile(opts, stale, scan);
    check(scans == 2);
    stale.mtime_ns++;
    llama_weight_stream_cached_profile(opts, stale, scan);
    check(scans == 3);
    for (const auto * corrupt : {"{", "{}", "{\"schema_version\":99}", "null"}) {
        std::ofstream(llama_weight_stream_profile_path(opts, id)) << corrupt;
        llama_weight_stream_cached_profile(opts, id, scan);
    }
    check(scans == 7);
    auto corrupt_profile = nlohmann::json::parse(serialized);
    for (int i = 0; i < 5; ++i) {
        auto corrupt = corrupt_profile;
        if (i == 0) { corrupt["schema_version"] = 99; }
        if (i == 1) { corrupt["profile_version"] = 99; }
        if (i == 2) { corrupt["blocks"][0]["tensors"][0].erase("name"); }
        if (i == 3) { corrupt["blocks"][0]["tensors"][0]["type"] = 999; }
        if (i == 4) { corrupt["blocks"][0]["tensors"][0]["stored_bytes"] = -1; }
        std::ofstream(llama_weight_stream_profile_path(opts,id)) << corrupt.dump();
        llama_weight_stream_cached_profile(opts,id,scan);
    }
    check(scans == 12);
    const auto key = llama_weight_stream_plan_key(model,hw,opts);
    std::ofstream(root / "cache" / "plans" / (key + ".json")) << "{truncated";
    llama_weight_stream_cached_plan(opts,model,hw,planner);
    check(planners == 4);
    for (const auto & entry : std::filesystem::recursive_directory_iterator(root)) { check(entry.path().string().find(".tmp.") == std::string::npos); }
    auto bad_directory = opts; bad_directory.cache_dir = file.string();
    bool write_failed = false;
    try { llama_weight_stream_cached_profile(bad_directory,id,scan); } catch (const std::exception &) { write_failed = true; }
    check(write_failed);
    check(count_fds() == initial_fds);
    puts("PASS cache failure/recovery: no file descriptor or temporary file leaks");
    llama_weight_stream_config runtime_config;
    runtime_config.stream = true; runtime_config.cached_descriptors = true;
    runtime_config.discovered = model; runtime_config.auto_plan = plan; runtime_config.slot_count = 2;
    for (const auto & b : plan.pager.blocks) { runtime_config.target_blocks.push_back(b.block_index); }
    llama_weight_stream_runtime runtime(runtime_config);
    const auto descriptor = plan.pager.blocks.front().tensors.front().profile;
    ggml_tensor materialized = {};
    materialized.type = descriptor.type;
    std::copy(descriptor.ne.begin(),descriptor.ne.end(),materialized.ne);
    std::copy(descriptor.nb.begin(),descriptor.nb.end(),materialized.nb);
    ggml_set_name(&materialized,descriptor.name.c_str());
    runtime.register_tensor(&materialized,descriptor.block_index);
    auto wrong_tensor = materialized;
    wrong_tensor.ne[1]++;
    wrong_tensor.nb[2] += wrong_tensor.nb[1]; wrong_tensor.nb[3] = wrong_tensor.nb[2];
    rejects([&] { runtime.register_tensor(&wrong_tensor,descriptor.block_index); });
    auto actual = model.blocks.front().tensors.front();
    llama_weight_stream_validate_descriptor(actual, actual);
    auto wrong = actual; wrong.name += "wrong";
    rejects([&] { llama_weight_stream_validate_descriptor(actual, wrong); });
    common_params params;
    check(params.vn_wstream_mode == "off");
    const auto config = root / "config.ini";
    std::ofstream(config) << "[vn-mod]\ngpu = 11.4G\n[vn-mod-gpu-wstream]\nmode = auto\nreserve = 2048M\ncache = false\ncache_dir = " << opts.cache_dir << "\n";
    common_params_apply_vn_config(params, config.string());
    check(params.vn_wstream_mode == "auto" && params.vn_wstream_reserve == 2147483648ULL && !params.vn_wstream_cache);
    char bin[] = "test-weight-stream", flag[] = "--vn-mod-gpu-wstream-reserve", value[] = "1024M";
    char model_flag[] = "--model"; std::string model_path = file.string();
    char * arguments[] = {bin,flag,value,model_flag,model_path.data()};
    check(common_params_parse(5,arguments,params,LLAMA_EXAMPLE_CLI));
    check(params.vn_wstream_mode == "auto" && params.vn_wstream_reserve == 1073741824ULL);
    std::filesystem::remove_all(root);
    puts("PASS Gate 7 size/config/profile/cache/planner/plan/budget/descriptor contract");
}

int main() {
    try {
        static_assert(std::is_same<decltype(llama_weight_stream_tensor_profile::ne), std::array<int64_t, GGML_MAX_DIMS>>::value, "value dimensions");
        static_assert(std::is_same<decltype(llama_weight_stream_model_profile::blocks), std::vector<llama_weight_stream_block_profile>>::value, "dynamic profile data");
        const auto buft = ggml_backend_cpu_buffer_type();
        rejects([&] { llama_weight_stream_profile_block(0, {}); });
        rejects([&] { llama_weight_stream_validate_profile({}); });
        auto invalid = profile("a", 3, 2);
        invalid.logical_bytes = 0;
        rejects([&] { llama_weight_stream_profile_block(3, {invalid}); });
        invalid = profile("a", 3, 2);
        invalid.stored_bytes -= 1;
        rejects([&] { llama_weight_stream_profile_block(3, {invalid}); });
        invalid = profile("a", 3, 2);
        invalid.ne[0] = 0;
        rejects([&] { llama_weight_stream_profile_block(3, {invalid}); });
        invalid = profile("a", 3, 2);
        invalid.nb[1] += 1;
        rejects([&] { llama_weight_stream_profile_block(3, {invalid}); });
        puts("PASS empty, zero size, invalid shape/span/stride rejection");

        std::vector<llama_weight_stream_tensor_profile> tensors = {profile("b", 3, 3), profile("a", 3, 2), profile("c", 3, 5)};
        const auto block = llama_weight_stream_profile_block(3, tensors);
        const auto packed = llama_weight_stream_pack(block, buft);
        llama_weight_stream_validate(packed, packed.slot_bytes);
        std::reverse(tensors.begin(), tensors.end());
        const auto reverse = llama_weight_stream_pack(llama_weight_stream_profile_block(3, tensors), buft);
        check(packed.slot_bytes == reverse.slot_bytes);
        for (size_t i = 0; i < packed.tensors.size(); ++i) {
            check(packed.tensors[i].profile.name == reverse.tensors[i].profile.name);
            check(packed.tensors[i].offset == reverse.tensors[i].offset);
            check(packed.tensors[i].offset % packed.alignment == 0);
            check(packed.tensors[i].profile.ne == reverse.tensors[i].profile.ne);
            check(packed.tensors[i].profile.nb == reverse.tensors[i].profile.nb);
        }
        puts("PASS deterministic layout, alignment, preserved metadata without runtime pointers");

        auto bad = packed;
        bad.tensors[1].profile.name = bad.tensors[0].profile.name;
        rejects([&] { llama_weight_stream_validate(bad, bad.slot_bytes); });
        bad = packed;
        bad.tensors[1].offset = bad.tensors[0].offset;
        rejects([&] { llama_weight_stream_validate(bad, bad.slot_bytes); });
        bad = packed;
        bad.tensors[1].offset += 1;
        rejects([&] { llama_weight_stream_validate(bad, bad.slot_bytes); });
        rejects([&] { llama_weight_stream_validate(packed, packed.slot_bytes - 1); });
        bad = packed;
        bad.tensors[0].offset = SIZE_MAX - SIZE_MAX % packed.alignment;
        rejects([&] { llama_weight_stream_validate(bad, SIZE_MAX); });
        bad = packed;
        bad.alignment = 0;
        rejects([&] { llama_weight_stream_validate(bad, bad.slot_bytes); });
        invalid = profile("overflow", 3, 2);
        invalid.ne[1] = INT64_MAX;
        rejects([&] { llama_weight_stream_profile_block(3, {invalid}); });
        puts("PASS duplicate, overlap, capacity and overflow rejection");

        check(llama_weight_stream_tensor_profile{}.capability == capability::UNKNOWN);
        auto mixed = llama_weight_stream_profile_block(999, {
            profile("stream", 999, 2), profile("resident", 999, 3, capability::MUST_RESIDENT),
            profile("unknown", 999, 5, capability::UNKNOWN), profile("state", 999, 7, capability::STATE_NOT_WEIGHT)});
        check(mixed.streamable_bytes == 64 && mixed.resident_bytes == 96 && mixed.unknown_bytes == 160);
        const auto safe = llama_weight_stream_pack(mixed, buft);
        check(safe.block_index == 999 && safe.tensors.size() == 1 && safe.tensors[0].profile.name == "stream");
        check(mixed.tensors[2].capability == capability::UNKNOWN);
        rejects([&] { llama_weight_stream_pack(llama_weight_stream_profile_block(9, {profile("u", 9, 1, capability::UNKNOWN)}), buft); });
        bad = packed;
        bad.tensors[0].profile.capability = capability::UNKNOWN;
        rejects([&] { llama_weight_stream_validate(bad, bad.slot_bytes); });
        puts("PASS dynamic block 999 and UNKNOWN/state/resident fail-safe exclusion");

        llama_weight_stream_model_profile model;
        model.architecture = "synthetic-test";
        model.model_fingerprint = "test-fingerprint";
        model.blocks.push_back(llama_weight_stream_profile_block(0, {profile("m0", 0, 1)}));
        model.blocks.push_back(llama_weight_stream_profile_block(11, {profile("m1a", 11, 2), profile("m1b", 11, 3)}));
        model.blocks.push_back(llama_weight_stream_profile_block(999, {profile("m2a", 999, 4), profile("m2b", 999, 5), profile("m2c", 999, 6)}));
        model.non_block_tensors.push_back(profile("output", -1, 2, capability::MUST_RESIDENT));
        llama_weight_stream_validate_profile(model);
        const auto copy = model;
        check(copy.blocks.size() == 3 && copy.non_block_tensors.size() == 1 && copy.schema_version == 1);
        for (size_t i = 0; i < model.blocks.size(); ++i) {
            const auto layout = llama_weight_stream_pack(model.blocks[i], buft);
            check(layout.tensors.size() == i + 1);
        }
        auto bad_model = model;
        bad_model.blocks.push_back(model.blocks[0]);
        rejects([&] { llama_weight_stream_validate_profile(bad_model); });
        puts("PASS synthetic non-Qwen model: dynamic blocks 0/11/999, 1/2/3 tensors and version/fingerprint data");
        gate7_contract(model, buft);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
