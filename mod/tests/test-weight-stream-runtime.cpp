// SPDX-License-Identifier: MIT
// Copyright (c) 2026 dienmaytientam-coder

#include "llama-weight-stream.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <stdexcept>

static void check(bool ok) {
    if (!ok) { throw std::runtime_error("runtime contract failed"); }
}

template<class F> static void rejects(F f) {
    try { f(); } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("unsafe path accepted");
}

int main() {
    try {
        ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
        auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 8);
        auto * bypass = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
        auto * unknown = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 8);
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 1);
        ggml_set_name(w, "arbitrary_projection");
        ggml_set_name(bypass, "arbitrary_norm");
        ggml_set_name(unknown, "unobserved");
        llama_weight_stream_config cfg;
        llama_weight_stream_runtime runtime(cfg);
        runtime.register_tensor(w, 917);
        runtime.register_tensor(bypass, 917);
        runtime.register_tensor(unknown, 917);
        runtime.begin_graph();
        runtime.observe_projection(w);
        auto * mm = ggml_mul_mat(ctx.get(), w, input);
        auto * out = ggml_mul(ctx.get(), mm, bypass);
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, out);
        runtime.finish_graph(graph);
        auto profile = runtime.profile("fixture");
        check(profile.blocks.size() == 1 && profile.blocks[0].block_index == 917);
        check(profile.blocks[0].streamable_bytes == 256);
        check(profile.blocks[0].resident_bytes == 32 && profile.blocks[0].unknown_bytes == 256);
        auto plan = llama_weight_stream_plan(profile, 917);
        check(plan.tensors.size() == 3 && plan.streamable_bytes == 256);
        rejects([&] { llama_weight_stream_plan(profile, 3); });
        check(llama_weight_stream_resolve(false, false, w, unknown) == w);
        check(llama_weight_stream_resolve(true, true, w, nullptr) == w);
        check(llama_weight_stream_resolve(true, true, w, unknown) == unknown);
        rejects([&] { llama_weight_stream_resolve(true, false, w, unknown); });
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        rejects([&] { llama_weight_stream_require_cuda(cpu.get()); });
        llama_weight_stream_config resident_config;
        resident_config.cached_descriptors = true;
        resident_config.auto_plan.plan_key = "resident-mtp-fixture";
        resident_config.auto_plan.options.runtime_cuda_bytes = 900;
        resident_config.mtp_runtime_cuda_bytes = 300;
        llama_weight_stream_runtime resident(resident_config);
        resident.check_context_memory(600);
        resident.check_context_memory(300, true);
        for (bool draft : {false, true}) {
            bool rejected = false;
            try { resident.check_context_memory(draft ? 301 : 601, draft); }
            catch (const std::runtime_error &) { rejected = true; }
            check(rejected);
        }
        resident.begin_graph();
        resident.tag_node(out, 918);
        resident.finish_graph(graph);
        ggml_backend_t backends[] = {cpu.get()};
        ggml_backend_sched_ptr scheduler(ggml_backend_sched_new(backends, nullptr, 1, 64, false, true));
        ggml_backend_sched_set_tensor_backend(scheduler.get(), mm, cpu.get());
        bool cpu_rejected = false;
        try { resident.check_backends(graph, scheduler.get()); }
        catch (const std::runtime_error &) { cpu_rejected = true; }
        check(cpu_rejected && resident.statistics().cpu_transformer_nodes == 1);
        check(runtime.profile("fixture").blocks[0].streamable_bytes == 256);
        puts("PASS resident draft graph isolation, separate runtime budgets and CPU transformer rejection");
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (device) {
            ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
            llama_weight_stream_require_cuda(gpu.get());
            auto layout = llama_weight_stream_pack(plan, ggml_backend_get_default_buffer_type(gpu.get()));
            std::vector<float> source(64, 1.25f);
            llama_weight_stream_transfer store(gpu.get(), layout, {{"arbitrary_projection", source.data(), source.size() * sizeof(float)}});
            store.persist();
            source.clear();
            source.shrink_to_fit();
            for (int i = 0; i < 3; ++i) { store.upload(); store.synchronize(); store.verify(); }
            check(store.pinned());
            check(static_cast<const float *>(store.staging_data())[0] == 1.25f);
            try { store.stage(); throw std::runtime_error("persistent store accepted mutation"); }
            catch (const std::logic_error &) {}
            ggml_backend_buffer_ptr original(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(gpu.get()), 256));
            check(ggml_backend_tensor_alloc(original.get(), w, ggml_backend_buffer_get_base(original.get())) == GGML_STATUS_SUCCESS);
            llama_weight_stream_config stream_config;
            stream_config.stream = true;
            stream_config.target_block = 917;
            stream_config.discovered = profile;
            llama_weight_stream_runtime duplicate(stream_config);
            duplicate.register_tensor(w, 917);
            duplicate.register_tensor(bypass, 917);
            duplicate.register_tensor(unknown, 917);
            bool rejected = false;
            try { duplicate.initialize(device, "fixture"); } catch (const std::runtime_error &) { rejected = true; }
            check(rejected);
            puts("PASS persistent pinned ownership after source release, repeated packed uploads and duplicate CUDA original rejection");
        } else {
            puts("SKIP CUDA lifetime test: no GPU backend");
        }
        runtime.begin_graph();
        runtime.observe_projection(w);
        auto * unsafe = ggml_add(ctx.get(), w, unknown);
        auto * graph2 = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph2, unsafe);
        runtime.finish_graph(graph2);
        check(runtime.profile("fixture").blocks[0].streamable_bytes == 0);
        puts("PASS runtime metadata/profile, generic one-block plan, unknown resident, resolver and CPU rejection");
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
