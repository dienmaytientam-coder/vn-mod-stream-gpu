#include "llama-model.h"
#include "llama-context.h"
#include "llama-ext.h"
#include "llama-weight-stream.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static void memory(llama_model * model, llama_context * ctx, const std::string & path) {
    std::ofstream out(path);
    out << "buffer\tmodel\tcontext\tcompute\n";
    for (const auto & item : ctx->memory_breakdown()) {
        out << ggml_backend_buft_name(item.first) << '\t' << item.second.model << '\t' << item.second.context << '\t' << item.second.compute << '\n';
    }
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(device, &free, &total);
    out << "device_total\t" << total << "\ndevice_free\t" << free << '\n';
    const auto * store = model->weight_stream->store();
    size_t slot_bytes = store ? store->slot_capacity() : 0;
    size_t pinned_bytes = store ? store->staging_capacity() : 0;
    if (const auto * pager = model->weight_stream->pager()) {
        for (const auto & slot : pager->slot_states()) { slot_bytes += slot.capacity; out << "slot_" << slot.id << '\t' << slot.capacity << '\n'; }
        for (const auto & block : pager->plan().blocks) { pinned_bytes += pager->block_store(block.block_index)->staging_capacity(); }
    }
    out << "stream_slot\t" << slot_bytes << "\npersistent_pinned\t" << pinned_bytes << '\n';
}

static void dump_profile(const llama_weight_stream_model_profile & profile, int64_t time_us, const std::string & root) {
    std::ofstream meta(root + "/profile-metadata.txt");
    meta << "architecture=" << profile.architecture << "\nschema_version=" << profile.schema_version << "\nprofile_version=" << profile.profile_version << "\nfingerprint=" << profile.model_fingerprint << "\nblocks=" << profile.blocks.size() << "\nprofile_us=" << time_us << '\n';
    std::ofstream out(root + "/profile.tsv");
    out << "block\tname\ttype\tne0\tne1\tne2\tne3\tnb0\tnb1\tnb2\tnb3\tstored_bytes\tallocation_bytes\talignment\tcapability\n";
    auto emit = [&](const llama_weight_stream_tensor_profile & tensor) {
        out << tensor.block_index << '\t' << tensor.name << '\t' << ggml_type_name(tensor.type);
        for (auto dimension : tensor.ne) { out << '\t' << dimension; }
        for (auto stride : tensor.nb) { out << '\t' << stride; }
        out << '\t' << tensor.stored_bytes << '\t' << tensor.allocation_bytes << '\t' << tensor.alignment << '\t' << int(tensor.capability) << '\n';
    };
    for (const auto & block : profile.blocks) { for (const auto & tensor : block.tensors) { emit(tensor); } }
    for (const auto & tensor : profile.non_block_tensors) { emit(tensor); }
}

static llama_weight_stream_model_profile run(const char * filename, const std::string & root, bool stream, const llama_weight_stream_model_profile & discovered, bool paging, bool prefetch = false, bool automatic = false, const std::string & reserve = "2048M", const std::string & budget = "10.2G", const std::string & cache_dir = "/datas/serverai/cache/llama-weight-stream/gate7-validation", uint32_t n_ctx = 8192, const std::string & cache_type = "f16") {
    const std::string label = stream ? "stream" : "baseline";
    llama_weight_stream_config config;
    config.stream = stream;
    config.prefetch = prefetch;
    config.target_block = 3;
    config.discovered = discovered;
    if (stream && paging) { config.target_blocks = {3,56,63}; config.slot_count = 2; }
    llama_weight_stream_config_scope scope(config);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.no_host = true;
    mp.load_mtp = false;
    mp.progress_callback = nullptr;
    auto cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    bool found_type = false;
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        const auto * name = ggml_type_name(ggml_type(type));
        if (name && cache_type == name) { cp.type_k = cp.type_v = ggml_type(type); found_type = true; break; }
    }
    check(found_type, "invalid cache type");
    if (ggml_is_quantized(cp.type_v)) { cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED; }
    cp.n_batch = 256;
    cp.n_ubatch = 128;
    cp.n_seq_max = 1;
    cp.n_threads = 16;
    cp.n_threads_batch = 16;
    cp.no_perf = false;
    llama_weight_stream_options options;
    options.mode = "auto";
    options.gpu_budget = llama_weight_stream_parse_size(budget);
    options.reserve = llama_weight_stream_parse_size(reserve,true);
    options.cache_dir = cache_dir;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        automatic && stream ? llama_weight_stream_load_auto(filename,mp,cp,options) : llama_model_load_from_file(filename,mp), llama_model_free);
    check(bool(model), "real model load failed");
    check(bool(model->weight_stream), "internal runtime is not connected to real model");
    if (automatic && stream) { paging = model->weight_stream->paging(); prefetch = paging; }
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    check(bool(ctx), "real context creation failed");
    memory(model.get(), ctx.get(), root + "/memory-" + label + ".tsv");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::string prompt = "<|im_start|>user\nVi\341\272\277t m\341\273\231t h\303\240m PHP nh\341\272\255n v\303\240o m\341\272\243ng s\341\273\221 nguy\303\252n v\303\240 tr\341\272\243 v\341\273\201 ph\341\272\247n t\341\273\255 l\341\273\233n nh\341\272\245t.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    std::vector<llama_token> input(prompt.size() + 16);
    int n_prompt = llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), false, true);
    check(n_prompt > 0 && n_prompt <= 128, "prompt tokenization failed");
    input.resize(n_prompt);
    std::ofstream logits(root + "/" + label + "-logits.f32", std::ios::binary);
    std::ofstream positions(root + "/" + label + "-positions.tsv");
    std::ofstream tokens(root + "/" + label + "-tokens.txt");
    std::ofstream text(root + "/" + label + "-output.txt");
    auto batch = llama_batch_init(256, 0, 1);
    batch.n_tokens = n_prompt;
    for (int i = 0; i < n_prompt; ++i) {
        batch.token[i] = input[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    std::ofstream evaluations(root + "/evaluations-" + label + ".csv");
    evaluations << "evaluation,cold,tokens,decode_us\n";
    auto disk_bytes = [] {
        std::ifstream in("/proc/self/io"); std::string key; uint64_t value;
        while (in >> key >> value) { if (key == "read_bytes:") { return value; } }
        return uint64_t(0);
    };
    const uint64_t disk_before = disk_bytes();
    const auto prompt_start = ggml_time_us();
    check(llama_decode(ctx.get(), batch) == 0, "prompt decode failed");
    evaluations << "1,1," << n_prompt << ',' << ggml_time_us() - prompt_start << '\n';
    for (int i = 0; i < n_prompt; ++i) {
        const auto * row = llama_get_logits_ith(ctx.get(), i);
        check(row != nullptr, "missing prompt logits");
        logits.write(reinterpret_cast<const char *>(row), n_vocab * sizeof(float));
        positions << "prompt\t" << i << '\n';
    }
    const auto generation_start = ggml_time_us();
    for (int i = 0; i < 128; ++i) {
        const auto * row = llama_get_logits_ith(ctx.get(), -1);
        check(row != nullptr, "missing generation logits");
        const llama_token token = std::max_element(row, row + n_vocab) - row;
        tokens << token << '\n';
        char piece[1024];
        int count = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        check(count >= 0, "token piece buffer too small");
        text.write(piece, count);
        batch.n_tokens = 1;
        batch.token[0] = token; batch.pos[0] = n_prompt + i; batch.logits[0] = 1;
        const auto evaluation_start = ggml_time_us();
        check(llama_decode(ctx.get(), batch) == 0, "generation decode failed");
        evaluations << i + 2 << ",0,1," << ggml_time_us() - evaluation_start << '\n';
        row = llama_get_logits_ith(ctx.get(), 0);
        check(row != nullptr, "missing decoded logits");
        logits.write(reinterpret_cast<const char *>(row), n_vocab * sizeof(float));
        positions << "decode\t" << n_prompt + i << '\n';
    }
    const auto generation_us = ggml_time_us() - generation_start;
    llama_batch_free(batch);
    llama_synchronize(ctx.get());
    const auto disk_delta = disk_bytes() - disk_before;
    std::ofstream(root + "/disk-" + label + ".txt") << "physical_read_bytes=" << disk_delta << '\n';
    if (stream) { check(disk_delta == 0, "physical disk reads during streaming inference"); }
    memory(model.get(), ctx.get(), root + "/memory-post-" + label + ".tsv");
    auto profile = model->weight_stream->profile(model->arch_name());
    if (!stream) { dump_profile(profile, model->weight_stream->profile_time_us(), root); }
    else {
        std::ofstream trace(root + "/backend-trace.tsv");
        trace << "evaluation\tname\top\tbackend\tblock\n";
        for (const auto & node : model->weight_stream->backend_trace()) {
            trace << node.evaluation << '\t' << node.name << '\t' << node.operation << '\t' << node.backend << '\t' << node.block << '\n';
        }
        const auto & stats = model->weight_stream->statistics();
        std::ofstream transport(root + "/transport.txt");
        transport << "evaluations=" << stats.evaluations << "\ncuda_nodes=" << stats.cuda_nodes << "\ncpu_nodes=" << stats.cpu_nodes << "\ncuda_transformer_nodes=" << stats.cuda_transformer_nodes << "\ncpu_transformer_nodes=" << stats.cpu_transformer_nodes << '\n';
        check(stats.cpu_transformer_nodes == 0 && stats.cuda_transformer_nodes > 0 && stats.cpu_nodes == 0 && stats.cuda_nodes > 0, "GPU contract failed");
        std::vector<const llama_weight_stream_transfer *> stores;
        if (paging) {
            const auto * pager = model->weight_stream->pager();
            check(pager != nullptr, "missing runtime pager");
            const auto & telemetry = pager->statistics();
            check(telemetry.evaluations == 129 && telemetry.loads == pager->plan().blocks.size() * telemetry.evaluations + (prefetch ? 2 : 0) && telemetry.reuses > 0 && telemetry.evictions > 0 && telemetry.verified_blocks == pager->plan().blocks.size() && telemetry.runtime_host_copy_bytes == 0, "pager runtime contract failed");
            transport << "loads=" << telemetry.loads << "\nreuses=" << telemetry.reuses << "\nevictions=" << telemetry.evictions << "\nhits=" << telemetry.hits << "\nmisses=" << telemetry.misses << "\nupload_bytes=" << telemetry.upload_bytes << "\nupload_us=" << telemetry.upload_us << "\ncompletion_us=" << telemetry.completion_us << "\ncompletion_waits=" << telemetry.completion_waits << "\nverified_blocks=" << telemetry.verified_blocks << "\ninitial_host_copy_bytes=" << telemetry.initial_host_copy_bytes << "\nruntime_host_copy_bytes=" << telemetry.runtime_host_copy_bytes << '\n';
            if (prefetch) {
                check(pager->prefetching() && telemetry.demand_misses == 0 && telemetry.prefetched_hits == pager->plan().blocks.size() * telemetry.evaluations, "prefetch coverage failed");
                const_cast<llama_weight_stream_pager *>(pager)->finish_timeline();
                std::ofstream transfers(root + "/transfers.csv");
                transfers << "launch_evaluation,demand,slot,block,generation,bytes,cold,consumed,launch_us,ready_us,needed_us,wait_us,start_ms,end_ms\n";
                transfers.precision(12);
                for (const auto & row : pager->transfer_history()) {
                    transfers << row.evaluation << ',' << row.demand << ',' << row.ticket.slot << ',' << row.ticket.block << ',' << row.ticket.generation << ',' << row.bytes << ',' << row.cold << ',' << row.consumed << ',' << row.launch_us << ',' << row.ready_us << ',' << row.needed_us << ',' << row.wait_us << ',' << row.start_ms << ',' << row.end_ms << '\n';
                }
                transport << "prefetched_hits=" << telemetry.prefetched_hits << "\ndemand_misses=" << telemetry.demand_misses << "\nexposed_wait_us=" << telemetry.exposed_wait_us << '\n';
            }
            std::ofstream plan(root + "/plan.tsv");
            plan << "block\ttensors\tpacked_bytes\tslot\n";
            for (size_t i = 0; i < pager->plan().blocks.size(); ++i) {
                const auto & block = pager->plan().blocks[i];
                plan << block.block_index << '\t' << block.tensors.size() << '\t' << block.slot_bytes << '\t' << i % pager->plan().slot_count << '\n';
                stores.push_back(pager->block_store(block.block_index));
            }
            std::ofstream slots(root + "/slots.tsv"), reuse(root + "/reuse.tsv");
            slots << "evaluation\tslot\tblock\tgeneration\tphase\taction\tbytes\n";
            reuse << "evaluation\tslot\tblock\tgeneration\tphase\taction\tbytes\n";
            for (const auto & event : pager->history()) {
                const std::string row = std::to_string(event.evaluation) + "\t" + std::to_string(event.slot) + "\t" + std::to_string(event.block) + "\t" + std::to_string(event.generation) + "\t" + std::to_string(int(event.phase)) + "\t" + event.action + "\t" + std::to_string(event.bytes) + "\n";
                slots << row;
                if (event.action == "EVICT" || event.action == "LOADING" || event.action == "BYTE_VERIFIED" || event.action == "LAST_CONSUMER_COMPLETE" || event.action == "REUSABLE") { reuse << row; }
            }
            std::ofstream final(root + "/slots-final.tsv");
            final << "slot\tblock\tgeneration\tloads\treuses\tevictions\tcapacity\n";
            for (const auto & slot : pager->slot_states()) { final << slot.id << '\t' << slot.block << '\t' << slot.generation << '\t' << slot.loads << '\t' << slot.reuses << '\t' << slot.evictions << '\t' << slot.capacity << '\n'; }
        } else {
            transport << "uploads=" << stats.uploads << "\nupload_bytes=" << stats.upload_bytes << "\nevent_waits=" << stats.event_waits << "\nupload_us=" << stats.upload_us << "\nwait_us=" << stats.wait_us << "\ninitial_host_copy_bytes=" << stats.initial_host_copy_bytes << "\nruntime_host_copy_bytes=" << stats.runtime_host_copy_bytes << "\nbyte_verified=" << stats.byte_verified << '\n';
            check(stats.evaluations == 129 && stats.uploads == stats.evaluations + 1 && stats.byte_verified && stats.runtime_host_copy_bytes == 0, "one-block transfer contract failed");
            stores.push_back(model->weight_stream->store());
        }
        std::ofstream residency(root + "/residency-stream.tsv");
        residency << "name\toriginal_buffer\tstream_buffer\thost_offset\tslot_offset\n";
        for (const auto * store : stores) {
            for (size_t i = 0; i < store->block().tensors.size(); ++i) {
                const auto & descriptor = store->block().tensors[i];
                const ggml_tensor * original = nullptr;
                for (const auto & tensor : model->tensors_by_name) { if (tensor.first == descriptor.profile.name) { original = tensor.second; } }
                check(original && ggml_backend_buffer_is_host(original->buffer), "selected original retained on CUDA");
                residency << descriptor.profile.name << '\t' << ggml_backend_buft_name(ggml_backend_buffer_get_type(original->buffer)) << '\t' << ggml_backend_buft_name(ggml_backend_buffer_get_type(store->tensor(i)->buffer)) << '\t' << descriptor.offset << '\t' << descriptor.offset << '\n';
            }
        }
    }
    const auto perf = llama_perf_context(ctx.get());
    std::ofstream timing(root + "/performance-" + label + ".txt");
    timing << "seed=424242\ntemperature=0\nprompt_tokens=" << n_prompt << "\nvocab=" << n_vocab << "\npositions=" << n_prompt + 128 << "\ngenerated=128\nwall_generation_us=" << generation_us << "\nwall_tg=" << 128e6 / generation_us << "\nt_eval_ms=" << perf.t_eval_ms << "\nn_eval=" << perf.n_eval << "\ntg=" << 1000 * perf.n_eval / perf.t_eval_ms << '\n';
    printf("%s: 128 tokens, %.3f wall tok/s, profile tensors=%zu\n", label.c_str(), 128e6 / generation_us, model->tensors_by_name.size());
    return profile;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 9) { fprintf(stderr, "usage: %s MODEL ARTIFACT_DIRECTORY [--pager|--prefetch|--auto|--auto-only|--auto-target|--auto-context] [RESERVE] [BUDGET] [CACHE_DIRECTORY] [CONTEXT] [CACHE_TYPE]\n", argv[0]); return 2; }
    const std::string mode = argc >= 4 ? argv[3] : "";
    const bool automatic = mode == "--auto" || mode == "--auto-only" || mode == "--auto-target" || mode == "--auto-context";
    const bool prefetch = mode == "--prefetch" || automatic;
    const bool paging = mode == "--pager" || prefetch;
    const std::string reserve = argc >= 5 ? argv[4] : "2048M";
    const std::string budget = argc >= 6 ? argv[5] : "10.2G";
    const std::string cache_dir = argc >= 7 ? argv[6] : mode == "--auto-target" ? "/datas/serverai/cache/llama-weight-stream" : "/datas/serverai/cache/llama-weight-stream/gate7-validation";
    if (!mode.empty() && !paging) { return 2; }
    try {
        ggml_backend_load_all();
        llama_backend_init();
        size_t end = 0;
        const uint64_t n_ctx = argc >= 8 ? std::stoull(argv[7], &end) : 8192;
        check(n_ctx > 0 && n_ctx <= UINT32_MAX && (argc < 8 || end == std::string(argv[7]).size()), "invalid context size");
        const std::string cache_type = argc >= 9 ? argv[8] : "f16";
        llama_weight_stream_model_profile profile;
        if (mode != "--auto-only" && mode != "--auto-target" && mode != "--auto-context") { profile = run(argv[1], argv[2], false, {}, paging, prefetch); }
        if (!automatic) {
            const auto plan = llama_weight_stream_plan(profile, 3);
            size_t stream_count = 0, bypass_count = 0;
            for (const auto & tensor : plan.tensors) {
                if (tensor.capability == llama_weight_stream_capability::STREAMABLE) { ++stream_count; }
                if (tensor.capability == llama_weight_stream_capability::MUST_RESIDENT) { ++bypass_count; }
            }
            check(stream_count == 7 && plan.streamable_bytes == 115056640 && bypass_count == 4 && plan.resident_bytes == 43008 && plan.unknown_bytes == 0, "runtime blk.3 profile does not match fixture");
        }
        run(argv[1], argv[2], true, profile, paging, prefetch, automatic, reserve, budget, cache_dir, n_ctx, cache_type);
        llama_backend_free();
    } catch (const std::exception & e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
    return 0;
}
