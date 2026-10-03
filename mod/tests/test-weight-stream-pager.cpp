// SPDX-License-Identifier: MIT
// Copyright (c) 2026 dienmaytientam-coder

#include "llama-weight-stream.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <stdexcept>

static void check(bool ok) {
    if (!ok) { throw std::runtime_error("pager contract assertion failed"); }
}
template<class F> static void rejects(F f) {
    try { f(); } catch (const std::invalid_argument &) { return; }
    throw std::runtime_error("unsafe pager action accepted");
}

static void test_prefetch(ggml_backend_t gpu, const llama_weight_stream_model_profile & profile) {
    ggml_backend_ptr compute(ggml_backend_dev_init(ggml_backend_get_device(gpu), nullptr));
    llama_weight_stream_demand_queue queue({91,917,1337});
    check(queue.next({91,917}) == 1337);
    queue.consume(91); queue.consume(917); queue.consume(1337);
    check(queue.position() == 3 && queue.next({}) == 91);
    rejects([&] { queue.consume(917); });
    auto plan = llama_weight_stream_make_pager_plan(profile, {91,917,1337}, 2, ggml_backend_get_default_buffer_type(gpu));
    std::vector<std::vector<float>> values;
    std::vector<std::vector<llama_weight_stream_source>> sources;
    for (const auto & block : plan.blocks) {
        values.emplace_back(block.tensors[0].profile.stored_bytes/4, float(block.block_index));
        sources.push_back({{block.tensors[0].profile.name,values.back().data(),values.back().size()*4}});
    }
    for (int repetition = 0; repetition < 6; ++repetition) {
        llama_weight_stream_pager pager(gpu, plan, sources, true);
        for (int evaluation = 0; evaluation < 12; ++evaluation) {
            pager.begin_evaluation();
            for (const auto & block : plan.blocks) {
                auto ticket = pager.enter(block.block_index, compute.get());
                rejects([&] { pager.slot_states().at(ticket.slot).validate(ticket.block, ticket.generation+1); });
                auto * tensor = pager.checked_tensor(ticket,block.tensors[0].profile.name);
                std::vector<float> read(block.tensors[0].profile.stored_bytes/4);
                ggml_backend_tensor_get(tensor,read.data(),0,read.size()*4);
                check(read.front()==float(block.block_index) && read.back()==float(block.block_index));
                pager.complete(ticket,compute.get());
                const auto & pending = pager.pending_prefetches().at(ticket.slot);
                check(pending.active && pending.ticket.generation==ticket.generation+1);
                check(pager.slot_states().at(ticket.slot).phase==llama_weight_stream_slot_phase::LOADING);
                rejects([&] { pager.validate_pending({ticket.slot,pending.ticket.block,pending.ticket.generation-1}); });
            }
        }
        check(pager.statistics().loads==38 && pager.statistics().demand_misses==0);
        check(pager.statistics().prefetched_hits==36);
        if (repetition % 2 == 0) { pager.finish_timeline(); }
        check(pager.transfer_history().size()==38);
        puts("PASS cyclic async LOADING, pending generation, no duplicate, required-only wait, rapid next evaluation and pending destruction");
    }
}

static void test_pending_lifetime(ggml_backend_t gpu) {
    ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
    llama_weight_stream_model_profile profile;
    profile.architecture = "pending-lifetime";
    std::vector<std::vector<unsigned char>> values;
    std::vector<std::vector<llama_weight_stream_source>> sources;
    for (int i = 0; i < 3; ++i) {
        auto * tensor = ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I8,8*1024*1024);
        const std::string name = "pending" + std::to_string(i);
        ggml_set_name(tensor,name.c_str());
        profile.blocks.push_back(llama_weight_stream_profile_block(i,{llama_weight_stream_describe(*tensor,i,llama_weight_stream_capability::STREAMABLE)}));
        values.emplace_back(8*1024*1024,(unsigned char)i);
        sources.push_back({{name,values.back().data(),values.back().size()}});
    }
    auto plan = llama_weight_stream_make_pager_plan(profile,{0,1,2},2,ggml_backend_get_default_buffer_type(gpu));
    ggml_backend_ptr compute(ggml_backend_dev_init(ggml_backend_get_device(gpu),nullptr));
    for (int repetition = 0; repetition < 12; ++repetition) {
        bool early_error = false;
        try {
            llama_weight_stream_pager pager(gpu,plan,sources,true);
            pager.begin_evaluation();
            auto ticket = pager.enter(0,compute.get());
            rejects([&] { pager.enter(0,compute.get()); });
            pager.complete(ticket,compute.get());
            check(pager.pending_prefetches()[0].active && pager.block_store(2)->transfer_pending());
            if (repetition % 2) { pager.enter(2,compute.get()); }
        } catch (const std::invalid_argument &) { early_error = true; }
        check(early_error == bool(repetition % 2));
    }
    puts("PASS large queued H2D destruction, early error, no finish_timeline drain and compute-context lifetime");
}

int main() {
    try {
        ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, true}));
        llama_weight_stream_model_profile profile;
        profile.architecture = "generic-test";
        for (int i = 0; i < 3; ++i) {
            auto * w = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 8 + 2 * i);
            ggml_set_name(w, ("weight" + std::to_string(i)).c_str());
            const int block = std::vector<int>{91, 917, 1337}[i];
            profile.blocks.push_back(llama_weight_stream_profile_block(block, {llama_weight_stream_describe(*w, block, llama_weight_stream_capability::STREAMABLE)}));
        }
        auto plan = llama_weight_stream_make_pager_plan(profile, {91,917,1337}, 2, ggml_backend_cpu_buffer_type());
        check(plan.blocks.size() == 3 && plan.slot_count == 2 && plan.slot_capacity == 384);
        rejects([&] { llama_weight_stream_make_pager_plan(profile, {91,91,1337}, 2, ggml_backend_cpu_buffer_type()); });
        rejects([&] { llama_weight_stream_make_pager_plan(profile, {91,917,9}, 2, ggml_backend_cpu_buffer_type()); });
        rejects([&] { llama_weight_stream_make_pager_plan(profile, {91,917,1337}, 0, ggml_backend_cpu_buffer_type()); });
        rejects([&] { llama_weight_stream_make_pager_plan(profile, {91,917,1337}, 2, ggml_backend_cpu_buffer_type(), 256); });
        auto unknown = profile;
        unknown.blocks[2].tensors[0].capability = llama_weight_stream_capability::UNKNOWN;
        unknown.blocks[2] = llama_weight_stream_profile_block(1337, unknown.blocks[2].tensors);
        rejects([&] { llama_weight_stream_make_pager_plan(unknown, {91,917,1337}, 2, ggml_backend_cpu_buffer_type()); });
        auto bad = plan;
        bad.blocks[0].tensors[0].profile.capability = llama_weight_stream_capability::UNKNOWN;
        rejects([&] { llama_weight_stream_validate_pager_plan(bad); });
        llama_weight_stream_slot_state slot;
        slot.id = 0; slot.capacity = 512;
        rejects([&] { slot.finish_load(); });
        slot.begin_load(91, 256);
        check(slot.phase == llama_weight_stream_slot_phase::LOADING && slot.generation == 1);
        rejects([&] { slot.begin_load(917, 320); });
        slot.finish_load(); slot.begin_use(91, 1);
        rejects([&] { slot.begin_load(1337, 384); });
        rejects([&] { slot.complete(91, 1, false); });
        slot.complete(91, 1, true);
        slot.begin_load(1337, 384); slot.finish_load();
        check(slot.generation == 2 && slot.reuses == 1 && slot.evictions == 1);
        rejects([&] { slot.validate(91, 1); });
        rejects([&] { slot.validate(1337, 1); });
        slot.validate(1337, 2); slot.begin_use(1337, 2); slot.complete(1337, 2, true);
        rejects([&] { slot.begin_load(91, 513); });
        puts("PASS generic three-block/two-slot plan, capacity, UNKNOWN, duplicates, states, generation and illegal overwrite rejection");

        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (device) {
            ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
            auto cuda_plan = llama_weight_stream_make_pager_plan(profile, {91,917,1337}, 2, ggml_backend_get_default_buffer_type(gpu.get()));
            std::vector<std::vector<float>> data;
            std::vector<std::vector<llama_weight_stream_source>> sources;
            for (const auto & block : cuda_plan.blocks) {
                data.emplace_back(block.tensors[0].profile.stored_bytes/4, float(block.block_index));
                sources.push_back({{block.tensors[0].profile.name,data.back().data(),data.back().size()*4}});
            }
            llama_weight_stream_pager pager(gpu.get(), cuda_plan, sources);
            data.clear(); data.shrink_to_fit();
            check(pager.block_store(91)->tensor(0)->buffer == pager.block_store(1337)->tensor(0)->buffer);
            check(pager.block_store(91)->tensor(0)->buffer != pager.block_store(917)->tensor(0)->buffer);
            check(pager.block_store(91)->slot_capacity() == cuda_plan.slot_capacity);
            llama_weight_stream_ticket stale;
            for (int evaluation = 0; evaluation < 3; ++evaluation) {
                pager.begin_evaluation();
                for (const auto & block : cuda_plan.blocks) {
                    auto ticket = pager.enter(block.block_index);
                    if (evaluation == 0 && block.block_index == 91) { stale = ticket; }
                    auto * view = pager.checked_tensor(ticket, block.tensors[0].profile.name);
                    std::vector<float> read(block.tensors[0].profile.stored_bytes/4);
                    ggml_backend_tensor_get(view,read.data(),0,read.size()*4);
                    check(read.front() == float(block.block_index) && read.back() == float(block.block_index));
                    if (block.block_index == 1337) {
                        rejects([&] { pager.checked_tensor(stale, "weight0"); });
                        check(ticket.slot == 0);
                    }
                    pager.complete(ticket, gpu.get());
                }
            }
            check(pager.statistics().loads == 9 && pager.statistics().reuses == 7 && pager.statistics().evictions == 5);
            check(pager.statistics().verified_blocks == 3 && pager.statistics().runtime_host_copy_bytes == 0);
            check(pager.slot_states().size() == 2 && pager.slot_states()[0].generation == 6 && pager.slot_states()[1].generation == 3);
            puts("PASS actual CUDA slot reuse, persistent host lifetime, checked resolver after reuse, exact bytes and counters");
            test_prefetch(gpu.get(),profile);
            test_pending_lifetime(gpu.get());
        } else { puts("SKIP CUDA pager transport: no GPU"); }
    } catch (const std::exception & error) { fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
    return 0;
}
