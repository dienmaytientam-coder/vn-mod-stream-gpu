# VN Mod Stream GPU

![VN Mod Stream GPU — llama.cpp GPU Weight Streaming Mod](docs/vn-mod-stream-gpu-infographic.webp)

VN Mod Stream GPU is an experimental `llama.cpp` modification for running AI models that are larger than available GPU VRAM while keeping the main transformer computation on CUDA.

The project is designed for single-GPU systems where VRAM is limited but users still need larger models, long context windows, Vision workloads, or integrated MTP.

> **Experimental community release:** the current implementation is shared so the community can test, improve, and extend it. At this time, only the models and hardware explicitly listed in the Tested section are declared as validated.

## Overview

The core idea is to replace traditional CPU-heavy offloading with a GPU weight-streaming path:

**GGUF Model → Persistent Pinned RAM → Reusable CUDA Slots → GPU Compute**

The streamed weights remain available in pinned host memory and are transferred asynchronously to reusable VRAM slots when needed. The transformer computation itself stays on CUDA instead of silently falling back to CPU.

This design is intended for GPUs that do not have enough VRAM to hold the complete model while still allowing practical large-context inference.

## Key features

- **Automatic Weight Streaming** based on the configured or detected VRAM budget.
- **Models larger than VRAM** can be loaded through streamed weight residency.
- **Persistent pinned RAM** keeps streamed model weights available without re-reading the GGUF for every token.
- **Two reusable CUDA slots** are used for streamed transformer blocks.
- **Async H2D transfer + prefetch** overlaps host-to-device transfer with evaluation where possible.
- **CUDA transformer compute** is enforced for the supported streaming path; unsupported layouts fail during load rather than silently CPU-offloading transformer work.
- **Profile Cache + Plan Cache** reduce repeated profiling/planning work after a validated cache is created.
- **Flash Attention** can be used with supported `llama.cpp` configurations.
- **Q4_0 K/V cache** can be used in configurations such as the tested setup.
- **Vision / mmproj accounting** is integrated into the server path so Vision memory is included in the VRAM plan.
- **Integrated MTP** is supported for the resident embedded-MTP path.
- Intended for **single-GPU PCs and workstations**, including 12 GB-class GPUs.

## Why this approach

Traditional RAM + CPU offloading reduces VRAM pressure by moving work away from the GPU, but that can significantly reduce inference performance.

VN Mod Stream GPU instead focuses on the path:

**RAM → VRAM → CUDA GPU**

On the tested configuration, this approach provides better practical performance than CPU-heavy offloading because streamed model weights are moved into VRAM while transformer computation remains on the GPU.

Actual performance depends on the model architecture, quantization, context size, PCIe bandwidth, CPU/RAM bandwidth, CUDA version, and GPU.

## Tested configuration

The current public test was performed with:

- **GPU:** NVIDIA GeForce RTX 3060
- **VRAM:** 12 GB
- **Model:** Qwen3.8 27B
- **Model source:** [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
- **IQ2_S model size:** approximately **9.61 GB**
- **IQ3_S model size:** approximately **12.1 GB**

### Observed context results

| Quantization | Model size | GPU | Observed context |
| --- | ---: | --- | ---: |
| IQ2_S | ~9.61 GB | RTX 3060 12 GB | ~250K |
| IQ3_S | ~12.1 GB | RTX 3060 12 GB | ~124K |

These context values are observations from the declared test environment, not a universal guarantee for every system or model.

### Current tested scope

The mod currently operates well on the declared tested models above.

**No other model family is currently declared as tested.**

Other GGUF models may work, but until they are actually validated they should be treated as untested.

## Architecture highlights

The current implementation includes the following ideas represented by the project design:

1. The GGUF model is profiled.
2. The planner calculates VRAM budget, runtime CUDA allocation, reserve, and external CUDA usage.
3. Weights selected for streaming remain in persistent pinned RAM.
4. Two reusable CUDA slots hold the currently required streamed blocks.
5. Prefetch prepares future block demand asynchronously.
6. H2D transfer can overlap with GPU evaluation.
7. Transformer nodes in the supported path remain on CUDA.
8. Profile and plan metadata can be cached for faster validated restarts.

This is intended to make larger models and longer contexts practical on limited-VRAM GPUs without relying on CPU transformer offload.

## Installation

The integration patch targets this exact upstream `llama.cpp` revision:

`73c941b11165cc0f7a36ba17380e79e8fe9797dc`

```sh
git clone https://github.com/dienmaytientam-coder/vn-mod-stream-gpu.git
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp
git checkout 73c941b11165cc0f7a36ba17380e79e8fe9797dc

cp ../vn-mod-stream-gpu/mod/src/llama-weight-stream* src/
cp ../vn-mod-stream-gpu/mod/tests/test-weight-stream* tests/

git apply --check ../vn-mod-stream-gpu/mod/integration.patch
git apply ../vn-mod-stream-gpu/mod/integration.patch

cmake -S . -B vn-mod-stream-gpu \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build vn-mod-stream-gpu --target llama-server -j
```

Do not move an already configured CMake build directory. Configure it again with `-S` and `-B` if its location changes.

## Run

Minimal weight-streaming server command:

```sh
./vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --parallel 1
```

When `--vn-mod-gpu` is omitted, the mod uses the detected VRAM capacity of the selected CUDA GPU as the upper budget. Existing CUDA allocations and the reserve are still included in planner accounting.

Manual budget example:

```sh
./vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --vn-mod-gpu 11G \
  --vn-mod-gpu-wstream-reserve 2G \
  --parallel 1
```

### Weight-streaming parameters

- `--vn-mod-gpu-wstream auto`  
  Enables the automatic GPU weight-streaming path. The planner profiles the model, detects the selected CUDA GPU, calculates runtime CUDA usage, chooses which transformer blocks must remain resident or be streamed, and creates the reusable CUDA-slot plan. The default mode is `off`.

- `--vn-mod-gpu 11G`  
  Sets the **upper GPU-memory budget used by the streaming planner**. `G` means GiB and `M` means MiB. This is a ceiling, not a request to allocate exactly 11 GiB. If this option is omitted, VN Mod Stream GPU uses the detected total VRAM of the selected GPU as the upper budget.

- `--vn-mod-gpu-wstream-reserve 2G`  
  Keeps 2 GiB of safety headroom outside the streaming plan. This space helps absorb CUDA/runtime allocations and memory fluctuations instead of planning weights up to the absolute limit. The default reserve is 2048 MiB.

The planner also accounts for runtime CUDA allocations and other detected CUDA memory usage. Conceptually:

```text
planner limit = GPU budget - reserve - accounted runtime/external CUDA usage
```

Example on a 12 GiB GPU:

- `--vn-mod-gpu 12G --vn-mod-gpu-wstream-reserve 2G` → planner ceiling after reserve is about 10 GiB before additional accounted runtime/external CUDA usage.
- `--vn-mod-gpu 11G --vn-mod-gpu-wstream-reserve 2G` → planner ceiling after reserve is about 9 GiB before additional accounted runtime/external CUDA usage.
- Omitting `--vn-mod-gpu` on a detected 12 GiB GPU with the default 2 GiB reserve is approximately equivalent to using a 12 GiB upper budget with 2 GiB reserved.

Use a larger reserve when you prefer more OOM safety. Use a smaller reserve only when you understand the CUDA memory requirements of the selected model and context.

## Cache

Cache directory resolution order:

1. `VN_WSTREAM_CACHE_DIR`
2. `$XDG_CACHE_HOME/vn-mod-stream-gpu`
3. `$HOME/.cache/vn-mod-stream-gpu`
4. Operating-system temporary directory

Example:

```sh
VN_WSTREAM_CACHE_DIR=/mnt/fast-cache/vn-mod-stream-gpu \
  ./vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --parallel 1
```

An optional `vn-gpu-wstream.conf` file can be placed in the process working directory.

```ini
[vn-mod]
gpu=auto

[vn-mod-gpu-wstream]
mode=auto
reserve=2048M
cache=true
cache_dir=/mnt/fast-cache/vn-mod-stream-gpu
```

## Current runtime constraints

The project is experimental.

Current supported/tested direction:

- Linux
- NVIDIA CUDA
- exactly one selected CUDA GPU
- single-sequence streaming path (`--parallel 1`)
- declared tested model configuration above

Current limitations include:

- LoRA is not supported in the auto streaming path.
- Non-MTP speculative modes are not supported in the auto streaming path.
- Windows is not currently a supported runtime target.
- Model layouts outside the supported streaming contract fail during load.
- Models not listed in the Tested section have not yet been validated by this project.

## Company and contact information

**CÔNG TY TNHH TM KỸ THUẬT CÔNG NGHIỆP TIẾN TÂM**

**Tax code / MST:** 3703030204

- 61/18A Đường Lê Văn Tiên, Khu phố Đông Chiêu, Dĩ An, Thành Phố Hồ Chí Minh
- +84931855546 — Ngọc Long
- [dienmaytientam@gmail.com](mailto:dienmaytientam@gmail.com)

## License and upstream attribution

This repository is distributed under the MIT License.

The files created for VN Mod Stream GPU under `mod/src` and `mod/tests` carry SPDX MIT identifiers and the project copyright notice.

`mod/integration.patch` modifies existing files from `ggml-org/llama.cpp` and contains patch context from upstream revision `73c941b11165cc0f7a36ba17380e79e8fe9797dc`. Upstream material remains copyright of the ggml authors and is used under the upstream MIT License.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). This project is independent and is not an official `llama.cpp` feature.

---

# Tiếng Việt

## Giới thiệu

**VN Mod Stream GPU** là bản mod thử nghiệm trên nền `llama.cpp`, hướng tới việc chạy model AI có dung lượng lớn hơn VRAM của GPU trong khi vẫn giữ phần tính toán transformer chính trên CUDA.

Dự án dành cho các hệ thống chỉ có một GPU, đặc biệt trong trường hợp GPU thiếu VRAM nhưng người dùng vẫn cần:

- model lớn,
- context dài,
- Vision,
- MTP,
- và hiệu năng thực tế tốt hơn giải pháp offloading nặng sang CPU.

> **Bản thử nghiệm chia sẻ cho cộng đồng:** dự án hiện được công khai để cộng đồng tiếp tục kiểm thử, cải tiến và phát triển. Hiện tại chỉ những model và phần cứng được liệt kê trong mục Tested mới được công bố là đã kiểm thử.

## Cơ chế chính

Luồng xử lý được thiết kế theo hướng:

**GGUF Model → Persistent Pinned RAM → CUDA Slots → GPU Compute**

Thay vì đưa transformer compute sang CPU khi VRAM không đủ, VN Mod Stream GPU giữ các weight cần stream trong pinned RAM, chuyển chúng sang các CUDA slot trong VRAM khi cần và tiếp tục tính toán trên GPU.

## Tính năng

- **Tự động Weight Streaming** theo ngân sách VRAM.
- Có thể **tải model lớn vượt dung lượng VRAM**.
- Hỗ trợ GPU thiếu VRAM nhưng vẫn hướng tới **context lớn**.
- **Persistent pinned RAM** giữ weight đã chuẩn bị trong RAM, tránh đọc lại GGUF cho từng token.
- **2 CUDA slot tái sử dụng** cho các transformer block được stream.
- **Async H2D + Prefetch** để chồng lấp việc truyền RAM → VRAM với quá trình evaluation khi có thể.
- Transformer compute trong đường streaming được hỗ trợ sẽ chạy trên **CUDA GPU**; cấu hình không hỗ trợ sẽ fail thay vì âm thầm offload transformer sang CPU.
- **Profile Cache + Plan Cache** giúp giảm việc profile/planning lặp lại khi cache hợp lệ.
- Có thể dùng **Flash Attention** theo cấu hình `llama.cpp` tương thích.
- Có thể dùng **Q4_0 K/V Cache** như trong hướng cấu hình đã thử nghiệm.
- Có tích hợp accounting cho **Vision / mmproj**.
- Hỗ trợ hướng **embedded MTP** đã tích hợp.
- Phù hợp cho PC/workstation 1 GPU, bao gồm GPU lớp **12 GB VRAM**.

## Khác biệt so với RAM + CPU Offloading

Offloading truyền thống giảm áp lực VRAM bằng cách chuyển một phần công việc sang RAM và CPU. Điều này có thể làm giảm đáng kể tốc độ inference.

VN Mod Stream GPU tập trung vào luồng:

**RAM → VRAM → CUDA GPU**

Trong cấu hình đã test, hướng này cho hiệu năng thực tế tốt hơn kiểu CPU-heavy offloading vì weight được stream từ RAM vào VRAM, còn transformer compute tiếp tục chạy trên GPU.

Tốc độ thực tế vẫn phụ thuộc vào model, quantization, context, PCIe, băng thông RAM/CPU, CUDA và GPU.

## Cấu hình đã test

Bản test công khai hiện tại sử dụng:

- **GPU:** NVIDIA GeForce RTX 3060
- **VRAM:** 12 GB
- **Model:** Qwen3.8 27B
- **Nguồn model:** [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF)
- **IQ2_S:** khoảng **9.61 GB**
- **IQ3_S:** khoảng **12.1 GB**

### Kết quả context ghi nhận

| Quantization | Dung lượng model | GPU | Context ghi nhận |
| --- | ---: | --- | ---: |
| IQ2_S | ~9.61 GB | RTX 3060 12 GB | ~250K |
| IQ3_S | ~12.1 GB | RTX 3060 12 GB | ~124K |

Đây là kết quả ghi nhận trên đúng môi trường thử nghiệm đã khai báo, không phải cam kết chung cho mọi máy hoặc mọi model.

### Phạm vi tested hiện tại

Bản mod hiện hoạt động tốt trên các model tested đã khai báo phía trên.

**Hiện chưa thử nghiệm bất kỳ họ model nào khác.**

Các GGUF khác có thể hoạt động, nhưng trước khi được test thực tế thì vẫn được xem là **untested**.

## Điểm nổi bật về kiến trúc

1. Profile model GGUF.
2. Planner tính ngân sách VRAM, reserve, runtime CUDA allocation và external CUDA usage.
3. Weight cần streaming được giữ trong persistent pinned RAM.
4. Hai CUDA slot được tái sử dụng cho các block cần chạy.
5. Prefetch chuẩn bị block sẽ được dùng tiếp theo.
6. H2D có thể chạy chồng lấp với GPU evaluation.
7. Transformer node trong đường hỗ trợ vẫn chạy CUDA.
8. Profile/Plan cache hỗ trợ khởi động lại nhanh hơn sau khi đã tạo cache hợp lệ.

Mục tiêu là giúp GPU VRAM hạn chế có thể chạy model lớn và context dài mà không phải phụ thuộc vào CPU transformer offload.

## Cài đặt

Patch hiện khóa theo đúng upstream commit:

`73c941b11165cc0f7a36ba17380e79e8fe9797dc`

```sh
git clone https://github.com/dienmaytientam-coder/vn-mod-stream-gpu.git
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp
git checkout 73c941b11165cc0f7a36ba17380e79e8fe9797dc

cp ../vn-mod-stream-gpu/mod/src/llama-weight-stream* src/
cp ../vn-mod-stream-gpu/mod/tests/test-weight-stream* tests/

git apply --check ../vn-mod-stream-gpu/mod/integration.patch
git apply ../vn-mod-stream-gpu/mod/integration.patch

cmake -S . -B vn-mod-stream-gpu \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build vn-mod-stream-gpu --target llama-server -j
```

## Chạy

Lệnh tối thiểu:

```sh
./vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --parallel 1
```

Nếu không truyền `--vn-mod-gpu`, mod tự dùng dung lượng VRAM được phát hiện của GPU đã chọn làm giới hạn trên. Planner vẫn tính reserve và CUDA allocation đang tồn tại.

Ví dụ đặt thủ công:

```sh
./vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --vn-mod-gpu 11G \
  --vn-mod-gpu-wstream-reserve 2G \
  --parallel 1
```

### Giải thích 3 tham số Weight Streaming

- `--vn-mod-gpu-wstream auto`  
  **Bật chế độ Auto GPU Weight Streaming.** Planner sẽ profile model, nhận diện CUDA GPU đang chọn, tính lượng CUDA memory cần cho runtime, quyết định block transformer nào giữ resident và block nào phải stream, sau đó tạo kế hoạch dùng các CUDA slot tái sử dụng. Mặc định tính năng này là `off`.

- `--vn-mod-gpu 11G`  
  Đặt **trần ngân sách GPU memory cho planner**. Ký hiệu `G` được hiểu là GiB, `M` là MiB. Đây là giới hạn tối đa để planner tính toán, **không có nghĩa là bắt buộc phải cấp phát đủ 11 GiB**. Nếu bỏ tham số này, VN Mod Stream GPU tự dùng tổng VRAM phát hiện được của GPU đang chọn làm trần ngân sách.

- `--vn-mod-gpu-wstream-reserve 2G`  
  Chừa **2 GiB vùng an toàn** ngoài kế hoạch streaming. Phần này giúp tránh planner sử dụng VRAM sát giới hạn tuyệt đối, tạo khoảng trống cho CUDA/runtime và dao động bộ nhớ trong lúc inference. Giá trị mặc định là 2048 MiB.

Planner còn tính thêm CUDA memory của runtime và các allocation CUDA khác đã phát hiện. Có thể hiểu gần đúng:

```text
Giới hạn planner = GPU budget - reserve - runtime/external CUDA đã được tính
```

Ví dụ với GPU 12 GiB:

- `--vn-mod-gpu 12G --vn-mod-gpu-wstream-reserve 2G` → sau reserve còn khoảng 10 GiB làm trần planner, trước khi trừ tiếp runtime/external CUDA đã được accounting.
- `--vn-mod-gpu 11G --vn-mod-gpu-wstream-reserve 2G` → sau reserve còn khoảng 9 GiB làm trần planner, trước khi trừ tiếp runtime/external CUDA.
- Nếu **không truyền** `--vn-mod-gpu`, GPU được phát hiện là 12 GiB và reserve mặc định 2 GiB, thì gần tương đương đặt upper budget 12 GiB và chừa 2 GiB reserve.

Muốn an toàn OOM hơn thì tăng `reserve`. Chỉ nên giảm `reserve` khi đã hiểu rõ lượng CUDA memory mà model, context và các thành phần runtime cần sử dụng.

## Cache

Thứ tự xác định cache:

1. `VN_WSTREAM_CACHE_DIR`
2. `$XDG_CACHE_HOME/vn-mod-stream-gpu`
3. `$HOME/.cache/vn-mod-stream-gpu`
4. Thư mục temporary của hệ điều hành

Có thể đặt `vn-gpu-wstream.conf` tại working directory nếu muốn dùng file cấu hình.

## Giới hạn hiện tại

Đây vẫn là dự án thử nghiệm.

Hướng hỗ trợ/tested hiện tại:

- Linux
- NVIDIA CUDA
- đúng một CUDA GPU được chọn
- `--parallel 1`
- model tested đã khai báo phía trên

Các giới hạn hiện tại:

- LoRA chưa được hỗ trợ trong auto streaming path.
- Speculative mode ngoài embedded MTP chưa được hỗ trợ.
- Windows hiện chưa phải runtime target được hỗ trợ.
- Layout model ngoài streaming contract sẽ fail khi load.
- Model không nằm trong danh sách Tested hiện chưa được dự án xác nhận.

## Thông tin liên hệ

**CÔNG TY TNHH TM KỸ THUẬT CÔNG NGHIỆP TIẾN TÂM**

**MST:** 3703030204

- 61/18A Đường Lê Văn Tiên, Khu phố Đông Chiêu, Dĩ An, Thành Phố Hồ Chí Minh
- +84931855546 — Ngọc Long
- [dienmaytientam@gmail.com](mailto:dienmaytientam@gmail.com)

## Giấy phép

Dự án được phát hành theo MIT License.

Phần upstream `llama.cpp` / `ggml` vẫn thuộc bản quyền của các tác giả upstream và tuân theo MIT License của upstream.

Xem [`LICENSE`](LICENSE) và [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) để biết chi tiết.
