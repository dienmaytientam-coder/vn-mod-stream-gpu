# Mod source and integration

These are the mod's new source and test files. `integration.patch` contains only the changes to existing llama.cpp files, such as CLI, model loading, CUDA graph scheduling, CMake, and server integration. It does not contain the new files in this directory.

## Install

Use the exact upstream revision targeted by the patch:

```sh
git clone https://github.com/dienmaytientam-coder/vn-mod-stream-gpu.git
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp
git checkout 73c941b11165cc0f7a36ba17380e79e8fe9797dc

cp ../vn-mod-stream-gpu/mod/src/llama-weight-stream* src/
cp ../vn-mod-stream-gpu/mod/tests/test-weight-stream* tests/

git apply --check ../vn-mod-stream-gpu/mod/integration.patch
git apply ../vn-mod-stream-gpu/mod/integration.patch

cmake -S . -B build-vn-mod-stream-gpu -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vn-mod-stream-gpu --target llama-server -j
```

Do not move an already configured CMake build directory. Configure it again with `-S` and `-B` if its location changes.

## Run

Minimal weight-streaming server command:

```sh
./build-vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --parallel 1
```

When `--vn-mod-gpu` is omitted, the mod uses the detected VRAM capacity of the selected CUDA GPU as the upper budget. Existing CUDA allocations and the reserve are still included in the planner accounting.

To override the budget or reserve:

```sh
./build-vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --vn-mod-gpu 11G \
  --vn-mod-gpu-wstream-reserve 2G \
  --parallel 1
```

## Cache

The cache directory no longer depends on `/datas`. Resolution order is:

1. `VN_WSTREAM_CACHE_DIR`
2. `$XDG_CACHE_HOME/vn-mod-stream-gpu`
3. `$HOME/.cache/vn-mod-stream-gpu`
4. the operating-system temporary directory

Example override:

```sh
VN_WSTREAM_CACHE_DIR=/mnt/fast-cache/vn-mod-stream-gpu \
  ./build-vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --parallel 1
```

An optional legacy INI file is still read from `/datas/serverai/config/vn-gpu-wstream.conf` when that file exists; it is not required for normal use. In the INI file, `gpu=auto` selects automatic budget detection.

## Current runtime constraints

The mod is experimental. The supported path is Linux with CUDA and exactly one selected CUDA GPU. Weight streaming currently requires a single sequence (`--parallel 1`). Unsupported model layouts and runtime combinations fail during load. LoRA and non-MTP speculative modes are not supported by the auto streaming path. The cache writer currently uses POSIX file operations, so Windows is not a supported runtime target.
