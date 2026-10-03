# VN Mod Stream GPU — source and integration

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

cmake -S . -B vn-mod-stream-gpu -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
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

When `--vn-mod-gpu` is omitted, the mod uses the detected VRAM capacity of the selected CUDA GPU as the upper budget. Existing CUDA allocations and the reserve are still included in the planner accounting.

To override the budget or reserve:

```sh
./vn-mod-stream-gpu/bin/llama-server \
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
  ./vn-mod-stream-gpu/bin/llama-server \
  -m /path/to/model.gguf \
  --vn-mod-gpu-wstream auto \
  --parallel 1
```

An optional `vn-gpu-wstream.conf` file is read from the process working directory when it exists; it is not required for normal use. For example:

```ini
[vn-mod]
gpu=auto

[vn-mod-gpu-wstream]
mode=auto
reserve=2048M
cache=true
cache_dir=/mnt/fast-cache/vn-mod-stream-gpu
```

In the INI file, `gpu=auto` selects automatic budget detection. CLI arguments still override config values.

## Current runtime constraints

The mod is experimental. The supported path is Linux with CUDA and exactly one selected CUDA GPU. Weight streaming currently requires a single sequence (`--parallel 1`). Unsupported model layouts and runtime combinations fail during load. LoRA and non-MTP speculative modes are not supported by the auto streaming path. The cache writer currently uses POSIX file operations, so Windows is not a supported runtime target.
