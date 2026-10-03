# VN GPU weight streaming mod

This repository distributes the mod as a patch. It does not include the llama.cpp source tree, binaries, or model files.

The patch is based on upstream `llama.cpp` commit `73c941b11165cc0f7a36ba17380e79e8fe9797dc`. Apply it to that exact checkout:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 73c941b11165cc0f7a36ba17380e79e8fe9797dc
git apply /path/to/vn-mod-stream-gpu.patch
cmake -S . -B /path/to/build-vn-mod-stream-gpu -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/build-vn-mod-stream-gpu --target llama-server -j
```

The mod is experimental. It requires CUDA and exactly one CUDA GPU. Unsupported model layouts and runtime options fail during load. The current cache writer uses POSIX file operations; Linux with CUDA is the tested environment. The config file currently uses `/datas/serverai/config/vn-gpu-wstream.conf` by default, and the default cache directory is `/datas/serverai/cache/llama-weight-stream`.

The patch contains changes to upstream files and new mod files. The upstream project remains under its MIT license; this repository also provides the patch under the MIT license with attribution retained in `LICENSE`. The mod is not an official llama.cpp feature.
