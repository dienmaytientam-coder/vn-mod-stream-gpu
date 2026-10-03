# Mod source and integration

These are the mod's new source and test files. `integration.patch` contains only the changes to existing llama.cpp files, such as CLI, model loading, CUDA graph scheduling, CMake, and server integration. It does not contain the new files in this directory.

## Apply to llama.cpp

Use the matching upstream revision, then copy the new files and apply the integration patch:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 73c941b11165cc0f7a36ba17380e79e8fe9797dc

# Replace /path/to/vn-mod-stream-gpu with a local clone of this repository.
cp /path/to/vn-mod-stream-gpu/mod/src/llama-weight-stream* src/
cp /path/to/vn-mod-stream-gpu/mod/tests/test-weight-stream* tests/
git apply /path/to/vn-mod-stream-gpu/mod/integration.patch

cmake -S . -B /path/to/build-vn-mod-stream-gpu -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /path/to/build-vn-mod-stream-gpu --target llama-server -j
```

The build directory can be outside the llama.cpp checkout. Do not move an already configured CMake build directory; configure it again with `-S` and `-B`.

The mod is experimental and supports exactly one CUDA GPU. The config file currently defaults to `/datas/serverai/config/vn-gpu-wstream.conf`; the cache defaults to `/datas/serverai/cache/llama-weight-stream`. Linux with CUDA is the tested environment.
