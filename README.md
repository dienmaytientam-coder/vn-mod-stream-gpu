# VN GPU weight streaming mod

This repository shares only the mod source, tests, and the integration patch for llama.cpp. It does not contain the llama.cpp source tree, build output, binaries, or model files.

The patch targets upstream llama.cpp commit `73c941b11165cc0f7a36ba17380e79e8fe9797dc`. See [`mod/README.md`](mod/README.md) for the file layout and installation steps.

The mod is experimental. It requires CUDA and exactly one CUDA GPU. Unsupported model layouts and runtime options fail during load. Linux with CUDA is the tested environment; the current cache writer uses POSIX file operations. The config file and cache defaults still use `/datas/serverai/...` paths.

The upstream project remains under its MIT license. This repository includes the patch and mod files under MIT with upstream attribution in [`LICENSE`](LICENSE). This is not an official llama.cpp feature.

## License and upstream attribution

This repository is distributed under the MIT License.

The files created for the VN GPU weight streaming mod under `mod/src` and `mod/tests` carry SPDX MIT identifiers and the project copyright notice.

`mod/integration.patch` modifies existing files from `ggml-org/llama.cpp` and contains patch context from upstream revision `73c941b11165cc0f7a36ba17380e79e8fe9797dc`. Upstream material remains copyright of the ggml authors and is used under the upstream MIT License.

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the retained license text and provenance details. This project is independent and is not an official llama.cpp feature.

