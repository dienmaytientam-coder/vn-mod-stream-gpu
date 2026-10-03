# Third-Party Notices

## llama.cpp / ggml

This repository contains source additions and an integration patch intended to be applied to:

- Project: llama.cpp
- Upstream: https://github.com/ggml-org/llama.cpp
- Upstream revision: `73c941b11165cc0f7a36ba17380e79e8fe9797dc`
- License: MIT
- Upstream copyright: Copyright (c) 2023-2026 The ggml authors

The file `mod/integration.patch` contains modifications to existing llama.cpp files and therefore also contains patch context originating from those upstream files. That upstream material remains subject to the llama.cpp MIT License and copyright notice retained in this repository's [`LICENSE`](LICENSE).

The VN GPU weight streaming additions in `mod/src`, the related tests in `mod/tests`, and project-specific modifications are distributed under the MIT License with:

Copyright (c) 2026 dienmaytientam-coder

This repository does not claim ownership of upstream llama.cpp/ggml code, names, or third-party components used by llama.cpp. It is an independent experimental modification and is not an official llama.cpp feature.

When redistributing a patched source tree or binaries built from it, retain the applicable MIT license and copyright notices for upstream and project-specific portions.
