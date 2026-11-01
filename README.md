# llama-server-t4

A prebuilt [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` binary for **Kaggle GPU notebooks (NVIDIA T4)**.

Compiling llama.cpp with CUDA on Kaggle takes about 20 minutes per session, and that time counts against your weekly GPU quota. This repo holds the finished binary as a GitHub Release, so a notebook can download it in seconds instead.

Together with a Qwen3-Coder GGUF model, an ngrok tunnel and Claude Code, it gives you a self-hosted coding agent.

```
Claude Code (Mac) → ngrok → Kaggle notebook → llama-server → Qwen3-Coder-30B-A3B
```

## What is in the release

| Item | Value |
|---|---|
| Release tag | `llama-server-t4` |
| Asset | `llama-server-t4.tar.gz` (contains one file: `llama-server`) |
| Target GPU | NVIDIA T4 (CUDA architecture 75), including the T4 x2 option |
| Platform | Linux x86_64 (Kaggle image, Ubuntu 24.04) |
| CUDA toolkit | 12.8 |
| Build type | Release, static llama.cpp/ggml libraries (`-DBUILD_SHARED_LIBS=OFF`) |
| Backend | CUDA |

The binary was built inside a Kaggle session, so it matches Kaggle's libraries. It is **not** built for other GPUs. On a P100 (architecture 60) or on a non-Kaggle machine it will fail, for example with a "no kernel image" error.

If you rebuild it, update the table above with the llama.cpp version or commit you used.

## Use it in a Kaggle notebook

Requirements: accelerator set to **GPU T4 x2**, and **Internet** turned on.

Replace the build cell with this download cell. Set `GITHUB_REPO` to your own `user/repo`:

```python
import os, subprocess

GITHUB_REPO = "YOUR_USER/llama-server-t4"
BIN_DIR = "/kaggle/working/bin"
SERVER_BIN = f"{BIN_DIR}/llama-server"

# CUDA libraries the binary needs at run time
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

if not os.path.exists(SERVER_BIN):
    os.makedirs(BIN_DIR, exist_ok=True)
    !curl -fL -o /tmp/llama-server-t4.tar.gz https://github.com/{GITHUB_REPO}/releases/download/llama-server-t4/llama-server-t4.tar.gz
    !tar xzf /tmp/llama-server-t4.tar.gz -C {BIN_DIR}
    !chmod +x {SERVER_BIN}

missing = subprocess.run(f"ldd {SERVER_BIN} | grep 'not found'", shell=True, capture_output=True, text=True).stdout
assert not missing, "Missing libraries:\n" + missing
print("OK:", SERVER_BIN)
```

It should print `OK: /kaggle/working/bin/llama-server` within seconds.

### Start the server

```python
import subprocess

command = [
    SERVER_BIN,
    "-m", MODEL_PATH,            # path to the .gguf file
    "--host", "0.0.0.0",
    "--port", "8000",
    "-c", "98304",               # context size, Claude Code needs a large one
    "-ngl", "99",                # offload all layers; splits across both T4s
    "-np", "1",                  # one slot, so the whole context serves one conversation
    "--jinja",                   # use the model's chat template: enables native tool calling
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",   # compressed cache so 98304 fits on 2x T4
    "--alias", "qwen3-coder",
    "--api-key", API_KEY,        # the ngrok URL is public, always set a key
    "--temp", "0.3",
]
server_process = subprocess.Popen(command)
```

If the server runs out of GPU memory, lower `-c` to `65536` and remove the three cache flags (`-fa on -ctk q8_0 -ctv q8_0`).

Then expose port 8000 with ngrok (store the token as the Kaggle secret `NGROK_AUTHTOKEN`, never in the notebook) and test the endpoint:

```bash
curl https://YOUR-NGROK-HOST/v1/models -H "Authorization: Bearer YOUR_API_KEY"
```

## Use it with Claude Code

Claude Code needs an API-compatible endpoint. Add this function to `~/.zshrc` on your Mac:

```bash
export QWEN_API_KEY="your-api-key"
claude-qwen() {
  local url="${1%/}"; url="${url%/v1}"
  env -u ANTHROPIC_API_KEY \
    ANTHROPIC_BASE_URL="$url" \
    ANTHROPIC_AUTH_TOKEN="$QWEN_API_KEY" \
    ANTHROPIC_MODEL="qwen3-coder" \
    ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3-coder" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3-coder" \
    ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3-coder" \
    ANTHROPIC_SMALL_FAST_MODEL="qwen3-coder" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
    ANTHROPIC_CUSTOM_HEADERS="ngrok-skip-browser-warning: true" \
    claude "${@:2}"
}
```

The ngrok URL changes every session, so pass the current one each time:

```bash
cd ~/your-project
claude-qwen https://YOUR-NGROK-HOST
```

Because the server has a fixed context size, also tell Claude Code to compact early:

```bash
export CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=55
```

Inside Claude Code, run `/clear` between tasks and `/compact` when `/context` shows around 50,000 tokens.

## Updating the binary

Rebuild only when you want a newer llama.cpp.

1. In a Kaggle GPU session, build llama.cpp with CUDA for the T4:
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
   cmake --build build --config Release -j4 --target llama-server
   ```
   If CMake cannot find the CUDA driver library (`CUDA::cuda_driver`), point it at the real file, which on Kaggle is `/usr/local/nvidia/lib64/libcuda.so.1`.
2. Package it:
   ```bash
   tar czf llama-server-t4.tar.gz -C build/bin llama-server
   ```
3. Download the archive from the notebook's output panel before the session ends.
4. On GitHub, edit the `llama-server-t4` release and replace the asset.

## Troubleshooting

| Problem | What to try |
|---|---|
| `curl: (22) ... 404` | The repo must be public, and the release tag and file name must match exactly (`llama-server-t4`, `llama-server-t4.tar.gz`). |
| Library check lists `not found` | Make sure the `LD_LIBRARY_PATH` line ran, and that the session uses a GPU accelerator. |
| "no kernel image is available" | The session has a GPU other than a T4 (for example a P100). Rebuild with the matching `CMAKE_CUDA_ARCHITECTURES`. |
| Out of GPU memory at start | Lower `-c` to `65536` and drop the cache-compression flags. |
| `401 Unauthorized` | The request is missing `Authorization: Bearer <API_KEY>`. |
| 400 error: "exceeds the available context size" | The conversation is larger than `-c`. Use `/compact` or `/clear`, and keep the compaction settings above. |
| Tool calls come back as plain text | Check that `--jinja` is in the command, and consider a newer llama.cpp build. |

## Security

- The release contains only a compiled binary. No keys or tokens belong in this repo.
- Your ngrok URL is public while the notebook runs. Always start the server with `--api-key`.
- Store `NGROK_AUTHTOKEN` and your API key as Kaggle secrets. If a token has ever appeared in a notebook you shared, rotate it.

## License and credits

`llama-server` is part of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT License). The model weights you load have their own licenses. Check the license of the model you download.