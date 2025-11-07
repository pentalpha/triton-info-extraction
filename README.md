# Triton Information Extraction

Containers configured for information extraction based on NVIDIA Triton

## Backends

### Python

- Model: CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it
- Container: updated version of nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3

### vLLM

- Model: Qwen/Qwen3-0.6B;
- Container: nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3

## Build

## Running backends

```sh
cd <backend_dir> 
chmod +x up.sh
./up.sh
```