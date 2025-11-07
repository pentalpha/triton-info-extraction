# Triton Information Extraction

Containers configured for information extraction based on NVIDIA Triton

Model: CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it

## Backends

### Python

- Container: updated version of nvcr.io/nvidia/tritonserver:24.08-pyt-python-py3

### vLLM

- Container: updated version of nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3

## Build

## Running backends

```sh
cd <backend_dir> 
chmod +x up.sh
./up.sh
```