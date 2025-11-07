export MODEL_REPO_PATH=$(pwd)/model_repository

sudo docker run --gpus all --rm -it \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v $MODEL_REPO_PATH:/models \
  nvcr.io/nvidia/tritonserver:25.10-vllm-python-py3 \
  tritonserver --model-repository=/models