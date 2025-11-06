# Make sure your my_model_repo is in your current directory
export MODEL_REPO_PATH=$(pwd)/model_repository

# Use the container tag that includes the vLLM backend
#docker run --gpus all --rm -it \
#  -p 8000:8000 \
#  -p 8001:8001 \
#  -p 8002:8002 \
#  -v $MODEL_REPO_PATH:/models \
#  nvcr.io/nvidia/tritonserver:24.08-vllm-python-py3 \
#  tritonserver --model-repository=/models
#

curl -X POST localhost:8000/v2/models/gaia/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["O que é a computação em nuvem?"]
    }
  ]
}'