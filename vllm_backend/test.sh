# Make sure your my_model_repo is in your current directory
export TRITON_SERVER_URL=10.0.0.5:8003

curl -X POST $TRITON_SERVER_URL/v2/models/text_llm/generate -d \
'{
  "text_input": "Quem é Pitágoras de Azevedo Alves Sobrinho?",
  "parameters": {
    "max_tokens": 512,
    "stream": false
  }
}'