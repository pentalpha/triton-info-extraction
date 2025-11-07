export TRITON_SERVER_URL=0.0.0.0:8000

curl -X POST $TRITON_SERVER_URL/v2/models/text_llm/infer -d \
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

curl -X POST $TRITON_SERVER_URL/v2/models/text_llm/generate -d \
'{
  "text_input": "Quem é Pitágoras de Azevedo Alves Sobrinho?",
  "parameters": {
    "max_tokens": 512,
    "stream": false
  }
}'