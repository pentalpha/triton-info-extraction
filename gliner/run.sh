#!/bin/bash

export TRITON_SERVER_URL=0.0.0.0:8000

# Definimos a lista de labels uma vez, como uma única string
# (Note que removemos as aspas extras do seu exemplo)
LABEL_STRING="rua_ou_logradouro, rua, bairro, municipio, cidade, ponto_de_referencia, nome_do_solicitante, pessoa, numero, street_number, number, complemento, endereço_complemento"

echo "--- Teste 1: Emergência na Rua das Flores ---"
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Preciso de uma ambulância rápido, tem uma pessoa caída na Rua das Flores, número 123, perto da padaria."]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["'"$LABEL_STRING"'"]
    }
  ]
}'

echo -e "\n\n--- Teste 2: Incêndio na Avenida Principal ---"
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Meu nome é João Silva, estou ligando do bairro Centro, na Avenida Principal, 500. Tem um incêndio no apartamento 101."]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["'"$LABEL_STRING"'"]
    }
  ]
}'

echo -e "\n\n--- Teste 3: Acidente em Petrópolis ---"
time curl -X POST $TRITON_SERVER_URL/v2/models/gliner_x_large/infer -d \
'{
  "inputs": [
    {
      "name": "PROMPT",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["Alô, é da polícia? Acabou de acontecer um acidente aqui na esquina da Sete de Setembro com a Rua Nova, em Petrópolis."]
    },
    {
      "name": "LABEL_LIST",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["'"$LABEL_STRING"'"]
    }
  ]
}'