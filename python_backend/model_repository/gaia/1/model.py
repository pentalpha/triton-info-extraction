import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """
    Classe do modelo Python para o Triton.
    """

    def initialize(self, args):
        """
        Chamado uma vez quando o modelo é carregado.
        Carrega o tokenizador e o modelo da Hugging Face.
        """
        self.model_id = "CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it"
        
        print(f"Carregando tokenizador {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        print(f"Carregando modelo {self.model_id}...")
        # Carregar em 4-bit (quantização) para economizar memória
        # Requer 'bitsandbytes' (instalaremos no Dockerfile)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True,
        )
        print("Modelo carregado com sucesso.")

    def execute(self, requests):
        """
        Chamado para cada request de inferência.
        """
        responses = []
        
        # Como max_batch_size = 1, iteramos sobre um único request
        for request in requests:
            # 1. Obter o tensor de entrada (o prompt)
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "PROMPT")
            # Decodificar de tensor bytes para string python
            prompt_bytes = prompt_tensor.as_numpy().flatten()[0]
            prompt_string = prompt_bytes.decode("utf-8")

            # 2. Formatar o prompt para o modelo de instrução (chat)
            # Isso é crucial para modelos -it (instruct-tuned)
            messages = [{"role": "user", "content": prompt_string}]
            chat_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            # 3. Tokenizar o prompt formatado
            inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)

            # 4. Gerar o texto
            # Hardcoding dos parâmetros de geração para simplicidade
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            # 5. Decodificar a saída
            # Pegamos apenas os tokens gerados, pulando o prompt de entrada
            input_length = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                outputs[0][input_length:], 
                skip_special_tokens=True
            )

            # 6. Criar o tensor de saída
            output_tensor = pb_utils.Tensor(
                "GENERATED_TEXT",
                np.array([generated_text], dtype=object)
            )

            # 7. Enviar a resposta
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        Chamado quando o modelo é descarregado.
        """
        print("Limpando o modelo...")
        self.model = None
        self.tokenizer = None
        print("Finalizado.")