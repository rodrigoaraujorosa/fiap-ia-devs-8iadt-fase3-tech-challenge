---
language:
- pt
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.2
tags:
- medical
- portuguese
- lora
- qlora
- peft
- fine-tuned
pipeline_tag: text-generation
---



# mistral-7b-assistente-hospitalar-v1

## Descrição

Adaptador LoRA fine-tuned sobre o modelo base **Mistral-7B-Instruct-v0.2** para o domínio de assistência médica hospitalar em **português brasileiro**.

Desenvolvido como parte do **Tech Challenge — Fase 3** do curso de Inteligência Artificial para DEVs (8IADT), com foco em responder perguntas clínicas e científicas no contexto hospitalar.

---

## Informações do Treinamento

| Parâmetro | Valor |
|---|---|
| Modelo base | `mistralai/Mistral-7B-Instruct-v0.2` |
| Técnica | QLoRA (PEFT + SFTTrainer) |
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Quantização | 4-bit NF4 + double quantization |
| Dataset | 211.269 exemplos em português brasileiro (domínio médico) |
| Épocas | 1 |
| Batch efetivo | 8 (batch=1 + grad_accum=8) |
| Max sequence length | 256 tokens |
| Infraestrutura | GPU L4 (22 GB VRAM) — Google Colab |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 |

---

## Resultados de Avaliação

Avaliado com 5 perguntas clínicas em português, comparando com respostas do dataset original:

| Configuração de Inferência | Precisão |
|---|---|
| max_new=512, temp=0.7, top_p=0.9, rep=1.1 | ⭐⭐⭐☆☆ 3.2/5 (64%) |
| max_new=1024, temp=0.4, top_p=0.9, rep=1.15 | ⭐⭐☆☆☆ 2.8/5 (56%) |
| **max_new=1024, temp=0.6, top_p=0.85, rep=1.1** | **⭐⭐⭐⭐☆ 4.0/5 (80%)** ← recomendado |

> Treinado com apenas 1 época em GPU L4 (22 GB VRAM). Com épocas adicionais e refinamento do dataset, espera-se melhora significativa na precisão.

---

## Como Usar

### Instalação das dependências

```bash
pip install transformers peft bitsandbytes accelerate
```

### Carregamento do modelo

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuração de quantização 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Carrega o modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.pad_token = tokenizer.eos_token

# Aplica o adaptador LoRA
model = PeftModel.from_pretrained(base_model, "rodrigoaraujorosa/mistral-7b-assistente-hospitalar-v1")
model.eval()
```

### Geração de respostas

```python
from transformers import GenerationConfig, pipeline

model.generation_config = GenerationConfig(
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=1024,
)

def gerar_resposta(pergunta: str) -> str:
    prompt = f"<s>[INST] {pergunta} [/INST]"
    result = generator(
        prompt,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = result[0]["generated_text"]
    return output.split("[/INST]")[-1].strip()

# Exemplo
pergunta = "Os altos níveis de procalcitonina na fase inicial após o transplante de fígado pediátrico indicam um resultado pós-operatório ruim?"
print(gerar_resposta(pergunta))
```

---

## Limitações

- O modelo foi treinado com dataset gerado por IA, o que pode resultar em respostas com linguagem acadêmica e genérica em vez de respostas clínicas diretas
- Treinado com apenas 1 época — retreinamento com mais épocas pode melhorar a qualidade
- **Não deve ser usado para diagnóstico ou decisão clínica real** — destina-se a fins educacionais e de pesquisa

---

## Citação

```bibtex
@misc{mistral7b-assistente-hospitalar-v1,
  title={mistral-7b-assistente-hospitalar-v1},
  author={Rosa, Rodrigo de Araújo and Silva, Elias Maximiano da and Jesus, Fábia Gomes de and Pereira, Danilo},
  year={2026},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/rodrigoaraujorosa/mistral-7b-assistente-hospitalar-v1}}
}
```

---

## Licença

Este adaptador segue a licença do modelo base [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).