# Assistente Virtual Médico Hospitalar

Sistema de inteligência artificial para auxílio a médicos em condutas clínicas, desenvolvido como **Tech Challenge — Fase 3** do curso de Inteligência Artificial para DEVs (**8IADT — Grupo 49**).

O sistema combina um modelo de linguagem fine-tunado em dados médicos em português brasileiro com um pipeline de recuperação aumentada por geração (RAG) sobre prontuários de pacientes, orquestrado por um grafo de decisão LangGraph com interface Gradio.

> ⚠️ **Uso exclusivamente acadêmico.** Este sistema não deve ser utilizado para diagnóstico, prescrição ou qualquer conduta clínica em pacientes reais.

---

## Arquitetura

O fluxo de uma consulta é modelado como um **grafo de estados LangGraph** com 9 nós e arestas condicionais explícitas:

```
START
  │
  ▼
[detectar_prescricao]
  │
  ├─(prescrição)─────────────────────────► [recusar_prescricao]
  │                                                  │
  └─(ok)                                             │
       ▼                                             │
[identificar_paciente]                               │
  │                                                  │
  ├─(encontrado)──► [rag_paciente]                   │
  │                      │                           │
  │               [buscar_alertas]                   │
  │                      │                           │
  └─(genérico)──────► [rag_generico]                 │
                          │                          │
                          ▼                          │
                    [invocar_llm]                    │
                          │                          │
                          ▼                          │
                  [formatar_resposta] ◄──────────────┘
                          │
                   [registrar_log]
                          │
                         END
```

**Nós do grafo:**

| Nó | Responsabilidade |
|---|---|
| `detectar_prescricao` | Verifica se a consulta solicita prescrição médica |
| `recusar_prescricao` | Retorna recusa imediata sem acionar a LLM |
| `identificar_paciente` | Fuzzy-match do nome do paciente na query |
| `rag_paciente` | RAG filtrado por `patient_id` — recupera chunks do prontuário no ChromaDB |
| `buscar_alertas` | Exames pendentes/alterados via metadados do ChromaDB |
| `rag_generico` | RAG sem filtro para consultas clínicas genéricas |
| `invocar_llm` | Monta prompt Mistral Instruct e chama o modelo fine-tunado |
| `formatar_resposta` | Monta `MedicalResponse` estruturado com fontes e alertas |
| `registrar_log` | Persiste a interação no audit log (JSON Lines, Google Drive) |

**Componentes principais:**

| Componente | Tecnologia |
|---|---|
| Modelo LLM | `mistralai/Mistral-7B-Instruct-v0.2` + adaptador LoRA (QLoRA 4-bit) |
| Fine-tuning | PEFT + TRL SFTTrainer — Google Colab GPU L4 (17 h) |
| Vector Store (RAG) | ChromaDB persistido no Google Drive |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers) |
| Orquestração | LangChain + LangGraph |
| Prontuários | `data/prontuarios.json` — 20 registros fictícios em PT-BR |
| Interface | Gradio — URL pública gerada no Colab, válida por 72 h |
| Logging de auditoria | JSON Lines com timestamp ISO 8601, rotação diária, 30 dias |

---

## Estrutura do Projeto

Apenas os arquivos e pastas versionados no repositório:

```
├── data/
│   ├── preprocessed/
│   │   └── train_data.json         # 211.269 exemplos médicos PT-BR (formato Alpaca)
│   ├── medical_dict.json           # 131 termos médicos EN→PT-BR
│   ├── prontuarios.json            # Base de prontuários fictícios (20 pacientes)
│   └── README.md
├── docs/
│   └── comparativo_inferencia_mistral.xlsx   # Comparativo de 3 configurações de inferência
├── images/
│   └── decision_flow.png           # Diagrama do fluxo de decisão LangGraph
├── models/
│   └── README.md                   # Model card — publicado no HuggingFace Hub
├── notebooks/
│   ├── 01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb
│   ├── 02.Fine-Tuning_TC_Fase3_8IADT.ipynb
│   └── 03.LangGraph_RAG_Pipeline_TC_Fase3_8IADT.ipynb
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- Git
- Conta no [Google Colab](https://colab.research.google.com) com acesso a GPU
- Conta no [HuggingFace](https://huggingface.co) com token de acesso (tipo *Read*)

O fine-tuning exige GPU L4 (22 GB VRAM). O pipeline RAG e a inferência rodam em GPU T4 (15 GB VRAM).

---

## Instalação local (dependências)

```bash
# 1. Clonar o repositório
git clone https://github.com/rodrigoaraujorosa/fiap-ia-devs-8iadt-fase3-tech-challenge.git
cd fiap-ia-devs-8iadt-fase3-tech-challenge

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # Linux / macOS

# 3. Instalar dependências
pip install -r requirements.txt
```

---

## Notebooks

O projeto possui três notebooks executados no Google Colab, cada um com uma responsabilidade distinta:

### Notebook 01 — Preprocessamento do Dataset

**`01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb`** | Google Colab GPU T4 | ~12 h 10 min

Processa o subconjunto PQA-A do dataset **PubMedQA** (211.269 pares QA biomédicos em inglês):

1. Carrega `ori_pqaa.json` do Google Drive
2. Extrai os campos `QUESTION` e `LONG_ANSWER`
3. Traduz EN→PT-BR com o modelo `Helsinki-NLP/opus-mt-tc-big-en-pt`
4. Aplica pós-processamento com `medical_dict.json` (131 termos clínicos)
5. Formata no padrão **Alpaca** (`instruction` / `input` / `output`)
6. Salva o resultado em `preprocessed/train_data.json`

> O arquivo original `ori_pqaa.json` (509 MB) não está no repositório. Obtê-lo em [pubmedqa.github.io](https://pubmedqa.github.io/).

---

### Notebook 02 — Fine-Tuning

**`02.Fine-Tuning_TC_Fase3_8IADT.ipynb`** | Google Colab GPU L4 (22 GB VRAM) | ~17 h

Fine-tuning do **Mistral-7B-Instruct-v0.2** com QLoRA sobre os dados preprocessados:

| Parâmetro | Valor |
|---|---|
| Técnica | QLoRA — 4-bit NF4 + double quantization |
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Batch efetivo | 8 (batch=1 + gradient_accumulation=8) |
| Max sequence length | 256 tokens |
| Optimizer | paged_adamw_8bit |
| Learning rate | 2e-4 |
| Épocas | 1 |
| Formato de prompt | Mistral Instruct `[INST] ... [/INST]` |

Ao final do treinamento, o adaptador LoRA é salvo no Google Drive e pode ser carregado em uma sessão limpa para inferência (evitando OOM por duplo carregamento do modelo base).

---

### Notebook 03 — Pipeline RAG LangGraph

**`03.LangGraph_RAG_Pipeline_TC_Fase3_8IADT.ipynb`** | Google Colab GPU T4/L4

Pipeline completo e autocontido. Passos:

1. Instala dependências (`langchain`, `langgraph`, `chromadb`, `gradio`, etc.)
2. Monta Google Drive — persiste ChromaDB e logs de auditoria entre sessões
3. Carrega prontuários (`prontuarios.json`) e indexa no **ChromaDB**
4. Carrega o modelo fine-tunado + adaptador LoRA em 4-bit
5. Define o `GraphState` e os 9 nós do grafo LangGraph
6. Compila e visualiza o grafo
7. Executa consultas de teste (paciente específico, genérica, solicitação de prescrição)
8. Sobe interface **Gradio** com URL pública válida por 72 h

#### Configurar token HuggingFace no Colab

O notebook carrega o modelo base diretamente do Hub, o que requer autenticação:

1. Gere um token em **huggingface.co → Settings → Access Tokens → New token** (tipo: *Read*)
2. No Colab, clique no ícone **Secrets** (🔑) no painel esquerdo
3. Adicione: Nome `HF_TOKEN` — Valor: seu token — ative **Notebook access**

#### Passos de execução

1. Abra o notebook no Google Colab
2. Selecione **Ambiente de execução → Alterar tipo → T4 GPU**
3. Execute a célula de instalação de dependências
4. **Reinicie a sessão** (Ambiente de execução → Reiniciar sessão)
5. Execute as demais células em ordem
6. A URL pública do Gradio aparecerá na saída da última célula

---

## Modelo Fine-Tunado

O adaptador LoRA está publicado no HuggingFace Hub:

**[rodrigoaraujorosa/mistral-7b-assistente-hospitalar-v1](https://huggingface.co/rodrigoaraujorosa/mistral-7b-assistente-hospitalar-v1)**

| Parâmetro | Valor |
|---|---|
| Modelo base | `mistralai/Mistral-7B-Instruct-v0.2` |
| Técnica | QLoRA (4-bit NF4 + double quantization) |
| LoRA rank | 64 |
| Dataset | 211.269 exemplos médicos em PT-BR |
| Infraestrutura | GPU L4 (22 GB VRAM) — Google Colab |
| Tempo de treinamento | ~17 horas |

**Comparativo de configurações de inferência** (5 perguntas clínicas avaliadas):

| Configuração | Avaliação |
|---|---|
| `max_new=512, temp=0.7, top_p=0.9, rep=1.1` | ⭐⭐⭐☆☆ 3.2/5 (64%) |
| `max_new=1024, temp=0.4, top_p=0.9, rep=1.15` | ⭐⭐☆☆☆ 2.8/5 (56%) |
| **`max_new=1024, temp=0.6, top_p=0.85, rep=1.1`** | **⭐⭐⭐⭐☆ 4.0/5 (80%) ← recomendado** |

---

## Funcionalidades de Segurança

- **Recusa de prescrições**: consultas que solicitam prescrição médica são recusadas pelo nó `recusar_prescricao` antes de acionar a LLM
- **Identificação fuzzy de paciente**: o nó `identificar_paciente` faz correspondência aproximada do nome para evitar falsos negativos por variação ortográfica
- **Fallback para paciente inexistente**: quando nenhum prontuário é encontrado, o pipeline usa RAG genérico e a resposta informa ausência de dados do paciente
- **Alertas clínicos automáticos**: o nó `buscar_alertas` verifica exames pendentes e alterados via metadados do ChromaDB e os inclui na resposta estruturada
- **Aviso de validação humana**: toda resposta com sugestão clínica inclui aviso obrigatório de validação por profissional habilitado
- **Auditoria completa**: o nó `registrar_log` persiste todas as interações em JSON Lines com timestamp ISO 8601 no Google Drive

---

## Autores

- Rodrigo de Araújo Rosa
- Elias Maximiano da Silva
- Fábia Gomes de Jesus
- Danilo Pereira

---

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).

O dataset PubMedQA utilizado no treinamento deve ser citado conforme:

> Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. *Proceedings of EMNLP-IJCNLP 2019*, pp. 2567–2577.
