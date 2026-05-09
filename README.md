# Assistente Virtual Médico Hospitalar

Sistema de inteligência artificial para auxílio a médicos em condutas clínicas, desenvolvido como **Tech Challenge — Fase 3** do curso de Inteligência Artificial para DEVs (8IADT).

O sistema combina um modelo de linguagem fine-tunado em dados médicos em português brasileiro com um pipeline de recuperação aumentada por geração (RAG) sobre prontuários de pacientes, orquestrado por um fluxo de decisão LangGraph.

> ⚠️ **Uso exclusivamente acadêmico.** Este sistema não deve ser utilizado para diagnóstico, prescrição ou qualquer conduta clínica em pacientes reais.

---

## Arquitetura

```
Consulta do médico (PT-BR)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Fluxo LangGraph (src/graph/)            │
│                                                     │
│  Receber → RAG (FAISS) → Verificar Exames           │
│                │                                    │
│         Pendentes/Alterados?                        │
│          Sim ↓        Não ↓                         │
│       Gerar Alerta → Gerar Resposta (LLM)           │
│                          ↓                         │
│                   Formatar Resposta                 │
└─────────────────────────────────────────────────────┘
        │
        ▼
Resposta + Fontes + Alertas Clínicos
```

**Componentes principais:**

| Componente | Tecnologia |
|---|---|
| Modelo LLM | Mistral-7B-Instruct-v0.2 + adaptador LoRA (QLoRA 4-bit) |
| Fine-tuning | PEFT + TRL SFTTrainer — executado no Google Colab |
| Vector Store (RAG) | FAISS in-memory |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Orquestração | LangChain + LangGraph |
| Prontuários | JSON local (20 registros fictícios) |
| Logging | Python `TimedRotatingFileHandler` — rotação diária, 30 dias. No Colab, salvo em `MyDrive/assistente-medico/logs/` |

---

## Estrutura do Projeto

```
├── data/
│   ├── preprocessed/               # Dados traduzidos e formatados para treino
│   ├── medical_dict.json           # Dicionário de termos médicos EN→PT-BR
│   ├── prontuarios.json            # Base de prontuários fictícios (20 pacientes)
│   └── README.md
├── docs/                           # Relatório técnico e comparativos de inferência
├── images/                         # Diagramas do fluxo de decisão
├── logs/
│   └── audit.log                   # Log de auditoria (JSON Lines, rotação diária)
├── models/
│   └── README.md                   # Documentação do modelo no HuggingFace Hub
├── notebooks/
│   ├── 01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb
│   ├── 02.Fine-Tuning_TC_Fase3_8IADT.ipynb
│   ├── 03.DemoGradio_Assistente_Hospitalar_TC_Fase3_8IADT.ipynb
│   └── 04.Pipeline_RAG_Assistente_Hospitalar_TC_Fase3_8IADT.ipynb
├── src/
│   ├── database/                   # Gerenciamento de prontuários e índice FAISS
│   ├── graph/                      # Fluxo de decisão LangGraph
│   ├── pipeline/                   # Pipeline RAG LangChain
│   └── utils/                      # Logger, modelos de dados, exceções
├── tests/
│   ├── integration/
│   ├── property/
│   ├── smoke/
│   └── unit/
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Pré-requisitos

- Python 3.10+
- Git

Para execução do pipeline completo com o modelo fine-tunado, é necessária uma GPU com pelo menos 15 GB de VRAM (ex.: NVIDIA T4 ou L4 no Google Colab).

---

## Instalação

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

O projeto possui quatro notebooks, cada um com uma responsabilidade distinta:

| Notebook | Descrição | Ambiente |
|---|---|---|
| `01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb` | Preprocessamento, tradução EN→PT-BR e curadoria do dataset PubMedQA | Google Colab (GPU T4) |
| `02.Fine-Tuning_TC_Fase3_8IADT.ipynb` | Fine-tuning do Mistral-7B com QLoRA sobre os dados preprocessados | Google Colab (GPU L4) |
| `03.DemoGradio_Assistente_Hospitalar_TC_Fase3_8IADT.ipynb` | Interface Gradio para consultas diretas ao modelo fine-tunado (sem RAG) | Google Colab (GPU T4/L4) |
| `04.Pipeline_RAG_Assistente_Hospitalar_TC_Fase3_8IADT.ipynb` | Pipeline completo: modelo + RAG + prontuários + alertas + interface Gradio. Logs de auditoria salvos no Google Drive | Google Colab (GPU T4/L4) |

### Executar o pipeline RAG no Colab

#### Pré-requisito: conta e token no HuggingFace

A seção 6 do notebook carrega o modelo `mistralai/Mistral-7B-Instruct-v0.2` diretamente do HuggingFace Hub, o que requer autenticação.

1. Crie uma conta em [huggingface.co](https://huggingface.co) (gratuita)
2. Gere um token de acesso em **Settings → Access Tokens → New token** (tipo: *Read*)
3. No Google Colab, adicione o token como secret:
   - Clique no ícone 🔑 **Secrets** no painel esquerdo
   - Clique em **+ Add new secret**
   - Nome: `HF_TOKEN` — Valor: seu token gerado
   - Ative a opção **Notebook access**

#### Passos de execução

1. Abra o notebook `04.Pipeline_RAG_Assistente_Hospitalar_TC_Fase3_8IADT.ipynb` no Google Colab
2. Selecione uma GPU em **Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU**
3. Execute a **seção 3** (Instalar Dependências)
4. **Reinicie a sessão**: Ambiente de execução → Reiniciar sessão
5. Na célula da **seção 4**, substitua `REPO_URL` pela URL deste repositório
6. Execute as demais células em ordem — o Google Drive será montado automaticamente na seção 2 para persistir os logs de auditoria entre sessões
7. A interface Gradio gerará uma URL pública válida por 72 horas

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
| Melhor configuração de inferência | `max_new_tokens=1024, temp=0.6, top_p=0.85, rep_penalty=1.1` → **4.0/5 (80%)** |

---

## Executar os Testes

```bash
source .venv/Scripts/activate
pytest tests/ -v
```

Os testes cobrem: banco de dados de prontuários, logger de auditoria, pipeline RAG e propriedades de corretude.

---

## Funcionalidades de Segurança

- **Recusa de prescrições**: consultas que solicitam prescrição médica direta são recusadas antes de invocar a LLM
- **Aviso de validação humana**: toda resposta com sugestão clínica inclui aviso obrigatório de validação por profissional habilitado
- **Fallback para paciente inexistente**: quando nenhum prontuário é encontrado, a resposta informa explicitamente a ausência de dados
- **Auditoria completa**: todas as interações, transições de grafo e erros são registrados em `logs/audit.log` com timestamp ISO 8601

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
