# Dados do Assistente Virtual Médico Hospitalar

Este diretório contém os dados utilizados no projeto do Assistente Virtual Médico Hospitalar, organizados em subdiretórios conforme descrito abaixo.

---

## Estrutura de Diretórios

```
data/
├── raw/
│   └── ori_pqaa.json          # Dataset médico de perguntas e respostas (EN)
├── preprocessed/
│   └── train_data.json        # Dados preprocessados e traduzidos (gerado pelo pipeline)
├── medical_dict.json          # Dicionário de termos médicos EN→PT-BR
├── prontuarios.json           # Base de prontuários fictícios de pacientes
└── README.md                  # Este arquivo
```

---

## Descrição dos Arquivos

### `raw/ori_pqaa.json`

Subconjunto do dataset **PubMedQA** — uma coleção de pares de perguntas e respostas biomédicas derivadas de abstracts do PubMed. O PubMedQA foi criado para a tarefa de resposta a perguntas de pesquisa com as opções *yes/no/maybe* (ex.: *"Do preoperative statins reduce atrial fibrillation after coronary artery bypass grafting?"*) usando os abstracts correspondentes como contexto.

O arquivo `ori_pqaa.json` corresponde às **211.269 instâncias geradas artificialmente** (subconjunto *PQA-A*, artificially generated) do dataset completo, que totaliza aproximadamente 273,5 mil instâncias entre exemplos rotulados por especialistas (1k), não rotulados (61,2k) e gerados artificialmente (211,3k).

- **Fonte**: [PubMedQA — A Dataset for Biomedical Research Question Answering](https://pubmedqa.github.io/)
- **Repositório oficial**: [github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)
- **Formato**: JSON com objetos contendo os campos `QUESTION`, `LONG_ANSWER` e `CONTEXTS`
- **Idioma original**: Inglês
- **Uso neste projeto**: Entrada do pipeline de preprocessamento (`src/preprocessing/`), onde os campos `QUESTION` e `LONG_ANSWER` são extraídos, traduzidos para Português Brasileiro e formatados para fine-tuning da LLM

#### Citação

Se você utilizar este dataset em pesquisas ou trabalhos acadêmicos, cite o trabalho original conforme abaixo:

```bibtex
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
```

> Jin, Q., Dhingra, B., Liu, Z., Cohen, W., & Lu, X. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. *Proceedings of EMNLP-IJCNLP 2019*, pp. 2567–2577.

### `preprocessed/train_data.json`

Arquivo gerado automaticamente pelo módulo de preprocessamento (`src/preprocessing/preprocessor.py`). Contém os dados traduzidos para Português Brasileiro, anonimizados, deduplicados e formatados para fine-tuning.

- **Formato**: JSON com array de objetos `{"instruction": "...", "input": "", "output": "..."}`
- **Idioma**: Português Brasileiro
- **Geração**: Executar o pipeline de preprocessamento no notebook `notebooks/fine_tuning_colab.ipynb`

### `medical_dict.json`

Dicionário de mapeamento de termos médicos do inglês para o Português Brasileiro, utilizado para corrigir traduções automáticas de terminologia clínica crítica após a tradução pelo modelo `Helsinki-NLP/opus-mt-en-ROMANCE`.

- **Versão**: 1.0
- **Total de termos**: 131 termos clínicos
- **Cobertura**: Cardiologia, pneumologia, nefrologia, endocrinologia, hematologia, neurologia, gastroenterologia, infectologia, procedimentos diagnósticos e terapêuticos
- **Exemplos de mapeamentos**:
  - `"chest pain"` → `"dor torácica"`
  - `"dyspnea"` → `"dispneia"`
  - `"myocardial infarction"` → `"infarto do miocárdio"`
  - `"complete blood count"` → `"hemograma completo"`
  - `"ICU"` → `"UTI"`

### `prontuarios.json`

Base de dados de prontuários fictícios de pacientes, utilizada pelo pipeline RAG para recuperação de informações contextuais durante as consultas ao assistente.

- **Versão**: 1.0
- **Total de registros**: 20 pacientes fictícios
- **Idioma**: Português Brasileiro

---

## Origem dos Dados Fictícios (`prontuarios.json`)

Os prontuários foram **inteiramente gerados de forma sintética** para fins acadêmicos. Nenhum dado real de paciente foi utilizado em nenhuma etapa do processo.

### Processo de Geração

Os registros foram gerados por Inteligência Artificial seguindo os critérios abaixo:

1. **Alinhamento semântico com o dataset de treinamento**: As condições clínicas representadas nos prontuários foram selecionadas para corresponder às categorias temáticas presentes no dataset `ori_pqaa.json`, garantindo coerência entre os dados de treinamento da LLM e os dados recuperados via RAG. As condições cobertas incluem:
   - Anemia ferropriva e anemia por doença crônica (P001, P011, P016)
   - Hipertensão arterial sistêmica (P002, P003, P004, P006, P010, P011, P018, P020)
   - Diabetes mellitus tipo 2 com complicações (P002, P011)
   - Dor torácica isquêmica e infarto agudo do miocárdio (P004, P018, P020)
   - Dispneia por diversas etiologias (P001, P003, P004, P005, P010, P012, P013)
   - Insuficiência cardíaca (P003), DPOC (P005), asma (P013)
   - AVC isquêmico (P009), sepse (P008), pneumonia (P012)
   - Fibrilação atrial (P010), cirrose hepática (P006), hepatite C (P019)
   - Doença renal crônica (P011), hipertireoidismo/Doença de Graves (P007)
   - Pancreatite aguda (P014), epilepsia (P017), dislipidemia/DAC (P020)

2. **Estrutura clínica realista**: Cada prontuário foi elaborado com dados clinicamente coerentes — valores laboratoriais, achados de imagem, diagnósticos e medicamentos compatíveis com as condições representadas — para simular registros hospitalares reais sem utilizar dados de pacientes reais.

3. **Cobertura de todos os status de exame**: Os registros foram distribuídos para incluir exemplos de todos os quatro status possíveis de exame: `pendente`, `concluido`, `alterado` e `cancelado`, viabilizando o teste completo do fluxo de decisão LangGraph.

4. **Nomes fictícios**: Todos os nomes de pacientes são fictícios, gerados aleatoriamente sem referência a pessoas reais.

5. **Datas fictícias**: Todas as datas de nascimento, datas de solicitação e datas de resultado são fictícias, situadas em 2024 para fins de consistência temporal.

---

## Limitações de Uso Acadêmico

> **⚠️ AVISO IMPORTANTE**: Este projeto e todos os seus dados são destinados **exclusivamente a fins acadêmicos e de pesquisa**. O uso em ambiente clínico real é expressamente contraindicado.

### Limitações dos Dados Sintéticos

1. **Não representam pacientes reais**: Os prontuários são inteiramente fictícios. Qualquer semelhança com pacientes reais é coincidência.

2. **Simplificação clínica**: Os registros foram simplificados para fins didáticos e não refletem a complexidade e variabilidade de prontuários hospitalares reais.

3. **Escala reduzida**: A base contém apenas 20 registros, insuficiente para representar a diversidade epidemiológica de uma população real.

4. **Ausência de validação clínica**: Os dados sintéticos não foram revisados por profissionais de saúde para validação clínica formal.

### Limitações do Dataset de Treinamento

1. **Idioma original**: O dataset PubMedQA (`ori_pqaa.json`) está em inglês e passa por tradução automática, o que pode introduzir imprecisões terminológicas não corrigidas pelo dicionário médico.

2. **Natureza das instâncias artificiais**: O subconjunto PQA-A utilizado contém instâncias geradas artificialmente a partir de abstracts do PubMed, não revisadas por especialistas. A qualidade das respostas pode variar em relação ao subconjunto rotulado por especialistas (PQA-L, 1k instâncias).

3. **Escopo biomédico de pesquisa**: O PubMedQA foi originalmente projetado para resposta a perguntas de pesquisa biomédica (yes/no/maybe), não para assistência clínica direta. O uso para fine-tuning de um assistente hospitalar representa uma adaptação de domínio que pode introduzir limitações.

4. **Desatualização**: O conhecimento médico contido no dataset reflete o estado da arte na época de sua criação (2019) e pode não incluir diretrizes clínicas mais recentes.

### Restrições de Uso

- **Proibido** o uso do sistema para diagnóstico, prescrição ou qualquer conduta clínica em pacientes reais.
- **Proibido** o uso comercial sem autorização expressa.
- **Obrigatório** citar este repositório em trabalhos acadêmicos que utilizem estes dados.
- **Obrigatório** manter os avisos de validação humana em qualquer derivação do sistema.

---

## Reprodutibilidade

Para reproduzir o pipeline completo de preprocessamento e geração dos dados de treinamento:

1. Certifique-se de que o arquivo `data/raw/ori_pqaa.json` está presente
2. Ative o ambiente virtual: `source .venv/Scripts/activate`
3. Execute o notebook `notebooks/fine_tuning_colab.ipynb` no Google Colab, ou execute localmente:

```bash
python -m src.preprocessing.preprocessor \
  --input data/raw/ori_pqaa.json \
  --output data/preprocessed/
```

O arquivo `data/preprocessed/train_data.json` será gerado automaticamente com o relatório de curadoria registrado em `logs/audit.log`.

---

*Dados gerados para o projeto acadêmico "Assistente Virtual Médico Hospitalar". Uso restrito a fins de pesquisa e ensino.*
