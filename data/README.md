# Dados do Assistente Virtual Médico Hospitalar

Este diretório contém os dados utilizados no projeto do Assistente Virtual Médico Hospitalar, organizados em subdiretórios conforme descrito abaixo.

---

## Estrutura de Diretórios

```
data/
├── preprocessed/
│   └── train_data.json        # Dados preprocessados e traduzidos (PT-BR, formato Alpaca)
├── medical_dict.json          # Dicionário de termos médicos EN→PT-BR
├── prontuarios.json           # Base de prontuários fictícios de pacientes
└── README.md                  # Este arquivo
```

> O dataset original `ori_pqaa.json` (509 MB) **não está incluído no repositório** por exceder o limite de tamanho do GitHub. Consulte a seção [Reprodutibilidade](#reprodutibilidade) para instruções de obtenção.

---

## Descrição dos Arquivos

### `preprocessed/train_data.json`

Arquivo gerado pelo notebook `01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb` no Google Colab. Contém os dados traduzidos para Português Brasileiro, deduplicados e formatados para fine-tuning no formato Alpaca.

- **Formato**: JSON com array de objetos `{"instruction": "...", "input": "", "output": "..."}`
- **Idioma**: Português Brasileiro
- **Geração**: Executar o notebook `notebooks/01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb` no Google Colab (ver seção Reprodutibilidade)
- **Tempo de geração**: ~12 horas e 10 minutos em GPU T4

### `medical_dict.json`

Dicionário de mapeamento de termos médicos do inglês para o Português Brasileiro, utilizado para corrigir traduções automáticas de terminologia clínica crítica após a tradução pelo modelo `Helsinki-NLP/opus-mt-tc-big-en-pt`.

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

## Dataset Original (não incluído no repositório)

O fine-tuning foi realizado com o subconjunto **PQA-A** do dataset **PubMedQA** — 211.269 pares de perguntas e respostas biomédicas gerados artificialmente a partir de abstracts do PubMed, originalmente em inglês.

O arquivo `ori_pqaa.json` (509 MB) não está incluído neste repositório por exceder o limite de tamanho do GitHub. Para reproduzir o preprocessamento, obtenha o arquivo diretamente na fonte oficial:

- **Fonte**: [pubmedqa.github.io](https://pubmedqa.github.io/)
- **Repositório oficial**: [github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa)
- **Formato**: JSON com campos `QUESTION`, `LONG_ANSWER` e `CONTEXTS` por registro
- **Idioma original**: Inglês

#### Citação

```bibtex
@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
```

---

## Reprodutibilidade

O preprocessamento foi executado integralmente no Google Colab via notebook, utilizando o Google Drive como armazenamento intermediário. Não há execução local desse pipeline.

### Pré-requisitos

- Conta Google com Google Drive disponível
- Sessão do Google Colab com GPU (T4 recomendada)
- Arquivo `ori_pqaa.json` salvo no Google Drive em `MyDrive/fase3-data/raw/`

O dataset original pode ser obtido em: [github.com/pubmedqa/pubmedqa](https://github.com/pubmedqa/pubmedqa) (arquivo `ori_pqaa.json` do subconjunto PQA-A).

### Passos

1. Faça upload do arquivo `ori_pqaa.json` para o Google Drive no caminho:
   ```
   MyDrive/fase3-data/raw/ori_pqaa.json
   ```

2. Abra o notebook no Google Colab:
   [`notebooks/01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb`](../notebooks/01.PreProcessamento_DataSet_TC_Fase3_8IADT.ipynb)

3. Selecione uma GPU em **Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU**

4. Execute todas as células em ordem. O notebook irá:
   - Montar o Google Drive automaticamente
   - Carregar `ori_pqaa.json` de `MyDrive/fase3-data/raw/`
   - Traduzir os 211.269 registros EN→PT-BR usando `Helsinki-NLP/opus-mt-tc-big-en-pt`
   - Salvar checkpoints intermediários a cada 1.000 registros (permite retomada em caso de interrupção)
   - Salvar o resultado final em `MyDrive/fase3-data/preprocessed/train_data.json`

5. Ao final, copie o arquivo gerado para `data/preprocessed/train_data.json` no repositório local, se necessário.

> **Tempo estimado**: ~12 horas em GPU T4. O notebook suporta retomada automática via checkpoint — se a sessão for interrompida, basta executar novamente que o progresso é preservado.

---

*Dados gerados para o projeto acadêmico "Assistente Virtual Médico Hospitalar". Uso restrito a fins de pesquisa e ensino.*
