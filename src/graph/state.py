"""
Definição do estado compartilhado do grafo LangGraph.

O ``GraphState`` é um ``TypedDict`` que representa o estado transmitido
entre todos os nós do grafo de decisão clínica. Cada nó lê e/ou atualiza
campos específicos deste estado, garantindo rastreabilidade completa do
fluxo de processamento de uma consulta médica.
"""

from __future__ import annotations

from typing import Optional

from typing_extensions import TypedDict

from src.utils.models import ClinicalAlert, Exame, LogEntry, MedicalResponse


class GraphState(TypedDict, total=False):
    """Estado compartilhado entre todos os nós do grafo LangGraph.

    Cada campo é populado progressivamente à medida que a consulta avança
    pelos nós do grafo. Campos opcionais (``total=False``) podem estar
    ausentes no estado inicial e são preenchidos pelos nós correspondentes.

    Fields:
        consulta: Consulta original submetida pelo médico em PT-BR.
            Populado pelo nó ``node_receber_consulta``.
        prontuarios_recuperados: Lista de dicionários representando os
            documentos recuperados via RAG (FAISS). Cada dict contém
            ``page_content`` e ``metadata``. Populado por
            ``node_recuperar_prontuario``.
        paciente_identificado: Nome do paciente identificado na consulta,
            ou ``None`` se nenhum paciente foi mencionado. Populado por
            ``node_receber_consulta``.
        exames_pendentes: Lista de exames com status ``pendente`` para o
            paciente identificado. Populado por ``node_verificar_exames``.
        exames_alterados: Lista de exames com status ``alterado`` para o
            paciente identificado. Populado por ``node_verificar_exames``.
        alertas: Lista de alertas clínicos gerados para exames pendentes
            e/ou alterados. Populado por ``node_gerar_alerta``.
        resposta_llm: Texto bruto gerado pela LLM fine-tunada antes da
            formatação final. Populado por ``node_gerar_resposta``.
        resposta_final: Resposta formatada com fontes estruturadas e
            alertas clínicos. Populado por ``node_formatar_resposta``.
        erro: Mensagem de erro genérica caso ocorra falha em qualquer nó.
            Populado pelo tratamento de exceções dos nós.
        log_entries: Lista de entradas de log geradas durante a execução
            do grafo. Acumulada por todos os nós para rastreabilidade.
    """

    consulta: str
    prontuarios_recuperados: list[dict]
    paciente_identificado: Optional[str]
    exames_pendentes: list[Exame]
    exames_alterados: list[Exame]
    alertas: list[ClinicalAlert]
    resposta_llm: str
    resposta_final: MedicalResponse
    erro: Optional[str]
    log_entries: list[LogEntry]
