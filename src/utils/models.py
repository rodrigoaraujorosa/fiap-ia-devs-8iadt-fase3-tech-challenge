"""
Dataclasses e enums compartilhados entre os módulos do Assistente Virtual Médico Hospitalar.

Este módulo centraliza todos os modelos de dados utilizados pelos módulos de
pipeline RAG, fluxo LangGraph, banco de dados e logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExameStatus(str, Enum):
    """Status possíveis de um exame médico."""

    PENDENTE = "pendente"
    CONCLUIDO = "concluido"
    ALTERADO = "alterado"
    CANCELADO = "cancelado"


class AlertType(str, Enum):
    """Tipo de alerta clínico gerado pelo sistema."""

    EXAME_PENDENTE = "EXAME_PENDENTE"
    EXAME_ALTERADO = "EXAME_ALTERADO"


class AlertPriority(str, Enum):
    """Prioridade de um alerta clínico."""

    ALTA = "ALTA"
    MEDIA = "MEDIA"
    BAIXA = "BAIXA"


# ---------------------------------------------------------------------------
# Modelos de prontuário
# ---------------------------------------------------------------------------


@dataclass
class Exame:
    """Representa um exame médico associado a um prontuário de paciente."""

    id: str
    nome: str
    data_solicitacao: str
    data_resultado: Optional[str]
    status: ExameStatus
    resultado: Optional[str]
    observacoes_medico: str

    def __post_init__(self) -> None:
        # Garante que status seja sempre um ExameStatus, mesmo quando recebido como str
        if isinstance(self.status, str):
            self.status = ExameStatus(self.status)


@dataclass
class Prontuario:
    """Registro completo de um paciente na base de prontuários."""

    id: str
    nome: str
    data_nascimento: str
    sexo: str
    exames: list[Exame]
    diagnosticos: list[str]
    medicamentos_em_uso: list[str]
    observacoes_gerais: str
    ultima_atualizacao: str

    def __post_init__(self) -> None:
        # Converte dicts para Exame quando necessário (ex.: carregamento via JSON)
        exames_convertidos: list[Exame] = []
        for exame in self.exames:
            if isinstance(exame, dict):
                exames_convertidos.append(Exame(**exame))
            else:
                exames_convertidos.append(exame)
        self.exames = exames_convertidos


# ---------------------------------------------------------------------------
# Modelos de resposta do pipeline
# ---------------------------------------------------------------------------


@dataclass
class ClinicalAlert:
    """Alerta clínico gerado para exames pendentes ou alterados."""

    tipo: AlertType
    prioridade: AlertPriority
    descricao: str
    exames_afetados: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.tipo, str):
            self.tipo = AlertType(self.tipo)
        if isinstance(self.prioridade, str):
            self.prioridade = AlertPriority(self.prioridade)


@dataclass
class ResponseSources:
    """Fontes de informação utilizadas na geração de uma resposta médica.

    Separa explicitamente os dados recuperados via RAG do conhecimento
    interno do modelo fine-tunado, atendendo ao requisito de explainability.
    """

    dados_prontuario: list[str] = field(default_factory=list)
    """Campos do prontuário utilizados como fonte (ex.: ['exames', 'diagnosticos'])."""

    conhecimento_modelo: bool = False
    """True quando a resposta utiliza conhecimento do modelo fine-tunado."""

    paciente_encontrado: bool = False
    """True quando um prontuário correspondente foi encontrado na base."""


@dataclass
class MedicalResponse:
    """Resposta completa gerada pelo assistente médico para uma consulta clínica."""

    resposta: str
    """Texto da resposta em Português Brasileiro."""

    fontes: ResponseSources
    """Fontes estruturadas utilizadas na geração da resposta."""

    alertas: list[ClinicalAlert]
    """Lista de alertas clínicos identificados durante o processamento."""

    timestamp: str
    """Timestamp ISO 8601 da geração da resposta."""

    consulta_original: str
    """Consulta original submetida pelo médico."""

    def __post_init__(self) -> None:
        if isinstance(self.fontes, dict):
            self.fontes = ResponseSources(**self.fontes)
        alertas_convertidos: list[ClinicalAlert] = []
        for alerta in self.alertas:
            if isinstance(alerta, dict):
                alertas_convertidos.append(ClinicalAlert(**alerta))
            else:
                alertas_convertidos.append(alerta)
        self.alertas = alertas_convertidos


# ---------------------------------------------------------------------------
# Modelos de logging e auditoria
# ---------------------------------------------------------------------------


@dataclass
class LogEntry:
    """Entrada de log de auditoria registrada pelo AuditLogger.

    Cobre os tipos de evento: INTERACTION, GRAPH_TRANSITION, ERROR.
    """

    timestamp: str
    """Timestamp ISO 8601 do evento."""

    event_type: str
    """Tipo do evento: INTERACTION | GRAPH_TRANSITION | ERROR."""

    consulta: Optional[str] = None
    """Consulta submetida pelo médico (presente em eventos INTERACTION)."""

    resposta: Optional[str] = None
    """Resposta gerada pelo assistente (presente em eventos INTERACTION)."""

    fontes: Optional[dict] = None
    """Fontes estruturadas serializadas como dict (presente em eventos INTERACTION)."""

    node_from: Optional[str] = None
    """Nó de origem da transição (presente em eventos GRAPH_TRANSITION)."""

    node_to: Optional[str] = None
    """Nó de destino da transição (presente em eventos GRAPH_TRANSITION)."""

    state_summary: Optional[dict] = None
    """Resumo do estado intermediário (presente em eventos GRAPH_TRANSITION)."""

    error_message: Optional[str] = None
    """Mensagem de erro (presente em eventos ERROR)."""

    stack_trace: Optional[str] = None
    """Stack trace completo (presente em eventos ERROR)."""
