"""
Implementação dos nós do grafo LangGraph do Assistente Virtual Médico Hospitalar.

Cada função de nó recebe o ``GraphState`` atual, executa sua responsabilidade
específica e retorna um dicionário com os campos do estado que foram atualizados.
Todos os nós registram suas transições no ``AuditLogger`` e capturam erros como
``GraphExecutionError``.

Nós implementados:
    1. ``node_receber_consulta``   — valida a consulta e identifica o paciente
    2. ``node_recuperar_prontuario`` — executa RAG via FAISS
    3. ``node_verificar_exames``   — filtra exames pendentes e alterados
    4. ``node_gerar_alerta``       — gera alertas clínicos (nó condicional)
    5. ``node_gerar_resposta``     — invoca a LLM fine-tunada
    6. ``node_formatar_resposta``  — formata a resposta final com fontes
"""

from __future__ import annotations

import re
import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.documents import Document

from src.graph.state import GraphState
from src.pipeline.prompt_templates import (
    GENERIC_ERROR_MESSAGE,
    HUMAN_VALIDATION_WARNING,
    MEDICAL_RAG_TEMPLATE,
    MEDICAL_SYSTEM_PROMPT,
    PATIENT_NOT_FOUND_PREFIX,
)
from src.pipeline.response_formatter import ResponseFormatter
from src.utils.exceptions import GraphExecutionError
from src.utils.models import (
    AlertPriority,
    AlertType,
    ClinicalAlert,
    ExameStatus,
    LogEntry,
    MedicalResponse,
    Prontuario,
    ResponseSources,
)


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Retorna o timestamp atual em formato ISO 8601 UTC."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_log_entry(event_type: str, **kwargs: Any) -> LogEntry:
    """Cria uma ``LogEntry`` com timestamp atual e campos fornecidos."""
    return LogEntry(
        timestamp=_now_iso(),
        event_type=event_type,
        consulta=kwargs.get("consulta"),
        resposta=kwargs.get("resposta"),
        fontes=kwargs.get("fontes"),
        node_from=kwargs.get("node_from"),
        node_to=kwargs.get("node_to"),
        state_summary=kwargs.get("state_summary"),
        error_message=kwargs.get("error_message"),
        stack_trace=kwargs.get("stack_trace"),
    )


def _extract_patient_name(consulta: str) -> Optional[str]:
    """Tenta extrair o nome do paciente mencionado na consulta.

    Usa heurísticas baseadas em padrões comuns em consultas médicas em PT-BR:
    - "paciente <Nome>"
    - "do paciente <Nome>"
    - "da paciente <Nome>"
    - "para <Nome>"
    - "de <Nome>"

    Args:
        consulta: Texto da consulta do médico.

    Returns:
        Nome do paciente identificado ou ``None`` se não encontrado.
    """
    patterns = [
        r"(?:paciente|do paciente|da paciente)\s+([A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ][a-záéíóúâêîôûãõàèìòùç]+(?:\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ][a-záéíóúâêîôûãõàèìòùç]+)*)",
        r"(?:para|de|sobre)\s+([A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ][a-záéíóúâêîôûãõàèìòùç]+(?:\s+[A-ZÁÉÍÓÚÂÊÎÔÛÃÕÀÈÌÒÙÇ][a-záéíóúâêîôûãõàèìòùç]+)+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, consulta)
        if match:
            return match.group(1).strip()
    return None


def _docs_to_dicts(docs: list[Document]) -> list[dict]:
    """Converte documentos LangChain para dicionários serializáveis."""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]


def _dicts_to_docs(dicts: list[dict]) -> list[Document]:
    """Converte dicionários de volta para documentos LangChain."""
    return [
        Document(
            page_content=d.get("page_content", ""),
            metadata=d.get("metadata", {}),
        )
        for d in dicts
    ]


# ---------------------------------------------------------------------------
# Nó 1: Receber Consulta
# ---------------------------------------------------------------------------


def node_receber_consulta(state: GraphState) -> dict:
    """Nó 1: Recebe e valida a consulta do médico.

    Responsabilidades:
    - Validar que a consulta não está vazia
    - Identificar o nome do paciente mencionado na consulta
    - Inicializar campos do estado para os nós subsequentes
    - Registrar entrada no log

    Args:
        state: Estado atual do grafo.

    Returns:
        Dicionário com campos atualizados: ``consulta``,
        ``paciente_identificado``, ``prontuarios_recuperados``,
        ``exames_pendentes``, ``exames_alterados``, ``alertas``,
        ``log_entries``.
    """
    logger = state.get("_logger")  # type: ignore[misc]
    log_entries: list[LogEntry] = list(state.get("log_entries", []))

    try:
        consulta = state.get("consulta", "").strip()

        if not consulta:
            raise GraphExecutionError(
                "Consulta vazia recebida.",
                context={"node": "node_receber_consulta"},
            )

        # Identificar paciente mencionado na consulta
        paciente_identificado = _extract_patient_name(consulta)

        # Registrar transição de entrada
        entry = _make_log_entry(
            "GRAPH_TRANSITION",
            node_from="START",
            node_to="node_receber_consulta",
            state_summary={
                "consulta_length": len(consulta),
                "paciente_identificado": paciente_identificado,
            },
        )
        log_entries.append(entry)

        if logger:
            logger.log_graph_transition(
                from_node="START",
                to_node="node_receber_consulta",
                state_summary={
                    "consulta_length": len(consulta),
                    "paciente_identificado": paciente_identificado,
                },
            )

        return {
            "consulta": consulta,
            "paciente_identificado": paciente_identificado,
            "prontuarios_recuperados": [],
            "exames_pendentes": [],
            "exames_alterados": [],
            "alertas": [],
            "log_entries": log_entries,
        }

    except GraphExecutionError:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        if logger:
            logger.log_error(exc, context={"node": "node_receber_consulta"})
        raise GraphExecutionError(
            f"Erro no nó de recebimento da consulta: {exc}",
            context={"node": "node_receber_consulta", "stack_trace": tb},
        ) from exc


# ---------------------------------------------------------------------------
# Nó 2: Recuperar Prontuário via RAG
# ---------------------------------------------------------------------------


def node_recuperar_prontuario(state: GraphState) -> dict:
    """Nó 2: Executa RAG para recuperar prontuários relevantes.

    Usa FAISS + embeddings multilinguais para busca semântica nos prontuários.
    Registra a transição no AuditLogger com o número de documentos recuperados.

    Args:
        state: Estado atual do grafo (requer ``consulta`` e ``_retriever``).

    Returns:
        Dicionário com ``prontuarios_recuperados`` e ``log_entries`` atualizados.
    """
    logger = state.get("_logger")  # type: ignore[misc]
    retriever = state.get("_retriever")  # type: ignore[misc]
    log_entries: list[LogEntry] = list(state.get("log_entries", []))
    consulta = state.get("consulta", "")

    try:
        docs: list[Document] = []

        if retriever is not None:
            docs = retriever.retrieve(consulta, k=3)

        prontuarios_recuperados = _docs_to_dicts(docs)

        entry = _make_log_entry(
            "GRAPH_TRANSITION",
            node_from="node_receber_consulta",
            node_to="node_recuperar_prontuario",
            state_summary={
                "documentos_recuperados": len(prontuarios_recuperados),
                "paciente_identificado": state.get("paciente_identificado"),
            },
        )
        log_entries.append(entry)

        if logger:
            logger.log_graph_transition(
                from_node="node_receber_consulta",
                to_node="node_recuperar_prontuario",
                state_summary={
                    "documentos_recuperados": len(prontuarios_recuperados),
                    "paciente_identificado": state.get("paciente_identificado"),
                },
            )

        return {
            "prontuarios_recuperados": prontuarios_recuperados,
            "log_entries": log_entries,
        }

    except GraphExecutionError:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        if logger:
            logger.log_error(
                exc,
                context={"node": "node_recuperar_prontuario", "consulta": consulta},
            )
        raise GraphExecutionError(
            f"Erro no nó de recuperação de prontuário: {exc}",
            context={
                "node": "node_recuperar_prontuario",
                "consulta": consulta,
                "stack_trace": tb,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Nó 3: Verificar Exames
# ---------------------------------------------------------------------------


def node_verificar_exames(state: GraphState) -> dict:
    """Nó 3: Verifica exames pendentes e alterados do paciente identificado.

    Popula ``exames_pendentes`` e ``exames_alterados`` no estado. Usa o
    ``ProntuarioDB`` para buscar o paciente pelo nome identificado na consulta.

    Args:
        state: Estado atual do grafo (requer ``paciente_identificado`` e ``_db``).

    Returns:
        Dicionário com ``exames_pendentes``, ``exames_alterados`` e
        ``log_entries`` atualizados.
    """
    logger = state.get("_logger")  # type: ignore[misc]
    db = state.get("_db")  # type: ignore[misc]
    log_entries: list[LogEntry] = list(state.get("log_entries", []))
    paciente_identificado = state.get("paciente_identificado")

    try:
        from src.utils.models import Exame

        exames_pendentes: list[Exame] = []
        exames_alterados: list[Exame] = []

        if db is not None and paciente_identificado:
            # Busca o prontuário pelo nome identificado
            prontuarios = db.search_by_name(paciente_identificado, fuzzy=True)
            if prontuarios:
                prontuario = prontuarios[0]
                exames_pendentes = db.get_pending_exams(prontuario.id)
                exames_alterados = db.get_altered_exams(prontuario.id)

        entry = _make_log_entry(
            "GRAPH_TRANSITION",
            node_from="node_recuperar_prontuario",
            node_to="node_verificar_exames",
            state_summary={
                "paciente_identificado": paciente_identificado,
                "exames_pendentes": len(exames_pendentes),
                "exames_alterados": len(exames_alterados),
            },
        )
        log_entries.append(entry)

        if logger:
            logger.log_graph_transition(
                from_node="node_recuperar_prontuario",
                to_node="node_verificar_exames",
                state_summary={
                    "paciente_identificado": paciente_identificado,
                    "exames_pendentes": len(exames_pendentes),
                    "exames_alterados": len(exames_alterados),
                },
            )

        return {
            "exames_pendentes": exames_pendentes,
            "exames_alterados": exames_alterados,
            "log_entries": log_entries,
        }

    except GraphExecutionError:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        if logger:
            logger.log_error(
                exc,
                context={
                    "node": "node_verificar_exames",
                    "paciente_identificado": paciente_identificado,
                },
            )
        raise GraphExecutionError(
            f"Erro no nó de verificação de exames: {exc}",
            context={
                "node": "node_verificar_exames",
                "paciente_identificado": paciente_identificado,
                "stack_trace": tb,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Nó 4: Gerar Alerta (condicional)
# ---------------------------------------------------------------------------


def node_gerar_alerta(state: GraphState) -> dict:
    """Nó 4 (condicional): Gera alertas clínicos para exames pendentes/alterados.

    Executado apenas quando ``exames_pendentes`` ou ``exames_alterados`` não
    estão vazios (controlado pela aresta condicional em ``edges.py``).

    Prioridades:
    - Exames ``alterados`` → ``AlertPriority.ALTA``
    - Exames ``pendentes`` → ``AlertPriority.MEDIA``

    Args:
        state: Estado atual do grafo.

    Returns:
        Dicionário com ``alertas`` e ``log_entries`` atualizados.
    """
    logger = state.get("_logger")  # type: ignore[misc]
    log_entries: list[LogEntry] = list(state.get("log_entries", []))
    exames_pendentes = state.get("exames_pendentes", [])
    exames_alterados = state.get("exames_alterados", [])
    paciente_identificado = state.get("paciente_identificado") or "paciente"

    try:
        alertas: list[ClinicalAlert] = []

        if exames_alterados:
            nomes_alterados = [e.nome for e in exames_alterados]
            alertas.append(
                ClinicalAlert(
                    tipo=AlertType.EXAME_ALTERADO,
                    prioridade=AlertPriority.ALTA,
                    descricao=(
                        f"Paciente {paciente_identificado} possui "
                        f"{len(exames_alterados)} exame(s) com resultado alterado "
                        "que requer(em) atenção imediata."
                    ),
                    exames_afetados=nomes_alterados,
                )
            )

        if exames_pendentes:
            nomes_pendentes = [e.nome for e in exames_pendentes]
            alertas.append(
                ClinicalAlert(
                    tipo=AlertType.EXAME_PENDENTE,
                    prioridade=AlertPriority.MEDIA,
                    descricao=(
                        f"Paciente {paciente_identificado} possui "
                        f"{len(exames_pendentes)} exame(s) pendente(s) "
                        "aguardando resultado."
                    ),
                    exames_afetados=nomes_pendentes,
                )
            )

        entry = _make_log_entry(
            "GRAPH_TRANSITION",
            node_from="node_verificar_exames",
            node_to="node_gerar_alerta",
            state_summary={
                "alertas_gerados": len(alertas),
                "exames_alterados": len(exames_alterados),
                "exames_pendentes": len(exames_pendentes),
            },
        )
        log_entries.append(entry)

        if logger:
            logger.log_graph_transition(
                from_node="node_verificar_exames",
                to_node="node_gerar_alerta",
                state_summary={
                    "alertas_gerados": len(alertas),
                    "exames_alterados": len(exames_alterados),
                    "exames_pendentes": len(exames_pendentes),
                },
            )

        return {
            "alertas": alertas,
            "log_entries": log_entries,
        }

    except GraphExecutionError:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        if logger:
            logger.log_error(
                exc,
                context={
                    "node": "node_gerar_alerta",
                    "paciente_identificado": paciente_identificado,
                },
            )
        raise GraphExecutionError(
            f"Erro no nó de geração de alerta: {exc}",
            context={
                "node": "node_gerar_alerta",
                "paciente_identificado": paciente_identificado,
                "stack_trace": tb,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Nó 5: Gerar Resposta
# ---------------------------------------------------------------------------


def node_gerar_resposta(state: GraphState) -> dict:
    """Nó 5: Invoca a LLM fine-tunada com contexto do prontuário para gerar resposta.

    Monta o prompt RAG com os documentos recuperados e invoca o modelo de
    linguagem. Inclui o aviso obrigatório de validação humana em sugestões
    clínicas. Quando nenhum prontuário é encontrado, informa explicitamente
    a ausência de dados.

    Args:
        state: Estado atual do grafo (requer ``consulta``, ``_llm_pipeline``
            e ``prontuarios_recuperados``).

    Returns:
        Dicionário com ``resposta_llm`` e ``log_entries`` atualizados.
    """
    logger = state.get("_logger")  # type: ignore[misc]
    llm_pipeline = state.get("_llm_pipeline")  # type: ignore[misc]
    log_entries: list[LogEntry] = list(state.get("log_entries", []))
    consulta = state.get("consulta", "")
    prontuarios_recuperados = state.get("prontuarios_recuperados", [])

    try:
        paciente_encontrado = len(prontuarios_recuperados) > 0

        # Montar contexto a partir dos documentos recuperados
        if paciente_encontrado:
            context = "\n\n---\n\n".join(
                d.get("page_content", "") for d in prontuarios_recuperados
            )
        else:
            context = "Nenhum prontuário encontrado para o paciente mencionado."

        # Montar prompt completo
        prompt = (
            f"{MEDICAL_SYSTEM_PROMPT}\n\n"
            + MEDICAL_RAG_TEMPLATE.format(context=context, question=consulta)
        )

        resposta_llm: str

        if llm_pipeline is not None:
            try:
                result = llm_pipeline(prompt)
                if isinstance(result, list) and result:
                    generated = result[0].get("generated_text", "")
                    # Remove o prompt do início se o modelo o repetir
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    resposta_llm = generated or "Não foi possível gerar uma resposta."
                else:
                    resposta_llm = "Não foi possível gerar uma resposta."
            except Exception as llm_exc:
                raise GraphExecutionError(
                    f"Erro durante inferência da LLM: {llm_exc}",
                    context={"node": "node_gerar_resposta", "consulta": consulta},
                ) from llm_exc
        else:
            # Modo sem LLM (testes / demonstração): resposta baseada no contexto
            if paciente_encontrado:
                resposta_llm = (
                    f"Com base no prontuário recuperado, seguem as informações "
                    f"disponíveis para a consulta: {consulta}\n\n{context}"
                )
            else:
                resposta_llm = (
                    PATIENT_NOT_FOUND_PREFIX
                    + f"Não há dados de prontuário disponíveis para responder "
                    f"à consulta: {consulta}"
                )

        entry = _make_log_entry(
            "GRAPH_TRANSITION",
            node_from="node_gerar_alerta",
            node_to="node_gerar_resposta",
            state_summary={
                "paciente_encontrado": paciente_encontrado,
                "resposta_length": len(resposta_llm),
                "alertas": len(state.get("alertas", [])),
            },
        )
        log_entries.append(entry)

        if logger:
            logger.log_graph_transition(
                from_node="node_gerar_alerta",
                to_node="node_gerar_resposta",
                state_summary={
                    "paciente_encontrado": paciente_encontrado,
                    "resposta_length": len(resposta_llm),
                    "alertas": len(state.get("alertas", [])),
                },
            )

        return {
            "resposta_llm": resposta_llm,
            "log_entries": log_entries,
        }

    except GraphExecutionError:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        if logger:
            logger.log_error(
                exc,
                context={"node": "node_gerar_resposta", "consulta": consulta},
            )
        raise GraphExecutionError(
            f"Erro no nó de geração de resposta: {exc}",
            context={
                "node": "node_gerar_resposta",
                "consulta": consulta,
                "stack_trace": tb,
            },
        ) from exc


# ---------------------------------------------------------------------------
# Nó 6: Formatar Resposta
# ---------------------------------------------------------------------------


def node_formatar_resposta(state: GraphState) -> dict:
    """Nó 6: Formata a resposta final com fontes estruturadas e alertas.

    Garante separação clara entre dados do prontuário (RAG) e conhecimento
    do modelo fine-tunado. Usa o ``ResponseFormatter`` do pipeline para
    montar o ``MedicalResponse`` final.

    Args:
        state: Estado atual do grafo.

    Returns:
        Dicionário com ``resposta_final`` e ``log_entries`` atualizados.
    """
    logger = state.get("_logger")  # type: ignore[misc]
    db = state.get("_db")  # type: ignore[misc]
    log_entries: list[LogEntry] = list(state.get("log_entries", []))
    consulta = state.get("consulta", "")
    resposta_llm = state.get("resposta_llm", "")
    prontuarios_recuperados = state.get("prontuarios_recuperados", [])
    alertas_estado = state.get("alertas", [])
    paciente_identificado = state.get("paciente_identificado")

    try:
        docs = _dicts_to_docs(prontuarios_recuperados)
        paciente_encontrado = len(docs) > 0

        # Recuperar prontuário completo para o formatter
        prontuario: Optional[Prontuario] = None
        if db is not None and paciente_identificado:
            prontuarios = db.search_by_name(paciente_identificado, fuzzy=True)
            if prontuarios:
                prontuario = prontuarios[0]

        formatter = ResponseFormatter()
        resposta_final = formatter.format(
            raw_response=resposta_llm,
            retrieved_docs=docs,
            consulta=consulta,
            prontuario=prontuario,
            paciente_encontrado=paciente_encontrado,
        )

        # Mescla alertas do estado (gerados pelo node_gerar_alerta) com os
        # alertas gerados pelo formatter (para evitar duplicatas, usa os do estado
        # quando disponíveis, pois já foram gerados com base nos exames verificados)
        if alertas_estado:
            resposta_final.alertas = alertas_estado

        entry = _make_log_entry(
            "GRAPH_TRANSITION",
            node_from="node_gerar_resposta",
            node_to="node_formatar_resposta",
            state_summary={
                "paciente_encontrado": paciente_encontrado,
                "alertas_na_resposta": len(resposta_final.alertas),
                "fontes": {
                    "dados_prontuario": resposta_final.fontes.dados_prontuario,
                    "conhecimento_modelo": resposta_final.fontes.conhecimento_modelo,
                    "paciente_encontrado": resposta_final.fontes.paciente_encontrado,
                },
            },
        )
        log_entries.append(entry)

        if logger:
            logger.log_graph_transition(
                from_node="node_gerar_resposta",
                to_node="node_formatar_resposta",
                state_summary={
                    "paciente_encontrado": paciente_encontrado,
                    "alertas_na_resposta": len(resposta_final.alertas),
                },
            )
            # Registrar interação completa no log de auditoria
            logger.log_interaction(
                consulta=consulta,
                resposta=resposta_final,
                sources=resposta_final.fontes,
            )

        return {
            "resposta_final": resposta_final,
            "log_entries": log_entries,
        }

    except GraphExecutionError:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        if logger:
            logger.log_error(
                exc,
                context={"node": "node_formatar_resposta", "consulta": consulta},
            )
        raise GraphExecutionError(
            f"Erro no nó de formatação da resposta: {exc}",
            context={
                "node": "node_formatar_resposta",
                "consulta": consulta,
                "stack_trace": tb,
            },
        ) from exc
