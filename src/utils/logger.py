"""
Módulo de logging e auditoria do Assistente Virtual Médico Hospitalar.

Registra todas as interações, transições de grafo e erros em formato
JSON Lines com rotação diária e retenção de 30 dias.

Falhas internas do logger fazem fallback para stderr sem propagar exceção,
garantindo que erros de logging nunca interrompam o fluxo principal.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from typing import Any

from src.utils.models import LogEntry, MedicalResponse, ResponseSources


class AuditLogger:
    """Logger de auditoria com rotação diária e retenção de 30 dias.

    Registra interações, transições de grafo LangGraph, erros internos e
    relatórios de curadoria em formato JSON Lines no arquivo ``logs/audit.log``.

    Todas as operações de logging são protegidas por try/except: falhas no
    logger fazem fallback para stderr sem propagar exceção ao chamador.

    Args:
        log_dir: Diretório onde o arquivo ``audit.log`` será criado.
            Criado automaticamente se não existir. Padrão: ``"logs/"``.

    Example::

        logger = AuditLogger(log_dir="logs/")
        logger.log_error(exc, context={"node": "node_gerar_resposta"})
    """

    def __init__(self, log_dir: str = "logs/") -> None:
        """Configura TimedRotatingFileHandler com rotação diária e retenção de 30 dias.

        Args:
            log_dir: Diretório de destino dos arquivos de log.
        """
        self._log_dir = log_dir
        self._logger = logging.getLogger(f"audit.{id(self)}")
        self._logger.setLevel(logging.DEBUG)
        # Evita propagação para o root logger (evita duplicação em stdout)
        self._logger.propagate = False

        try:
            os.makedirs(log_dir, exist_ok=True)

            handler = TimedRotatingFileHandler(
                filename=os.path.join(log_dir, "audit.log"),
                when="midnight",   # rotação diária
                interval=1,
                backupCount=30,    # retenção de 30 dias
                encoding="utf-8",
            )
            handler.setLevel(logging.DEBUG)
            # Formato mínimo: a mensagem já é JSON serializado
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

        except Exception as exc:
            print(
                f"[AuditLogger] Falha ao configurar handler de arquivo: {exc}. "
                "Usando stderr como fallback.",
                file=sys.stderr,
            )
            # Fallback: usa StreamHandler(stderr) para não perder mensagens de log
            fallback_handler = logging.StreamHandler(sys.stderr)
            fallback_handler.setLevel(logging.DEBUG)
            fallback_handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(fallback_handler)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        """Retorna o timestamp atual em formato ISO 8601 UTC."""
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _write_entry(self, entry: LogEntry) -> None:
        """Serializa ``entry`` como JSON e escreve no arquivo de log.

        Falhas na serialização ou escrita fazem fallback para stderr.

        Args:
            entry: Entrada de log a ser registrada.
        """
        try:
            line = json.dumps(dataclasses.asdict(entry), ensure_ascii=False)
            self._logger.info(line)
        except Exception as exc:  # pragma: no cover
            print(
                f"[AuditLogger] Falha ao escrever entrada de log: {exc}",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------
    # Métodos públicos de logging
    # ------------------------------------------------------------------

    def log_interaction(
        self,
        consulta: str,
        resposta: MedicalResponse,
        sources: ResponseSources,
    ) -> None:
        """Registra uma interação completa com timestamp, consulta, resposta e fontes.

        Atende ao Requisito 6.3: registra timestamp, consulta submetida,
        resposta gerada e fontes utilizadas.

        Args:
            consulta: Consulta submetida pelo médico.
            resposta: Objeto ``MedicalResponse`` com a resposta gerada.
            sources: Fontes estruturadas utilizadas na geração da resposta.
        """
        try:
            fontes_dict: dict[str, Any] = dataclasses.asdict(sources)
            entry = LogEntry(
                timestamp=self._now_iso(),
                event_type="INTERACTION",
                consulta=consulta,
                resposta=resposta.resposta,
                fontes=fontes_dict,
                node_from=None,
                node_to=None,
                state_summary=None,
                error_message=None,
                stack_trace=None,
            )
            self._write_entry(entry)
        except Exception as exc:  # pragma: no cover
            print(
                f"[AuditLogger] Falha em log_interaction: {exc}",
                file=sys.stderr,
            )

    def log_graph_transition(
        self,
        from_node: str,
        to_node: str,
        state_summary: dict,
    ) -> None:
        """Registra transição entre nós do grafo LangGraph.

        Atende ao Requisito 9.3: registra nó de origem, nó de destino e
        estado intermediário transmitido entre eles.

        Args:
            from_node: Nome do nó de origem.
            to_node: Nome do nó de destino.
            state_summary: Resumo do estado intermediário (campos relevantes).
        """
        try:
            entry = LogEntry(
                timestamp=self._now_iso(),
                event_type="GRAPH_TRANSITION",
                consulta=None,
                resposta=None,
                fontes=None,
                node_from=from_node,
                node_to=to_node,
                state_summary=state_summary,
                error_message=None,
                stack_trace=None,
            )
            self._write_entry(entry)
        except Exception as exc:  # pragma: no cover
            print(
                f"[AuditLogger] Falha em log_graph_transition: {exc}",
                file=sys.stderr,
            )

    def log_error(self, error: Exception, context: dict) -> None:
        """Registra erro interno com stack trace completo.

        Atende ao Requisito 6.5: registra o erro completo com stack trace
        sem expor detalhes técnicos ao usuário final.

        Args:
            error: Exceção capturada.
            context: Dicionário com informações de contexto (ex.: nome do nó,
                consulta original, identificador do paciente).
        """
        try:
            tb = traceback.format_exc()
            # Se não há traceback ativo, formata a partir da exceção diretamente
            if tb.strip() == "NoneType: None":
                tb = "".join(
                    traceback.format_exception(type(error), error, error.__traceback__)
                )
            entry = LogEntry(
                timestamp=self._now_iso(),
                event_type="ERROR",
                consulta=context.get("consulta"),
                resposta=None,
                fontes=None,
                node_from=context.get("node_from"),
                node_to=context.get("node_to"),
                state_summary={k: v for k, v in context.items()
                               if k not in ("consulta", "node_from", "node_to")}
                               or None,
                error_message=str(error),
                stack_trace=tb,
            )
            self._write_entry(entry)
        except Exception as exc:  # pragma: no cover
            print(
                f"[AuditLogger] Falha em log_error: {exc}",
                file=sys.stderr,
            )


