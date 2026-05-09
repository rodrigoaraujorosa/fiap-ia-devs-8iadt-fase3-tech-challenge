"""
Smoke tests para AuditLogger (Tarefa 3.1).

Verifica que:
- O logger inicializa sem erros e cria o diretório de logs.
- O arquivo audit.log é criado após a inicialização.
- O handler configurado é um TimedRotatingFileHandler com os parâmetros corretos.
- Todos os métodos de logging executam sem propagar exceções.
- As entradas escritas são JSON Lines válidos com os campos obrigatórios.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from logging.handlers import TimedRotatingFileHandler

import pytest

from src.utils.logger import AuditLogger
from src.utils.models import (
    AlertPriority,
    AlertType,
    ClinicalAlert,
    MedicalResponse,
    ResponseSources,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_log_dir(tmp_path):
    """Diretório temporário isolado para cada teste."""
    return str(tmp_path / "logs")


@pytest.fixture()
def audit_logger(tmp_log_dir):
    """Instância de AuditLogger apontando para diretório temporário."""
    return AuditLogger(log_dir=tmp_log_dir)


@pytest.fixture()
def sample_response():
    """MedicalResponse de exemplo para testes."""
    sources = ResponseSources(
        dados_prontuario=["exames", "diagnosticos"],
        conhecimento_modelo=False,
        paciente_encontrado=True,
    )
    return MedicalResponse(
        resposta="Paciente apresenta anemia ferropriva. Recomenda-se avaliação.",
        fontes=sources,
        alertas=[
            ClinicalAlert(
                tipo=AlertType.EXAME_PENDENTE,
                prioridade=AlertPriority.MEDIA,
                descricao="Eletrocardiograma pendente.",
                exames_afetados=["E002"],
            )
        ],
        timestamp="2024-01-15T10:30:00Z",
        consulta_original="Quais exames estão pendentes para Ana Clara?",
    )


@pytest.fixture()
def sample_sources():
    return ResponseSources(
        dados_prontuario=["exames"],
        conhecimento_modelo=True,
        paciente_encontrado=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_log_lines(log_dir: str) -> list[dict]:
    """Lê todas as linhas do audit.log e retorna como lista de dicts."""
    log_path = os.path.join(log_dir, "audit.log")
    if not os.path.exists(log_path):
        return []
    with open(log_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [json.loads(line) for line in lines]


def _flush_logger(logger: AuditLogger) -> None:
    """Força flush de todos os handlers do logger interno."""
    for handler in logger._logger.handlers:
        handler.flush()


# ---------------------------------------------------------------------------
# Testes de inicialização
# ---------------------------------------------------------------------------


class TestAuditLoggerInit:
    def test_creates_log_directory(self, tmp_log_dir):
        """O diretório de logs deve ser criado automaticamente."""
        assert not os.path.exists(tmp_log_dir)
        AuditLogger(log_dir=tmp_log_dir)
        assert os.path.isdir(tmp_log_dir)

    def test_creates_audit_log_file(self, tmp_log_dir):
        """O arquivo audit.log deve existir após a inicialização."""
        AuditLogger(log_dir=tmp_log_dir)
        assert os.path.exists(os.path.join(tmp_log_dir, "audit.log"))

    def test_handler_is_timed_rotating(self, audit_logger):
        """O handler configurado deve ser TimedRotatingFileHandler."""
        handlers = audit_logger._logger.handlers
        assert len(handlers) >= 1
        timed_handlers = [h for h in handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(timed_handlers) == 1

    def test_handler_rotation_when_midnight(self, audit_logger):
        """O handler deve ter rotação configurada para 'midnight'."""
        handler = next(
            h for h in audit_logger._logger.handlers
            if isinstance(h, TimedRotatingFileHandler)
        )
        assert handler.when == "MIDNIGHT"

    def test_handler_backup_count_30(self, audit_logger):
        """O handler deve reter 30 arquivos de backup."""
        handler = next(
            h for h in audit_logger._logger.handlers
            if isinstance(h, TimedRotatingFileHandler)
        )
        assert handler.backupCount == 30

    def test_handler_encoding_utf8(self, audit_logger):
        """O handler deve usar encoding UTF-8."""
        handler = next(
            h for h in audit_logger._logger.handlers
            if isinstance(h, TimedRotatingFileHandler)
        )
        assert handler.encoding == "utf-8"

    def test_default_log_dir(self):
        """O diretório padrão deve ser 'logs/'."""
        logger = AuditLogger()
        assert logger._log_dir == "logs/"
        for h in logger._logger.handlers[:]:
            h.close()
            logger._logger.removeHandler(h)

    def test_no_propagation_to_root_logger(self, audit_logger):
        """O logger não deve propagar para o root logger."""
        assert audit_logger._logger.propagate is False


# ---------------------------------------------------------------------------
# Testes de log_interaction
# ---------------------------------------------------------------------------


class TestLogInteraction:
    def test_writes_json_line(self, audit_logger, tmp_log_dir, sample_response, sample_sources):
        """log_interaction deve escrever uma linha JSON válida."""
        audit_logger.log_interaction("Consulta de teste", sample_response, sample_sources)
        _flush_logger(audit_logger)
        lines = _read_log_lines(tmp_log_dir)
        assert len(lines) == 1

    def test_event_type_is_interaction(self, audit_logger, tmp_log_dir, sample_response, sample_sources):
        """O event_type deve ser 'INTERACTION'."""
        audit_logger.log_interaction("Consulta de teste", sample_response, sample_sources)
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["event_type"] == "INTERACTION"

    def test_contains_required_fields(self, audit_logger, tmp_log_dir, sample_response, sample_sources):
        """A entrada deve conter timestamp, event_type, consulta e resposta."""
        consulta = "Quais exames estão pendentes?"
        audit_logger.log_interaction(consulta, sample_response, sample_sources)
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert "timestamp" in entry
        assert "event_type" in entry
        assert entry["consulta"] == consulta
        assert entry["resposta"] == sample_response.resposta

    def test_fontes_is_dict(self, audit_logger, tmp_log_dir, sample_response, sample_sources):
        """O campo fontes deve ser um dicionário serializado."""
        audit_logger.log_interaction("Consulta", sample_response, sample_sources)
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert isinstance(entry["fontes"], dict)

    def test_graph_fields_are_null(self, audit_logger, tmp_log_dir, sample_response, sample_sources):
        """Campos de grafo devem ser null em entradas INTERACTION."""
        audit_logger.log_interaction("Consulta", sample_response, sample_sources)
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["node_from"] is None
        assert entry["node_to"] is None
        assert entry["state_summary"] is None

    def test_does_not_raise_on_call(self, audit_logger, sample_response, sample_sources):
        """log_interaction não deve propagar exceções."""
        audit_logger.log_interaction("Consulta", sample_response, sample_sources)


# ---------------------------------------------------------------------------
# Testes de log_graph_transition
# ---------------------------------------------------------------------------


class TestLogGraphTransition:
    def test_writes_json_line(self, audit_logger, tmp_log_dir):
        """log_graph_transition deve escrever uma linha JSON válida."""
        audit_logger.log_graph_transition(
            "verificar_exames", "gerar_alerta", {"exames_pendentes": 1}
        )
        _flush_logger(audit_logger)
        lines = _read_log_lines(tmp_log_dir)
        assert len(lines) == 1

    def test_event_type_is_graph_transition(self, audit_logger, tmp_log_dir):
        """O event_type deve ser 'GRAPH_TRANSITION'."""
        audit_logger.log_graph_transition("node_a", "node_b", {})
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["event_type"] == "GRAPH_TRANSITION"

    def test_contains_node_fields(self, audit_logger, tmp_log_dir):
        """A entrada deve conter node_from, node_to e state_summary."""
        audit_logger.log_graph_transition(
            "verificar_exames", "gerar_alerta", {"exames_pendentes": 2, "exames_alterados": 0}
        )
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["node_from"] == "verificar_exames"
        assert entry["node_to"] == "gerar_alerta"
        assert entry["state_summary"] == {"exames_pendentes": 2, "exames_alterados": 0}

    def test_interaction_fields_are_null(self, audit_logger, tmp_log_dir):
        """Campos de interação devem ser null em entradas GRAPH_TRANSITION."""
        audit_logger.log_graph_transition("a", "b", {})
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["consulta"] is None
        assert entry["resposta"] is None
        assert entry["fontes"] is None

    def test_does_not_raise_on_call(self, audit_logger):
        """log_graph_transition não deve propagar exceções."""
        audit_logger.log_graph_transition("a", "b", {"key": "value"})


# ---------------------------------------------------------------------------
# Testes de log_error
# ---------------------------------------------------------------------------


class TestLogError:
    def test_writes_json_line(self, audit_logger, tmp_log_dir):
        """log_error deve escrever uma linha JSON válida."""
        try:
            raise ValueError("Erro de teste")
        except ValueError as exc:
            audit_logger.log_error(exc, {"node": "node_gerar_resposta"})
        _flush_logger(audit_logger)
        lines = _read_log_lines(tmp_log_dir)
        assert len(lines) == 1

    def test_event_type_is_error(self, audit_logger, tmp_log_dir):
        """O event_type deve ser 'ERROR'."""
        try:
            raise RuntimeError("Falha interna")
        except RuntimeError as exc:
            audit_logger.log_error(exc, {})
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["event_type"] == "ERROR"

    def test_contains_error_message(self, audit_logger, tmp_log_dir):
        """A entrada deve conter error_message com a mensagem da exceção."""
        try:
            raise ValueError("Mensagem de erro específica")
        except ValueError as exc:
            audit_logger.log_error(exc, {})
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["error_message"] == "Mensagem de erro específica"

    def test_contains_stack_trace(self, audit_logger, tmp_log_dir):
        """A entrada deve conter stack_trace não vazio."""
        try:
            raise ValueError("Erro com traceback")
        except ValueError as exc:
            audit_logger.log_error(exc, {})
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        assert entry["stack_trace"] is not None
        assert len(entry["stack_trace"]) > 0

    def test_does_not_raise_on_call(self, audit_logger):
        """log_error não deve propagar exceções."""
        exc = ValueError("Erro qualquer")
        audit_logger.log_error(exc, {"contexto": "teste"})


# ---------------------------------------------------------------------------
# Testes de múltiplas entradas e formato JSON Lines
# ---------------------------------------------------------------------------


class TestJsonLinesFormat:
    def test_multiple_entries_each_on_own_line(
        self, audit_logger, tmp_log_dir, sample_response, sample_sources
    ):
        """Cada entrada deve estar em sua própria linha no arquivo."""
        audit_logger.log_interaction("Consulta 1", sample_response, sample_sources)
        audit_logger.log_graph_transition("a", "b", {})
        try:
            raise ValueError("Erro")
        except ValueError as exc:
            audit_logger.log_error(exc, {})
        _flush_logger(audit_logger)

        log_path = os.path.join(tmp_log_dir, "audit.log")
        with open(log_path, encoding="utf-8") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        assert len(raw_lines) == 3
        for line in raw_lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "event_type" in parsed

    def test_timestamp_is_iso8601(self, audit_logger, tmp_log_dir):
        """O timestamp deve estar no formato ISO 8601."""
        import re
        audit_logger.log_graph_transition("a", "b", {})
        _flush_logger(audit_logger)
        entry = _read_log_lines(tmp_log_dir)[0]
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        assert re.match(pattern, entry["timestamp"]), (
            f"Timestamp '{entry['timestamp']}' não está no formato ISO 8601"
        )


# ---------------------------------------------------------------------------
# Testes de fallback para stderr (Tarefa 3.6)
# ---------------------------------------------------------------------------


class TestStderrFallback:
    """Garante que falhas no logger fazem fallback para stderr sem propagar exceção."""

    def test_init_fallback_to_stream_handler_on_invalid_path(self, capsys):
        """Se o diretório de log não puder ser criado, deve usar StreamHandler(stderr)."""
        invalid_path = "\x00invalid\x00path"
        logger = AuditLogger(log_dir=invalid_path)

        assert len(logger._logger.handlers) >= 1
        timed = [h for h in logger._logger.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(timed) == 0
        stream_handlers = [
            h for h in logger._logger.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        ]
        assert len(stream_handlers) >= 1

        for h in logger._logger.handlers[:]:
            h.close()
            logger._logger.removeHandler(h)

    def test_init_fallback_prints_to_stderr(self, capsys):
        """Falha na inicialização deve imprimir mensagem de aviso no stderr."""
        invalid_path = "\x00invalid\x00path"
        AuditLogger(log_dir=invalid_path)
        captured = capsys.readouterr()
        assert "AuditLogger" in captured.err
        assert "fallback" in captured.err.lower() or "stderr" in captured.err.lower() or "Falha" in captured.err

    def test_log_interaction_does_not_raise_on_handler_failure(self, capsys):
        """log_interaction não deve propagar exceção mesmo com handler com falha."""
        logger = AuditLogger(log_dir="\x00invalid\x00path")
        sources = ResponseSources(dados_prontuario=[], conhecimento_modelo=False, paciente_encontrado=False)
        response = MedicalResponse(
            resposta="Teste",
            fontes=sources,
            alertas=[],
            timestamp="2024-01-01T00:00:00Z",
            consulta_original="Teste",
        )
        logger.log_interaction("Consulta", response, sources)

        for h in logger._logger.handlers[:]:
            h.close()
            logger._logger.removeHandler(h)

    def test_log_graph_transition_does_not_raise_on_handler_failure(self):
        """log_graph_transition não deve propagar exceção mesmo com handler com falha."""
        logger = AuditLogger(log_dir="\x00invalid\x00path")
        logger.log_graph_transition("node_a", "node_b", {"key": "value"})

        for h in logger._logger.handlers[:]:
            h.close()
            logger._logger.removeHandler(h)

    def test_log_error_does_not_raise_on_handler_failure(self):
        """log_error não deve propagar exceção mesmo com handler com falha."""
        logger = AuditLogger(log_dir="\x00invalid\x00path")
        logger.log_error(ValueError("Erro de teste"), {"contexto": "teste"})

        for h in logger._logger.handlers[:]:
            h.close()
            logger._logger.removeHandler(h)

    def test_methods_do_not_raise_when_write_entry_fails(self, monkeypatch):
        """Métodos públicos não devem propagar exceção se _write_entry lançar erro."""
        logger = AuditLogger(log_dir=tempfile.mkdtemp())

        def _broken_write(entry):
            raise OSError("Disco cheio simulado")

        monkeypatch.setattr(logger, "_write_entry", _broken_write)

        sources = ResponseSources(dados_prontuario=[], conhecimento_modelo=False, paciente_encontrado=False)
        response = MedicalResponse(
            resposta="Teste",
            fontes=sources,
            alertas=[],
            timestamp="2024-01-01T00:00:00Z",
            consulta_original="Teste",
        )

        logger.log_interaction("Consulta", response, sources)
        logger.log_graph_transition("a", "b", {})
        logger.log_error(ValueError("err"), {})

    def test_methods_print_to_stderr_when_write_entry_fails(self, monkeypatch, capsys):
        """Métodos públicos devem imprimir no stderr quando _write_entry falha."""
        logger = AuditLogger(log_dir=tempfile.mkdtemp())

        def _broken_write(entry):
            raise OSError("Disco cheio simulado")

        monkeypatch.setattr(logger, "_write_entry", _broken_write)

        sources = ResponseSources(dados_prontuario=[], conhecimento_modelo=False, paciente_encontrado=False)
        response = MedicalResponse(
            resposta="Teste",
            fontes=sources,
            alertas=[],
            timestamp="2024-01-01T00:00:00Z",
            consulta_original="Teste",
        )
        logger.log_interaction("Consulta", response, sources)
        captured = capsys.readouterr()
        assert "AuditLogger" in captured.err
