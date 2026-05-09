"""
Testes unitários para o módulo ProntuarioDB.

Cobre: carregamento, índices em memória, busca por nome (exata e fuzzy),
busca por status, exames pendentes, exames alterados, serialização to_text
e requisito de performance (< 2s para 20+ registros).
"""

from __future__ import annotations

import time

import pytest

from src.database.prontuario_db import ProntuarioDB, _normalize
from src.utils.exceptions import DatabaseError
from src.utils.models import ExameStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db() -> ProntuarioDB:
    """Carrega a base real de prontuários uma única vez para todos os testes."""
    database = ProntuarioDB("data/prontuarios.json")
    database.load()
    return database


# ---------------------------------------------------------------------------
# Testes de carregamento
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_populates_prontuarios(self, db: ProntuarioDB) -> None:
        assert db.total >= 20, "Base deve ter pelo menos 20 prontuários"

    def test_load_invalid_path_raises_database_error(self) -> None:
        bad_db = ProntuarioDB("data/nao_existe.json")
        with pytest.raises(DatabaseError, match="não encontrado"):
            bad_db.load()

    def test_prontuarios_have_required_fields(self, db: ProntuarioDB) -> None:
        for p in db.prontuarios:
            assert p.id
            assert p.nome
            assert p.data_nascimento
            assert p.sexo
            assert isinstance(p.exames, list)
            assert isinstance(p.diagnosticos, list)
            assert isinstance(p.medicamentos_em_uso, list)


# ---------------------------------------------------------------------------
# Testes de normalização
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_removes_accents(self) -> None:
        assert _normalize("Ação") == "acao"

    def test_lowercases(self) -> None:
        assert _normalize("ANA CLARA") == "ana clara"

    def test_strips_whitespace(self) -> None:
        assert _normalize("  ana  ") == "ana"

    def test_combined(self) -> None:
        assert _normalize("José Antônio") == "jose antonio"


# ---------------------------------------------------------------------------
# Testes de search_by_name
# ---------------------------------------------------------------------------


class TestSearchByName:
    def test_exact_name_returns_result(self, db: ProntuarioDB) -> None:
        results = db.search_by_name("Ana Clara Ferreira")
        assert len(results) >= 1
        assert any("Ana Clara" in p.nome for p in results)

    def test_partial_name_returns_result(self, db: ProntuarioDB) -> None:
        results = db.search_by_name("Ana Clara")
        assert len(results) >= 1

    def test_case_insensitive(self, db: ProntuarioDB) -> None:
        results = db.search_by_name("ana clara ferreira")
        assert len(results) >= 1

    def test_accented_name_normalized(self, db: ProntuarioDB) -> None:
        # "José Antônio Pereira" deve ser encontrado sem acentos
        results = db.search_by_name("Jose Antonio Pereira")
        assert len(results) >= 1

    def test_fuzzy_match_typo(self, db: ProntuarioDB) -> None:
        # Typo leve: "Ana Klara" deve encontrar "Ana Clara" com fuzzy=True
        results = db.search_by_name("Ana Klara", fuzzy=True)
        assert len(results) >= 1

    def test_no_fuzzy_strict_miss(self, db: ProntuarioDB) -> None:
        # Nome completamente diferente não deve retornar nada
        results = db.search_by_name("XYZXYZXYZ", fuzzy=False)
        assert results == []

    def test_empty_name_returns_empty(self, db: ProntuarioDB) -> None:
        assert db.search_by_name("") == []
        assert db.search_by_name("   ") == []

    def test_nonexistent_name_returns_empty(self, db: ProntuarioDB) -> None:
        results = db.search_by_name("Paciente Inexistente ZZZZ", fuzzy=False)
        assert results == []


# ---------------------------------------------------------------------------
# Testes de search_by_status
# ---------------------------------------------------------------------------


class TestSearchByStatus:
    def test_pendente_returns_prontuarios(self, db: ProntuarioDB) -> None:
        results = db.search_by_status(ExameStatus.PENDENTE)
        assert len(results) >= 1
        # Todos os retornados devem ter pelo menos um exame pendente
        for p in results:
            statuses = [e.status for e in p.exames]
            assert ExameStatus.PENDENTE in statuses

    def test_alterado_returns_prontuarios(self, db: ProntuarioDB) -> None:
        results = db.search_by_status(ExameStatus.ALTERADO)
        assert len(results) >= 1
        for p in results:
            statuses = [e.status for e in p.exames]
            assert ExameStatus.ALTERADO in statuses

    def test_concluido_returns_prontuarios(self, db: ProntuarioDB) -> None:
        results = db.search_by_status(ExameStatus.CONCLUIDO)
        assert len(results) >= 1
        for p in results:
            statuses = [e.status for e in p.exames]
            assert ExameStatus.CONCLUIDO in statuses

    def test_string_status_accepted(self, db: ProntuarioDB) -> None:
        # Deve aceitar string e converter para ExameStatus
        results = db.search_by_status("pendente")  # type: ignore[arg-type]
        assert isinstance(results, list)

    def test_invalid_status_raises_database_error(self, db: ProntuarioDB) -> None:
        with pytest.raises(DatabaseError):
            db.search_by_status("status_invalido")  # type: ignore[arg-type]

    def test_no_duplicates_in_results(self, db: ProntuarioDB) -> None:
        results = db.search_by_status(ExameStatus.ALTERADO)
        ids = [p.id for p in results]
        assert len(ids) == len(set(ids)), "Não deve haver prontuários duplicados"


# ---------------------------------------------------------------------------
# Testes de get_pending_exams
# ---------------------------------------------------------------------------


class TestGetPendingExams:
    def test_returns_only_pending(self, db: ProntuarioDB) -> None:
        # P001 tem exame E003B com status pendente
        exames = db.get_pending_exams("P001")
        assert len(exames) >= 1
        for e in exames:
            assert e.status == ExameStatus.PENDENTE

    def test_nonexistent_patient_returns_empty(self, db: ProntuarioDB) -> None:
        assert db.get_pending_exams("P999") == []

    def test_patient_without_pending_returns_empty(self, db: ProntuarioDB) -> None:
        # Encontra um paciente que não tem exames pendentes
        for p in db.prontuarios:
            statuses = [e.status for e in p.exames]
            if ExameStatus.PENDENTE not in statuses:
                result = db.get_pending_exams(p.id)
                assert result == []
                break


# ---------------------------------------------------------------------------
# Testes de get_altered_exams
# ---------------------------------------------------------------------------


class TestGetAlteredExams:
    def test_returns_only_altered(self, db: ProntuarioDB) -> None:
        # P001 tem exames E001 e E002 com status alterado
        exames = db.get_altered_exams("P001")
        assert len(exames) >= 1
        for e in exames:
            assert e.status == ExameStatus.ALTERADO

    def test_nonexistent_patient_returns_empty(self, db: ProntuarioDB) -> None:
        assert db.get_altered_exams("P999") == []


# ---------------------------------------------------------------------------
# Testes de to_text
# ---------------------------------------------------------------------------


class TestToText:
    def test_contains_patient_name(self, db: ProntuarioDB) -> None:
        p = db.prontuarios[0]
        text = db.to_text(p)
        assert p.nome in text

    def test_contains_patient_id(self, db: ProntuarioDB) -> None:
        p = db.prontuarios[0]
        text = db.to_text(p)
        assert p.id in text

    def test_contains_diagnosticos(self, db: ProntuarioDB) -> None:
        p = db.prontuarios[0]
        text = db.to_text(p)
        if p.diagnosticos:
            assert p.diagnosticos[0] in text

    def test_contains_exam_names(self, db: ProntuarioDB) -> None:
        p = db.prontuarios[0]
        text = db.to_text(p)
        if p.exames:
            assert p.exames[0].nome in text

    def test_contains_exam_status(self, db: ProntuarioDB) -> None:
        p = db.prontuarios[0]
        text = db.to_text(p)
        if p.exames:
            assert p.exames[0].status.value in text

    def test_returns_string(self, db: ProntuarioDB) -> None:
        for p in db.prontuarios:
            assert isinstance(db.to_text(p), str)

    def test_nonempty_for_all_prontuarios(self, db: ProntuarioDB) -> None:
        for p in db.prontuarios:
            assert len(db.to_text(p)) > 0


# ---------------------------------------------------------------------------
# Teste de performance (subtask 4.8)
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_search_by_name_under_2_seconds(self, db: ProntuarioDB) -> None:
        start = time.perf_counter()
        db.search_by_name("Ana Clara", fuzzy=True)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"search_by_name levou {elapsed:.3f}s (limite: 2s)"

    def test_search_by_status_under_2_seconds(self, db: ProntuarioDB) -> None:
        start = time.perf_counter()
        db.search_by_status(ExameStatus.PENDENTE)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"search_by_status levou {elapsed:.3f}s (limite: 2s)"

    def test_get_pending_exams_under_2_seconds(self, db: ProntuarioDB) -> None:
        start = time.perf_counter()
        db.get_pending_exams("P001")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"get_pending_exams levou {elapsed:.3f}s (limite: 2s)"

    def test_get_altered_exams_under_2_seconds(self, db: ProntuarioDB) -> None:
        start = time.perf_counter()
        db.get_altered_exams("P001")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"get_altered_exams levou {elapsed:.3f}s (limite: 2s)"

    def test_to_text_all_prontuarios_under_2_seconds(self, db: ProntuarioDB) -> None:
        start = time.perf_counter()
        for p in db.prontuarios:
            db.to_text(p)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, (
            f"to_text para {db.total} prontuários levou {elapsed:.3f}s (limite: 2s)"
        )
