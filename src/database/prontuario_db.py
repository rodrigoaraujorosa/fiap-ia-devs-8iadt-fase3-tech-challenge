"""
Módulo de banco de dados de prontuários médicos.

Carrega prontuários de um arquivo JSON e constrói índices em memória para
buscas eficientes por nome, status de exame e identificador de paciente.

Todas as operações de busca retornam resultados em menos de 2 segundos
para bases com 20+ registros, conforme requisito de performance.
"""

from __future__ import annotations

import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from src.utils.exceptions import DatabaseError
from src.utils.models import Exame, ExameStatus, Prontuario


def _normalize(text: str) -> str:
    """Normaliza texto para comparação: minúsculas e sem acentos.

    Args:
        text: Texto a ser normalizado.

    Returns:
        Texto em minúsculas sem caracteres acentuados.
    """
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


class ProntuarioDB:
    """Gerenciador da base de prontuários de pacientes.

    Carrega prontuários de um arquivo JSON e mantém índices em memória
    para buscas eficientes por nome, status de exame e ID de paciente.

    Args:
        db_path: Caminho para o arquivo ``prontuarios.json``.
            Padrão: ``"data/prontuarios.json"``.

    Example::

        db = ProntuarioDB("data/prontuarios.json")
        db.load()
        resultados = db.search_by_name("Ana Clara")
        pendentes = db.get_pending_exams("P001")
    """

    def __init__(self, db_path: str = "data/prontuarios.json") -> None:
        self._db_path = Path(db_path)
        # Lista principal de prontuários
        self._prontuarios: list[Prontuario] = []
        # Índice por ID: {patient_id -> Prontuario}
        self._index_by_id: dict[str, Prontuario] = {}
        # Índice por status: {ExameStatus -> list[Prontuario]}
        self._index_by_status: dict[ExameStatus, list[Prontuario]] = {
            s: [] for s in ExameStatus
        }
        # Índice de nomes normalizados: {normalized_name -> Prontuario}
        self._index_by_name: dict[str, Prontuario] = {}

    # ------------------------------------------------------------------
    # Carregamento e indexação
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Carrega prontuários do JSON e constrói índices em memória.

        Raises:
            DatabaseError: Se o arquivo não existir, não for JSON válido
                ou não seguir o schema esperado.
        """
        try:
            if not self._db_path.exists():
                raise DatabaseError(
                    f"Arquivo de prontuários não encontrado: {self._db_path}",
                    context={"db_path": str(self._db_path)},
                )

            with self._db_path.open(encoding="utf-8") as f:
                data = json.load(f)

            raw_prontuarios = data.get("prontuarios", [])
            if not isinstance(raw_prontuarios, list):
                raise DatabaseError(
                    "Campo 'prontuarios' deve ser uma lista no JSON.",
                    context={"db_path": str(self._db_path)},
                )

            self._prontuarios = [Prontuario(**p) for p in raw_prontuarios]
            self._build_indexes()

        except DatabaseError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise DatabaseError(
                f"Erro ao carregar prontuários: {exc}",
                context={"db_path": str(self._db_path)},
            ) from exc

    def _build_indexes(self) -> None:
        """Constrói todos os índices em memória a partir da lista de prontuários."""
        # Reinicia índices
        self._index_by_id = {}
        self._index_by_status = {s: [] for s in ExameStatus}
        self._index_by_name = {}

        for prontuario in self._prontuarios:
            # Índice por ID
            self._index_by_id[prontuario.id] = prontuario

            # Índice por nome normalizado
            normalized = _normalize(prontuario.nome)
            self._index_by_name[normalized] = prontuario

            # Índice por status: um prontuário pode aparecer em múltiplos status
            statuses_presentes: set[ExameStatus] = set()
            for exame in prontuario.exames:
                if exame.status not in statuses_presentes:
                    self._index_by_status[exame.status].append(prontuario)
                    statuses_presentes.add(exame.status)

    # ------------------------------------------------------------------
    # Métodos de busca
    # ------------------------------------------------------------------

    def search_by_name(self, name: str, fuzzy: bool = True) -> list[Prontuario]:
        """Busca prontuários por nome do paciente.

        Realiza correspondência exata (substring normalizada) ou aproximada
        usando ``difflib.SequenceMatcher`` quando ``fuzzy=True``.

        Args:
            name: Nome ou parte do nome do paciente.
            fuzzy: Se ``True``, usa correspondência aproximada além da
                correspondência por substring. Padrão: ``True``.

        Returns:
            Lista de prontuários correspondentes, ordenada por relevância
            (correspondência exata primeiro). Retorna lista vazia se não
            encontrado.
        """
        if not name or not name.strip():
            return []

        query = _normalize(name)
        exact_matches: list[Prontuario] = []
        fuzzy_matches: list[tuple[float, Prontuario]] = []

        for normalized_name, prontuario in self._index_by_name.items():
            # Correspondência por substring (exata normalizada)
            if query in normalized_name or normalized_name in query:
                exact_matches.append(prontuario)
                continue

            # Correspondência aproximada via SequenceMatcher
            if fuzzy:
                # Compara query contra o nome completo
                ratio_full = SequenceMatcher(None, query, normalized_name).ratio()
                # Compara query contra cada token do nome (ex.: "ana klara" vs "ana clara")
                name_tokens = normalized_name.split()
                query_tokens = query.split()
                ratio_tokens = max(
                    (
                        SequenceMatcher(None, qt, nt).ratio()
                        for qt in query_tokens
                        for nt in name_tokens
                    ),
                    default=0.0,
                )
                ratio = max(ratio_full, ratio_tokens)
                if ratio >= 0.6:
                    fuzzy_matches.append((ratio, prontuario))

        # Ordena fuzzy matches por score decrescente
        fuzzy_matches.sort(key=lambda x: x[0], reverse=True)
        fuzzy_results = [p for _, p in fuzzy_matches]

        # Combina: exatos primeiro, depois aproximados (sem duplicatas)
        seen_ids: set[str] = {p.id for p in exact_matches}
        for p in fuzzy_results:
            if p.id not in seen_ids:
                exact_matches.append(p)
                seen_ids.add(p.id)

        return exact_matches

    def search_by_status(self, status: ExameStatus) -> list[Prontuario]:
        """Retorna todos os prontuários com pelo menos um exame no status dado.

        Args:
            status: Status de exame a filtrar (``ExameStatus.PENDENTE``,
                ``ExameStatus.CONCLUIDO``, ``ExameStatus.ALTERADO`` ou
                ``ExameStatus.CANCELADO``).

        Returns:
            Lista de prontuários que possuem pelo menos um exame com o
            status informado. Retorna lista vazia se nenhum encontrado.

        Raises:
            DatabaseError: Se o status informado não for um ``ExameStatus`` válido.
        """
        if not isinstance(status, ExameStatus):
            try:
                status = ExameStatus(status)
            except ValueError as exc:
                raise DatabaseError(
                    f"Status inválido: {status!r}. "
                    f"Valores válidos: {[s.value for s in ExameStatus]}",
                    context={"status": str(status)},
                ) from exc

        return list(self._index_by_status.get(status, []))

    def get_pending_exams(self, patient_id: str) -> list[Exame]:
        """Retorna exames com status ``pendente`` para um paciente.

        Args:
            patient_id: Identificador único do paciente (ex.: ``"P001"``).

        Returns:
            Lista de exames com status ``pendente``. Retorna lista vazia
            se o paciente não existir ou não tiver exames pendentes.
        """
        prontuario = self._index_by_id.get(patient_id)
        if prontuario is None:
            return []
        return [e for e in prontuario.exames if e.status == ExameStatus.PENDENTE]

    def get_altered_exams(self, patient_id: str) -> list[Exame]:
        """Retorna exames com status ``alterado`` para um paciente.

        Args:
            patient_id: Identificador único do paciente (ex.: ``"P001"``).

        Returns:
            Lista de exames com status ``alterado``. Retorna lista vazia
            se o paciente não existir ou não tiver exames alterados.
        """
        prontuario = self._index_by_id.get(patient_id)
        if prontuario is None:
            return []
        return [e for e in prontuario.exames if e.status == ExameStatus.ALTERADO]

    # ------------------------------------------------------------------
    # Serialização para indexação FAISS
    # ------------------------------------------------------------------

    def to_text(self, prontuario: Prontuario) -> str:
        """Serializa um prontuário para texto legível para indexação no FAISS.

        Inclui todos os campos relevantes do prontuário em formato de texto
        estruturado para maximizar a qualidade dos embeddings semânticos.

        Args:
            prontuario: Prontuário a ser serializado.

        Returns:
            Texto estruturado com todos os campos do prontuário.
        """
        lines: list[str] = [
            f"Paciente: {prontuario.nome}",
            f"ID: {prontuario.id}",
            f"Data de Nascimento: {prontuario.data_nascimento}",
            f"Sexo: {prontuario.sexo}",
        ]

        if prontuario.diagnosticos:
            lines.append("Diagnósticos: " + "; ".join(prontuario.diagnosticos))

        if prontuario.medicamentos_em_uso:
            lines.append(
                "Medicamentos em uso: " + "; ".join(prontuario.medicamentos_em_uso)
            )

        if prontuario.observacoes_gerais:
            lines.append(f"Observações gerais: {prontuario.observacoes_gerais}")

        if prontuario.exames:
            lines.append("Exames:")
            for exame in prontuario.exames:
                exame_parts = [
                    f"  - {exame.nome} (ID: {exame.id})",
                    f"    Status: {exame.status.value}",
                    f"    Solicitado em: {exame.data_solicitacao}",
                ]
                if exame.data_resultado:
                    exame_parts.append(f"    Resultado em: {exame.data_resultado}")
                if exame.resultado:
                    exame_parts.append(f"    Resultado: {exame.resultado}")
                if exame.observacoes_medico:
                    exame_parts.append(
                        f"    Observações médico: {exame.observacoes_medico}"
                    )
                lines.extend(exame_parts)

        lines.append(f"Última atualização: {prontuario.ultima_atualizacao}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Propriedades de acesso
    # ------------------------------------------------------------------

    @property
    def prontuarios(self) -> list[Prontuario]:
        """Lista de todos os prontuários carregados."""
        return list(self._prontuarios)

    @property
    def total(self) -> int:
        """Número total de prontuários na base."""
        return len(self._prontuarios)
