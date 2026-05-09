"""
Módulo de formatação de respostas do Assistente Virtual Médico Hospitalar.

Monta ``MedicalResponse`` com resposta, fontes estruturadas (``ResponseSources``)
e alertas clínicos (``ClinicalAlert``). Insere o aviso de validação humana como
pós-processamento quando a resposta contém sugestões clínicas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from langchain_core.documents import Document

from src.pipeline.prompt_templates import (
    CLINICAL_SUGGESTION_KEYWORDS,
    HUMAN_VALIDATION_WARNING,
    PATIENT_NOT_FOUND_PREFIX,
)
from src.utils.models import (
    AlertPriority,
    AlertType,
    ClinicalAlert,
    ExameStatus,
    MedicalResponse,
    Prontuario,
    ResponseSources,
)


class ResponseFormatter:
    """Formata a resposta bruta da LLM em um ``MedicalResponse`` estruturado.

    Responsabilidades:
    - Detectar sugestões clínicas e inserir aviso de validação humana
    - Construir ``ResponseSources`` a partir dos documentos recuperados
    - Gerar ``ClinicalAlert`` para exames pendentes e alterados
    - Montar seção de fontes estruturadas ao final da resposta
    - Prefixar resposta com aviso de ausência de dados quando paciente não encontrado

    Example::

        formatter = ResponseFormatter()
        response = formatter.format(
            raw_response="Sugiro solicitar hemograma completo.",
            retrieved_docs=docs,
            consulta="Quais exames solicitar para Ana Clara?",
            prontuario=prontuario,
        )
    """

    # ------------------------------------------------------------------
    # Método principal
    # ------------------------------------------------------------------

    def format(
        self,
        raw_response: str,
        retrieved_docs: list[Document],
        consulta: str,
        prontuario: Optional[Prontuario] = None,
        paciente_encontrado: bool = False,
    ) -> MedicalResponse:
        """Formata a resposta bruta em ``MedicalResponse`` estruturado.

        Args:
            raw_response: Texto bruto gerado pela LLM.
            retrieved_docs: Documentos recuperados via RAG.
            consulta: Consulta original do médico.
            prontuario: Prontuário do paciente identificado, se encontrado.
            paciente_encontrado: Se ``True``, indica que um prontuário foi
                encontrado para o paciente mencionado na consulta.

        Returns:
            ``MedicalResponse`` com resposta formatada, fontes e alertas.
        """
        # 1. Construir fontes estruturadas
        sources = self._build_sources(retrieved_docs, paciente_encontrado)

        # 2. Gerar alertas clínicos a partir do prontuário
        alertas = self._build_alerts(prontuario) if prontuario else []

        # 3. Prefixar com aviso de ausência de dados se paciente não encontrado
        resposta = raw_response
        if not paciente_encontrado and retrieved_docs == []:
            resposta = PATIENT_NOT_FOUND_PREFIX + resposta

        # 4. Inserir aviso de validação humana se resposta contém sugestão clínica
        if self._has_clinical_suggestion(resposta):
            resposta = resposta + HUMAN_VALIDATION_WARNING

        # 5. Montar seção de fontes estruturadas ao final
        resposta = resposta + self._build_sources_section(sources)

        return MedicalResponse(
            resposta=resposta,
            fontes=sources,
            alertas=alertas,
            timestamp=self._now_iso(),
            consulta_original=consulta,
        )

    # ------------------------------------------------------------------
    # Construção de fontes
    # ------------------------------------------------------------------

    def _build_sources(
        self,
        retrieved_docs: list[Document],
        paciente_encontrado: bool,
    ) -> ResponseSources:
        """Constrói ``ResponseSources`` a partir dos documentos recuperados.

        Args:
            retrieved_docs: Documentos recuperados via RAG.
            paciente_encontrado: Se um prontuário foi encontrado.

        Returns:
            ``ResponseSources`` com campos do prontuário utilizados.
        """
        dados_prontuario: list[str] = []

        if retrieved_docs and paciente_encontrado:
            # Extrai os campos presentes nos documentos recuperados
            dados_prontuario = self._extract_prontuario_fields(retrieved_docs)

        return ResponseSources(
            dados_prontuario=dados_prontuario,
            conhecimento_modelo=True,  # LLM sempre contribui com conhecimento
            paciente_encontrado=paciente_encontrado,
        )

    def _extract_prontuario_fields(self, docs: list[Document]) -> list[str]:
        """Extrai os campos do prontuário presentes nos documentos recuperados.

        Analisa o conteúdo dos documentos para identificar quais seções do
        prontuário foram utilizadas como contexto.

        Args:
            docs: Documentos LangChain recuperados via RAG.

        Returns:
            Lista de campos do prontuário identificados nos documentos.
        """
        fields_found: set[str] = set()
        field_markers = {
            "Diagnósticos:": "diagnosticos",
            "Medicamentos em uso:": "medicamentos_em_uso",
            "Observações gerais:": "observacoes_gerais",
            "Exames:": "exames",
            "Paciente:": "dados_basicos",
            "Data de Nascimento:": "dados_basicos",
            "Última atualização:": "ultima_atualizacao",
        }

        for doc in docs:
            content = doc.page_content
            for marker, field_name in field_markers.items():
                if marker in content:
                    fields_found.add(field_name)

        return sorted(fields_found)

    # ------------------------------------------------------------------
    # Construção de alertas clínicos
    # ------------------------------------------------------------------

    def _build_alerts(self, prontuario: Prontuario) -> list[ClinicalAlert]:
        """Gera alertas clínicos para exames pendentes e alterados.

        Exames com status ``alterado`` geram alerta de prioridade ALTA.
        Exames com status ``pendente`` geram alerta de prioridade MEDIA.

        Args:
            prontuario: Prontuário do paciente.

        Returns:
            Lista de ``ClinicalAlert`` para os exames que requerem atenção.
        """
        alertas: list[ClinicalAlert] = []

        exames_alterados = [
            e for e in prontuario.exames if e.status == ExameStatus.ALTERADO
        ]
        exames_pendentes = [
            e for e in prontuario.exames if e.status == ExameStatus.PENDENTE
        ]

        if exames_alterados:
            alertas.append(
                ClinicalAlert(
                    tipo=AlertType.EXAME_ALTERADO,
                    prioridade=AlertPriority.ALTA,
                    descricao=(
                        f"Paciente {prontuario.nome} possui "
                        f"{len(exames_alterados)} exame(s) com resultado alterado "
                        "que requer(em) atenção imediata."
                    ),
                    exames_afetados=[e.nome for e in exames_alterados],
                )
            )

        if exames_pendentes:
            alertas.append(
                ClinicalAlert(
                    tipo=AlertType.EXAME_PENDENTE,
                    prioridade=AlertPriority.MEDIA,
                    descricao=(
                        f"Paciente {prontuario.nome} possui "
                        f"{len(exames_pendentes)} exame(s) pendente(s) "
                        "aguardando resultado."
                    ),
                    exames_afetados=[e.nome for e in exames_pendentes],
                )
            )

        return alertas

    # ------------------------------------------------------------------
    # Detecção de sugestão clínica
    # ------------------------------------------------------------------

    def _has_clinical_suggestion(self, text: str) -> bool:
        """Verifica se o texto contém sugestão de conduta clínica.

        Busca por palavras-chave que indicam sugestão clínica no texto
        da resposta (case-insensitive e sem acentos para detecção robusta).

        Args:
            text: Texto da resposta gerada.

        Returns:
            ``True`` se o texto contém sugestão clínica.
        """
        import unicodedata

        def _normalize(s: str) -> str:
            nfkd = unicodedata.normalize("NFKD", s)
            return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

        text_normalized = _normalize(text)
        return any(_normalize(kw) in text_normalized for kw in CLINICAL_SUGGESTION_KEYWORDS)

    # ------------------------------------------------------------------
    # Seção de fontes estruturadas
    # ------------------------------------------------------------------

    def _build_sources_section(self, sources: ResponseSources) -> str:
        """Monta a seção de fontes estruturadas ao final da resposta.

        Separa claramente os dados do prontuário recuperados via RAG do
        conhecimento do modelo fine-tunado, atendendo ao Requisito 7.3.

        Args:
            sources: Fontes estruturadas da resposta.

        Returns:
            Texto formatado com a seção de fontes.
        """
        lines: list[str] = ["\n\n---\n📋 **Fontes da Resposta**"]

        if sources.paciente_encontrado and sources.dados_prontuario:
            campos = ", ".join(sources.dados_prontuario)
            lines.append(f"• Dados do prontuário (via RAG): {campos}")
        elif sources.paciente_encontrado:
            lines.append("• Dados do prontuário (via RAG): informações gerais do paciente")
        else:
            lines.append("• Dados do prontuário (via RAG): nenhum registro encontrado")

        if sources.conhecimento_modelo:
            lines.append(
                "• Conhecimento do modelo: protocolos e diretrizes clínicas "
                "aprendidos durante o treinamento"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now_iso() -> str:
        """Retorna o timestamp atual em formato ISO 8601 UTC."""
        return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
