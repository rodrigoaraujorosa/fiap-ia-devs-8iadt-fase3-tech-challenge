"""
Pipeline LangChain com RAG para o Assistente Virtual Médico Hospitalar.

Integra LLM fine-tunada + FAISS RAG + formatação de resposta em um pipeline
coeso que processa consultas médicas em Português Brasileiro.

Fluxo de uma consulta:
1. Verificar se é solicitação de prescrição → recusar imediatamente
2. Recuperar documentos relevantes via FAISS RAG
3. Verificar se paciente foi encontrado
4. Invocar LLM com contexto do prontuário
5. Formatar resposta com fontes e alertas
6. Registrar interação no AuditLogger
"""

from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.documents import Document

from src.database.prontuario_db import ProntuarioDB
from src.pipeline.prompt_templates import (
    GENERIC_ERROR_MESSAGE,
    MEDICAL_RAG_TEMPLATE,
    MEDICAL_SYSTEM_PROMPT,
    PATIENT_NOT_FOUND_PREFIX,
    PRESCRIPTION_KEYWORDS,
    PRESCRIPTION_REFUSAL_MESSAGE,
)
from src.pipeline.response_formatter import ResponseFormatter
from src.pipeline.retriever import ProntuarioRetriever
from src.utils.exceptions import PipelineError
from src.utils.logger import AuditLogger
from src.utils.models import (
    AlertPriority,
    AlertType,
    ClinicalAlert,
    MedicalResponse,
    Prontuario,
    ResponseSources,
)

_DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class MedicalRAGPipeline:
    """Pipeline LangChain com RAG para consultas médicas.

    Integra LLM fine-tunada, recuperação semântica via FAISS e formatação
    de resposta estruturada com fontes e alertas clínicos.

    Args:
        model_path: Caminho para o modelo fine-tunado local ou nome de um
            modelo no HuggingFace Hub (para testes sem modelo fine-tunado).
        db: Instância de ``ProntuarioDB`` já carregada com prontuários.
        logger: Instância de ``AuditLogger`` para registro de interações.

    Example::

        db = ProntuarioDB("data/prontuarios.json")
        db.load()
        logger = AuditLogger()
        pipeline = MedicalRAGPipeline(
            model_path="models/mistral-7b-assistente-hospitalar",
            db=db,
            logger=logger,
        )
        pipeline.build()
        response = pipeline.query("Quais exames estão pendentes para Ana Clara?")
    """

    def __init__(
        self,
        model_path: str,
        db: ProntuarioDB,
        logger: AuditLogger,
    ) -> None:
        self._model_path = model_path
        self._db = db
        self._logger = logger
        self._retriever: Optional[ProntuarioRetriever] = None
        self._formatter = ResponseFormatter()
        self._llm_pipeline: Any = None
        self._is_built = False

    # ------------------------------------------------------------------
    # Inicialização do pipeline
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Inicializa o pipeline: carrega modelo, constrói índice FAISS, configura chain.

        Carrega o modelo LLM (local ou HuggingFace Hub) e constrói o índice
        FAISS para recuperação semântica de prontuários.

        Raises:
            PipelineError: Se ocorrer erro durante a inicialização.
        """
        try:
            # 1. Construir índice FAISS via ProntuarioRetriever
            self._retriever = ProntuarioRetriever(
                db=self._db,
                embedding_model=_DEFAULT_EMBEDDING_MODEL,
            )
            self._retriever.build_index()

            # 2. Carregar modelo LLM
            self._llm_pipeline = self._load_model(self._model_path)

            self._is_built = True

        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Erro ao inicializar pipeline: {exc}",
                context={"model_path": self._model_path},
            ) from exc

    def _load_model(self, model_path: str) -> Any:
        """Carrega o modelo LLM para inferência.

        Suporta modelos locais fine-tunados e modelos do HuggingFace Hub.
        Para modelos locais com adaptadores LoRA, carrega o modelo base e
        aplica os pesos do adaptador.

        Args:
            model_path: Caminho local ou nome do modelo no HuggingFace Hub.

        Returns:
            Pipeline de geração de texto do transformers.

        Raises:
            PipelineError: Se o modelo não puder ser carregado.
        """
        try:
            from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM
            import os

            # Verifica se é um caminho local com adaptador LoRA (PEFT)
            is_local = os.path.isdir(model_path)
            has_adapter = is_local and os.path.exists(
                os.path.join(model_path, "adapter_config.json")
            )

            if has_adapter:
                # Carrega modelo base + adaptador LoRA
                try:
                    from peft import PeftModel, PeftConfig
                    peft_config = PeftConfig.from_pretrained(model_path)
                    base_model_name = peft_config.base_model_name_or_path
                    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        device_map="cpu",
                        torch_dtype="auto",
                    )
                    model = PeftModel.from_pretrained(base_model, model_path)
                    return hf_pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=1.0,
                    )
                except Exception as peft_exc:
                    raise PipelineError(
                        f"Erro ao carregar modelo com adaptador LoRA: {peft_exc}",
                        context={"model_path": model_path},
                    ) from peft_exc
            else:
                # Carrega modelo diretamente (local sem adaptador ou HuggingFace Hub)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return hf_pipeline(
                    "text-generation",
                    model=model_path,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=1.0,
                    device_map="cpu",
                )

        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                f"Erro ao carregar modelo '{model_path}': {exc}",
                context={"model_path": model_path},
            ) from exc

    # ------------------------------------------------------------------
    # Processamento de consultas
    # ------------------------------------------------------------------

    def query(self, consulta: str) -> MedicalResponse:
        """Processa uma consulta médica e retorna resposta com fontes.

        Fluxo:
        1. Verificar se é solicitação de prescrição → recusar imediatamente
        2. Recuperar documentos relevantes via FAISS RAG
        3. Verificar se paciente foi encontrado
        4. Invocar LLM com contexto do prontuário
        5. Formatar resposta com fontes e alertas
        6. Registrar interação no AuditLogger

        Args:
            consulta: Pergunta clínica em PT-BR.

        Returns:
            ``MedicalResponse`` com resposta, fontes e alertas.

        Raises:
            PipelineError: Em caso de erro interno (logado, mensagem genérica ao usuário).
        """
        if not self._is_built:
            raise PipelineError(
                "Pipeline não inicializado. Chame build() antes de query().",
                context={"consulta": consulta},
            )

        try:
            # 1. Verificar se é solicitação de prescrição (7.6)
            if self._is_prescription_request(consulta):
                return self._build_refusal_response(consulta)

            # 2. Recuperar documentos via RAG (7.2)
            retrieved_docs = self._retriever.retrieve(consulta, k=3)  # type: ignore[union-attr]

            # 3. Verificar se paciente foi encontrado (7.7)
            paciente_encontrado = len(retrieved_docs) > 0
            prontuario = self._find_prontuario_from_docs(retrieved_docs)

            # 4. Invocar LLM com contexto do prontuário
            raw_response = self._invoke_llm(consulta, retrieved_docs, paciente_encontrado)

            # 5. Formatar resposta com fontes e alertas (7.3, 7.5)
            response = self._formatter.format(
                raw_response=raw_response,
                retrieved_docs=retrieved_docs,
                consulta=consulta,
                prontuario=prontuario,
                paciente_encontrado=paciente_encontrado,
            )

            # 6. Registrar interação no AuditLogger
            self._logger.log_interaction(
                consulta=consulta,
                resposta=response,
                sources=response.fontes,
            )

            return response

        except PipelineError:
            raise
        except Exception as exc:
            self._logger.log_error(exc, context={"consulta": consulta, "node": "query"})
            raise PipelineError(
                GENERIC_ERROR_MESSAGE,
                context={"consulta": consulta},
            ) from exc

    # ------------------------------------------------------------------
    # Detecção de prescrição
    # ------------------------------------------------------------------

    def _is_prescription_request(self, consulta: str) -> bool:
        """Verifica se a consulta solicita geração de prescrição médica.

        Realiza comparação case-insensitive e sem acentos para garantir
        detecção robusta independente da grafia utilizada.

        Args:
            consulta: Texto da consulta do médico.

        Returns:
            ``True`` se a consulta solicita prescrição médica direta.
        """
        import unicodedata

        def _normalize(text: str) -> str:
            nfkd = unicodedata.normalize("NFKD", text)
            return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

        consulta_normalized = _normalize(consulta)
        return any(_normalize(kw) in consulta_normalized for kw in PRESCRIPTION_KEYWORDS)

    def _build_refusal_response(self, consulta: str) -> MedicalResponse:
        """Constrói resposta de recusa para solicitações de prescrição.

        Args:
            consulta: Consulta original do médico.

        Returns:
            ``MedicalResponse`` com mensagem de recusa.
        """
        from datetime import datetime, timezone

        sources = ResponseSources(
            dados_prontuario=[],
            conhecimento_modelo=False,
            paciente_encontrado=False,
        )

        return MedicalResponse(
            resposta=PRESCRIPTION_REFUSAL_MESSAGE,
            fontes=sources,
            alertas=[],
            timestamp=datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            consulta_original=consulta,
        )

    # ------------------------------------------------------------------
    # Invocação da LLM
    # ------------------------------------------------------------------

    def _invoke_llm(
        self,
        consulta: str,
        retrieved_docs: list[Document],
        paciente_encontrado: bool,
    ) -> str:
        """Invoca a LLM com o contexto do prontuário para gerar resposta.

        Monta o prompt RAG com o contexto dos documentos recuperados e
        invoca o modelo de linguagem para geração da resposta.

        Args:
            consulta: Consulta original do médico.
            retrieved_docs: Documentos recuperados via RAG.
            paciente_encontrado: Se um prontuário foi encontrado.

        Returns:
            Texto bruto gerado pela LLM.
        """
        # Montar contexto a partir dos documentos recuperados
        if retrieved_docs and paciente_encontrado:
            context = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
        else:
            context = "Nenhum prontuário encontrado para o paciente mencionado."

        # Montar prompt completo
        prompt = (
            f"{MEDICAL_SYSTEM_PROMPT}\n\n"
            + MEDICAL_RAG_TEMPLATE.format(context=context, question=consulta)
        )

        try:
            result = self._llm_pipeline(prompt)
            # O pipeline HuggingFace retorna lista de dicts com 'generated_text'
            if isinstance(result, list) and result:
                generated = result[0].get("generated_text", "")
                # Remove o prompt do início da resposta gerada (alguns modelos repetem)
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                return generated or "Não foi possível gerar uma resposta para esta consulta."
            return "Não foi possível gerar uma resposta para esta consulta."
        except Exception as exc:
            raise PipelineError(
                f"Erro durante inferência da LLM: {exc}",
                context={"consulta": consulta},
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_prontuario_from_docs(
        self, docs: list[Document]
    ) -> Optional[Prontuario]:
        """Encontra o prontuário correspondente ao primeiro documento recuperado.

        Usa o ``patient_id`` dos metadados do documento para buscar o
        prontuário completo na base de dados.

        Args:
            docs: Documentos recuperados via RAG.

        Returns:
            Prontuário correspondente ou ``None`` se não encontrado.
        """
        if not docs:
            return None

        # Tenta encontrar pelo patient_id nos metadados
        for doc in docs:
            patient_id = doc.metadata.get("patient_id")
            if patient_id:
                prontuario = self._db._index_by_id.get(patient_id)
                if prontuario:
                    return prontuario

        return None

    @property
    def is_built(self) -> bool:
        """Retorna ``True`` se o pipeline foi inicializado."""
        return self._is_built
