"""
Módulo de recuperação semântica de prontuários via FAISS.

Constrói um índice FAISS em memória a partir dos prontuários e implementa
busca semântica usando embeddings multilinguais para recuperar os documentos
mais relevantes para uma consulta médica.
"""

from __future__ import annotations

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.database.indexer import ProntuarioIndexer, _get_embeddings, _DEFAULT_EMBEDDING_MODEL
from src.database.prontuario_db import ProntuarioDB
from src.utils.exceptions import DatabaseError, PipelineError


class ProntuarioRetriever:
    """Recuperador semântico de prontuários usando FAISS.

    Constrói um índice FAISS em memória a partir dos prontuários carregados
    no ``ProntuarioDB`` e implementa busca semântica para recuperar os
    documentos mais relevantes para uma consulta médica.

    Args:
        db: Instância de ``ProntuarioDB`` já carregada com prontuários.
        embedding_model: Nome do modelo de embeddings no HuggingFace Hub.
            Padrão: ``"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"``.

    Example::

        db = ProntuarioDB("data/prontuarios.json")
        db.load()
        retriever = ProntuarioRetriever(db)
        retriever.build_index()
        docs = retriever.retrieve("exames pendentes Ana Clara", k=3)
    """

    def __init__(
        self,
        db: ProntuarioDB,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._db = db
        self._embedding_model = embedding_model
        self._indexer: Optional[ProntuarioIndexer] = None

    # ------------------------------------------------------------------
    # Construção do índice
    # ------------------------------------------------------------------

    def build_index(self) -> FAISS:
        """Constrói o índice FAISS em memória a partir dos prontuários.

        Delega a construção ao ``ProntuarioIndexer`` existente, reutilizando
        a lógica de embeddings e indexação já implementada.

        Returns:
            Instância do índice FAISS construído.

        Raises:
            PipelineError: Se a base de prontuários estiver vazia ou se
                ocorrer erro durante a construção do índice.
        """
        try:
            self._indexer = ProntuarioIndexer(
                db=self._db,
                embedding_model=self._embedding_model,
            )
            self._indexer.build()
            return self._indexer.faiss_index  # type: ignore[return-value]
        except DatabaseError as exc:
            raise PipelineError(
                f"Erro ao construir índice FAISS para recuperação: {exc}",
                context={"embedding_model": self._embedding_model},
            ) from exc
        except Exception as exc:
            raise PipelineError(
                f"Erro inesperado ao construir índice FAISS: {exc}",
                context={"embedding_model": self._embedding_model},
            ) from exc

    # ------------------------------------------------------------------
    # Recuperação semântica
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Recupera os k documentos mais relevantes para a consulta.

        Realiza busca semântica no índice FAISS usando o modelo de embeddings
        multilingual para encontrar os prontuários mais relevantes.

        Args:
            query: Consulta em linguagem natural (PT-BR).
            k: Número máximo de documentos a retornar. Padrão: ``3``.

        Returns:
            Lista de até ``k`` documentos LangChain mais relevantes para a
            consulta, ordenados por similaridade decrescente.

        Raises:
            PipelineError: Se o índice não foi construído ou se ocorrer
                erro durante a busca.
        """
        if self._indexer is None or not self._indexer.is_built:
            raise PipelineError(
                "Índice FAISS não construído. Chame build_index() antes de retrieve().",
                context={"query": query},
            )

        try:
            return self._indexer.search(query, k=k)
        except DatabaseError as exc:
            raise PipelineError(
                f"Erro durante recuperação semântica: {exc}",
                context={"query": query, "k": k},
            ) from exc
        except Exception as exc:
            raise PipelineError(
                f"Erro inesperado durante recuperação semântica: {exc}",
                context={"query": query, "k": k},
            ) from exc

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """Retorna ``True`` se o índice FAISS foi construído."""
        return self._indexer is not None and self._indexer.is_built
