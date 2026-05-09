"""
Módulo de indexação FAISS para busca semântica em prontuários médicos.

Constrói um índice FAISS em memória a partir dos prontuários usando embeddings
multilinguais ``paraphrase-multilingual-MiniLM-L12-v2``, integrando com
LangChain via ``langchain_community.vectorstores.FAISS``.

A busca semântica retorna resultados em menos de 2 segundos para bases
com 20+ registros, conforme requisito de performance.
"""

from __future__ import annotations

from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.database.prontuario_db import ProntuarioDB
from src.utils.exceptions import DatabaseError
from src.utils.models import Prontuario


_DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _get_embeddings(model_name: str):
    """Instancia o modelo de embeddings HuggingFace.

    Tenta primeiro ``langchain_huggingface.HuggingFaceEmbeddings`` (preferido)
    e faz fallback para ``langchain_community.embeddings.HuggingFaceEmbeddings``
    para compatibilidade com versões mais antigas do LangChain.

    Args:
        model_name: Nome do modelo de embeddings no HuggingFace Hub.

    Returns:
        Instância do modelo de embeddings.

    Raises:
        DatabaseError: Se nenhuma das importações for bem-sucedida.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError:
        pass

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except ImportError as exc:
        raise DatabaseError(
            "Não foi possível importar HuggingFaceEmbeddings. "
            "Instale 'langchain-huggingface' ou 'langchain-community'.",
            context={"model_name": model_name},
        ) from exc


class ProntuarioIndexer:
    """Construtor e gerenciador do índice FAISS para prontuários médicos.

    Converte prontuários em documentos LangChain, gera embeddings com o
    modelo multilingual ``paraphrase-multilingual-MiniLM-L12-v2`` e
    constrói um índice FAISS em memória para busca semântica eficiente.

    Args:
        db: Instância de ``ProntuarioDB`` já carregada com prontuários.
        embedding_model: Nome do modelo de embeddings no HuggingFace Hub.
            Padrão: ``"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"``.

    Example::

        db = ProntuarioDB("data/prontuarios.json")
        db.load()
        indexer = ProntuarioIndexer(db)
        indexer.build()
        docs = indexer.search("exames pendentes Ana Clara", k=3)
    """

    def __init__(
        self,
        db: ProntuarioDB,
        embedding_model: str = _DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._db = db
        self._embedding_model = embedding_model
        self._faiss_index: Optional[FAISS] = None
        self._embeddings = None

    # ------------------------------------------------------------------
    # Construção do índice
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Constrói o índice FAISS em memória a partir dos prontuários.

        Converte cada prontuário em um ``Document`` LangChain usando
        ``ProntuarioDB.to_text()``, gera embeddings e indexa no FAISS.

        Raises:
            DatabaseError: Se a base de prontuários estiver vazia ou se
                ocorrer erro durante a construção do índice.
        """
        prontuarios = self._db.prontuarios
        if not prontuarios:
            raise DatabaseError(
                "Base de prontuários vazia. Carregue os prontuários antes de indexar.",
                context={"total": 0},
            )

        try:
            self._embeddings = _get_embeddings(self._embedding_model)
            documents = self._prontuarios_to_documents(prontuarios)
            self._faiss_index = FAISS.from_documents(documents, self._embeddings)
        except DatabaseError:
            raise
        except Exception as exc:
            raise DatabaseError(
                f"Erro ao construir índice FAISS: {exc}",
                context={"embedding_model": self._embedding_model},
            ) from exc

    def _prontuarios_to_documents(
        self, prontuarios: list[Prontuario]
    ) -> list[Document]:
        """Converte prontuários em documentos LangChain para indexação.

        Args:
            prontuarios: Lista de prontuários a converter.

        Returns:
            Lista de ``Document`` com texto serializado e metadados.
        """
        documents: list[Document] = []
        for prontuario in prontuarios:
            text = self._db.to_text(prontuario)
            doc = Document(
                page_content=text,
                metadata={
                    "patient_id": prontuario.id,
                    "nome": prontuario.nome,
                    "ultima_atualizacao": prontuario.ultima_atualizacao,
                },
            )
            documents.append(doc)
        return documents

    # ------------------------------------------------------------------
    # Busca semântica
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 3) -> list[Document]:
        """Realiza busca semântica no índice FAISS.

        Args:
            query: Consulta em linguagem natural (PT-BR).
            k: Número máximo de documentos a retornar. Padrão: ``3``.

        Returns:
            Lista de até ``k`` documentos mais relevantes para a consulta,
            ordenados por similaridade decrescente.

        Raises:
            DatabaseError: Se o índice não foi construído ou se ocorrer
                erro durante a busca.
        """
        if self._faiss_index is None:
            raise DatabaseError(
                "Índice FAISS não construído. Chame build() antes de search().",
                context={"query": query},
            )

        try:
            results = self._faiss_index.similarity_search(query, k=k)
            return results
        except Exception as exc:
            raise DatabaseError(
                f"Erro durante busca semântica: {exc}",
                context={"query": query, "k": k},
            ) from exc

    def search_with_score(
        self, query: str, k: int = 3
    ) -> list[tuple[Document, float]]:
        """Realiza busca semântica retornando documentos com scores de similaridade.

        Args:
            query: Consulta em linguagem natural (PT-BR).
            k: Número máximo de documentos a retornar. Padrão: ``3``.

        Returns:
            Lista de tuplas ``(Document, score)`` ordenadas por score
            decrescente (maior score = maior similaridade).

        Raises:
            DatabaseError: Se o índice não foi construído ou se ocorrer
                erro durante a busca.
        """
        if self._faiss_index is None:
            raise DatabaseError(
                "Índice FAISS não construído. Chame build() antes de search_with_score().",
                context={"query": query},
            )

        try:
            return self._faiss_index.similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise DatabaseError(
                f"Erro durante busca semântica com score: {exc}",
                context={"query": query, "k": k},
            ) from exc

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """Retorna ``True`` se o índice FAISS foi construído."""
        return self._faiss_index is not None

    @property
    def faiss_index(self) -> Optional[FAISS]:
        """Retorna o índice FAISS interno (ou ``None`` se não construído)."""
        return self._faiss_index
