"""
Hierarquia de exceções do Assistente Virtual Médico Hospitalar.

Todas as exceções do sistema derivam de `AssistenteMedicoError`, permitindo
captura genérica ou específica conforme a necessidade do chamador.
"""

from __future__ import annotations

from typing import Any


class AssistenteMedicoError(Exception):
    """Exceção base para todos os erros do sistema.

    Todas as exceções customizadas do Assistente Virtual Médico Hospitalar
    herdam desta classe, possibilitando captura centralizada com um único
    bloco ``except AssistenteMedicoError``.

    Args:
        message: Mensagem descritiva do erro. Padrão: string vazia.
        context: Dicionário com informações adicionais de contexto (ex.:
            identificadores de paciente, nome do módulo, parâmetros de
            entrada). Padrão: ``None``.
    """

    def __init__(
        self,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, context={self.context!r})"
        )


class DatabaseError(AssistenteMedicoError):
    """Erro ocorrido no módulo de banco de dados de prontuários.

    Disparada pelo módulo ``src/database/`` quando operações de leitura,
    indexação ou busca em prontuários falham, incluindo erros de
    carregamento do arquivo JSON, construção do índice FAISS ou
    execução de consultas.

    Args:
        message: Mensagem descritiva do erro.
        context: Informações adicionais, como o caminho do arquivo de
            dados ou o identificador do paciente pesquisado.
    """

    def __init__(
        self,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)


class PipelineError(AssistenteMedicoError):
    """Erro ocorrido no pipeline LangChain com RAG.

    Disparada pelo módulo ``src/pipeline/`` quando o pipeline de
    recuperação e geração falha, incluindo erros na construção do
    índice de recuperação, na formatação de prompts ou na integração
    entre os componentes LangChain.

    Args:
        message: Mensagem descritiva do erro.
        context: Informações adicionais, como a consulta que originou
            o erro ou o componente do pipeline que falhou.
    """

    def __init__(
        self,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)


class GraphExecutionError(AssistenteMedicoError):
    """Erro ocorrido durante a execução do grafo LangGraph.

    Disparada pelo módulo ``src/graph/`` quando a execução do fluxo de
    decisão falha em qualquer nó do grafo, seja na recepção da consulta,
    recuperação de prontuários, verificação de exames, geração de alertas
    ou formatação da resposta final.

    Args:
        message: Mensagem descritiva do erro.
        context: Informações adicionais, como o nome do nó onde ocorreu
            a falha, o estado do grafo no momento do erro ou a consulta
            original do usuário.
    """

    def __init__(
        self,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)


class ModelInferenceError(AssistenteMedicoError):
    """Erro ocorrido durante a inferência do modelo de linguagem (LLM).

    Disparada quando a inferência do modelo fine-tunado falha, seja por
    indisponibilidade do modelo, por entrada que excede o comprimento
    máximo de contexto, por erro de alocação de memória GPU/CPU ou por
    qualquer outra falha durante a geração de texto pelo LLM.

    Args:
        message: Mensagem descritiva do erro de inferência.
        context: Informações adicionais, como o nome do modelo, o
            comprimento do prompt de entrada ou o código de erro
            retornado pelo framework de inferência.
    """

    def __init__(
        self,
        message: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)
