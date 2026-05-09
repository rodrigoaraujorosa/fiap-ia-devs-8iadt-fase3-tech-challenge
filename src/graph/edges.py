"""
Lógica de transição condicional entre nós do grafo LangGraph.

Define as funções de aresta condicional que determinam o próximo nó a ser
executado com base no estado atual do grafo. As arestas condicionais permitem
que o fluxo de decisão se ramifique dinamicamente conforme os dados clínicos
do paciente.
"""

from __future__ import annotations

from typing import Literal

from src.graph.state import GraphState


def should_generate_alert(
    state: GraphState,
) -> Literal["gerar_alerta", "gerar_resposta"]:
    """Determina se o fluxo deve passar pelo nó de alerta clínico.

    Avalia o estado após ``node_verificar_exames`` para decidir se há
    exames pendentes ou alterados que requerem geração de alerta antes
    da resposta final.

    Lógica de decisão:
    - Se ``exames_pendentes`` **ou** ``exames_alterados`` não estão vazios
      → redireciona para ``"gerar_alerta"`` (Nó 4)
    - Caso contrário → vai diretamente para ``"gerar_resposta"`` (Nó 5)

    Args:
        state: Estado atual do grafo após execução de ``node_verificar_exames``.

    Returns:
        ``"gerar_alerta"`` se há exames que requerem atenção clínica,
        ``"gerar_resposta"`` caso contrário.

    Example::

        # Com exames pendentes → vai para nó de alerta
        state = {"exames_pendentes": [exame1], "exames_alterados": []}
        assert should_generate_alert(state) == "gerar_alerta"

        # Sem exames críticos → vai direto para geração de resposta
        state = {"exames_pendentes": [], "exames_alterados": []}
        assert should_generate_alert(state) == "gerar_resposta"
    """
    exames_pendentes = state.get("exames_pendentes", [])
    exames_alterados = state.get("exames_alterados", [])

    if exames_pendentes or exames_alterados:
        return "gerar_alerta"

    return "gerar_resposta"
