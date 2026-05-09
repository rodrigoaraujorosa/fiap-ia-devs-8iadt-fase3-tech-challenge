"""
Módulo de fluxo de decisão LangGraph do Assistente Virtual Médico Hospitalar.

Exporta a classe principal ``MedicalDecisionGraph`` e os componentes
individuais do grafo para uso externo e testes.
"""

from src.graph.decision_graph import MedicalDecisionGraph
from src.graph.edges import should_generate_alert
from src.graph.state import GraphState

__all__ = [
    "MedicalDecisionGraph",
    "GraphState",
    "should_generate_alert",
]
