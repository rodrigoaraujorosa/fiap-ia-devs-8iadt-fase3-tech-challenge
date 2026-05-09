"""
Montagem e execução do grafo LangGraph do Assistente Virtual Médico Hospitalar.

Define a classe ``MedicalDecisionGraph`` que constrói o grafo de estados
conectando os 6 nós com as arestas corretas, incluindo a aresta condicional
após ``node_verificar_exames``.

Topologia do grafo:
    START
      ↓
    node_receber_consulta
      ↓
    node_recuperar_prontuario
      ↓
    node_verificar_exames
      ↓ (condicional: should_generate_alert)
    ┌─────────────────────────────────────┐
    │ "gerar_alerta"    │ "gerar_resposta" │
    ↓                   ↓                  │
    node_gerar_alerta → node_gerar_resposta ┘
                          ↓
                    node_formatar_resposta
                          ↓
                         END
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from src.graph.edges import should_generate_alert
from src.graph.nodes import (
    node_formatar_resposta,
    node_gerar_alerta,
    node_gerar_resposta,
    node_receber_consulta,
    node_recuperar_prontuario,
    node_verificar_exames,
)
from src.graph.state import GraphState
from src.pipeline.prompt_templates import GENERIC_ERROR_MESSAGE
from src.utils.exceptions import GraphExecutionError
from src.utils.models import MedicalResponse, ResponseSources


def _now_iso() -> str:
    """Retorna o timestamp atual em formato ISO 8601 UTC."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MedicalDecisionGraph:
    """Grafo de decisão clínica implementado com LangGraph.

    Orquestra o fluxo completo de processamento de uma consulta médica:
    recebimento → RAG → verificação de exames → (alerta opcional) →
    geração de resposta → formatação final.

    O grafo injeta dependências (``ProntuarioDB``, ``ProntuarioRetriever``,
    ``AuditLogger``, pipeline LLM) no estado inicial, tornando os nós
    testáveis de forma independente.

    Args:
        db: Instância de ``ProntuarioDB`` já carregada com prontuários.
        retriever: Instância de ``ProntuarioRetriever`` com índice FAISS
            já construído. Pode ser ``None`` para testes sem RAG.
        logger: Instância de ``AuditLogger`` para registro de auditoria.
            Pode ser ``None`` para testes sem logging.
        llm_pipeline: Pipeline HuggingFace de geração de texto já carregado.
            Pode ser ``None`` para testes sem LLM.

    Example::

        db = ProntuarioDB("data/prontuarios.json")
        db.load()
        retriever = ProntuarioRetriever(db)
        retriever.build_index()
        logger = AuditLogger()
        graph = MedicalDecisionGraph(db=db, retriever=retriever, logger=logger)
        graph.build()
        response = graph.run("Quais exames estão pendentes para Ana Clara?")
    """

    def __init__(
        self,
        db: Any = None,
        retriever: Any = None,
        logger: Any = None,
        llm_pipeline: Any = None,
    ) -> None:
        self._db = db
        self._retriever = retriever
        self._logger = logger
        self._llm_pipeline = llm_pipeline
        self._compiled_graph: Any = None
        self._is_built = False

    # ------------------------------------------------------------------
    # Construção do grafo
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Constrói e compila o grafo LangGraph.

        Registra os 6 nós, define as arestas sequenciais e a aresta
        condicional após ``node_verificar_exames``.

        Raises:
            GraphExecutionError: Se ocorrer erro durante a construção.
        """
        try:
            graph_builder = StateGraph(GraphState)

            # ── Registrar nós ──────────────────────────────────────────
            graph_builder.add_node("node_receber_consulta", node_receber_consulta)
            graph_builder.add_node("node_recuperar_prontuario", node_recuperar_prontuario)
            graph_builder.add_node("node_verificar_exames", node_verificar_exames)
            graph_builder.add_node("node_gerar_alerta", node_gerar_alerta)
            graph_builder.add_node("node_gerar_resposta", node_gerar_resposta)
            graph_builder.add_node("node_formatar_resposta", node_formatar_resposta)

            # ── Arestas sequenciais ────────────────────────────────────
            graph_builder.add_edge(START, "node_receber_consulta")
            graph_builder.add_edge("node_receber_consulta", "node_recuperar_prontuario")
            graph_builder.add_edge("node_recuperar_prontuario", "node_verificar_exames")

            # ── Aresta condicional após verificação de exames ──────────
            # should_generate_alert retorna "gerar_alerta" ou "gerar_resposta"
            graph_builder.add_conditional_edges(
                "node_verificar_exames",
                should_generate_alert,
                {
                    "gerar_alerta": "node_gerar_alerta",
                    "gerar_resposta": "node_gerar_resposta",
                },
            )

            # ── Aresta do nó de alerta para geração de resposta ────────
            graph_builder.add_edge("node_gerar_alerta", "node_gerar_resposta")

            # ── Aresta final ───────────────────────────────────────────
            graph_builder.add_edge("node_gerar_resposta", "node_formatar_resposta")
            graph_builder.add_edge("node_formatar_resposta", END)

            # ── Compilar grafo ─────────────────────────────────────────
            self._compiled_graph = graph_builder.compile()
            self._is_built = True

        except Exception as exc:
            raise GraphExecutionError(
                f"Erro ao construir grafo LangGraph: {exc}",
                context={"component": "MedicalDecisionGraph.build"},
            ) from exc

    # ------------------------------------------------------------------
    # Execução do grafo
    # ------------------------------------------------------------------

    def run(self, consulta: str) -> MedicalResponse:
        """Executa o grafo completo para uma consulta médica.

        Injeta as dependências no estado inicial e executa o grafo compilado.
        Captura erros de qualquer nó como ``GraphExecutionError``, loga com
        stack trace e retorna mensagem genérica ao usuário.

        Args:
            consulta: Consulta clínica em PT-BR submetida pelo médico.

        Returns:
            ``MedicalResponse`` com resposta formatada, fontes e alertas.

        Raises:
            GraphExecutionError: Se o grafo não foi construído antes de ``run()``.
        """
        if not self._is_built:
            raise GraphExecutionError(
                "Grafo não construído. Chame build() antes de run().",
                context={"consulta": consulta},
            )

        # Estado inicial com dependências injetadas via campos privados
        # (prefixo "_" indica que são dependências, não dados do fluxo)
        initial_state: GraphState = {
            "consulta": consulta,
            "prontuarios_recuperados": [],
            "paciente_identificado": None,
            "exames_pendentes": [],
            "exames_alterados": [],
            "alertas": [],
            "resposta_llm": "",
            "resposta_final": None,  # type: ignore[typeddict-item]
            "erro": None,
            "log_entries": [],
            # Dependências injetadas (acessadas pelos nós via state.get)
            "_db": self._db,  # type: ignore[typeddict-unknown-key]
            "_retriever": self._retriever,  # type: ignore[typeddict-unknown-key]
            "_logger": self._logger,  # type: ignore[typeddict-unknown-key]
            "_llm_pipeline": self._llm_pipeline,  # type: ignore[typeddict-unknown-key]
        }

        try:
            final_state = self._compiled_graph.invoke(initial_state)

            resposta_final: Optional[MedicalResponse] = final_state.get("resposta_final")

            if resposta_final is None:
                # Fallback: estado sem resposta formatada
                resposta_final = MedicalResponse(
                    resposta=GENERIC_ERROR_MESSAGE,
                    fontes=ResponseSources(
                        dados_prontuario=[],
                        conhecimento_modelo=False,
                        paciente_encontrado=False,
                    ),
                    alertas=[],
                    timestamp=_now_iso(),
                    consulta_original=consulta,
                )

            return resposta_final

        except GraphExecutionError as exc:
            # Loga e retorna mensagem genérica
            if self._logger:
                self._logger.log_error(
                    exc,
                    context={
                        "consulta": consulta,
                        "component": "MedicalDecisionGraph.run",
                    },
                )
            return MedicalResponse(
                resposta=GENERIC_ERROR_MESSAGE,
                fontes=ResponseSources(
                    dados_prontuario=[],
                    conhecimento_modelo=False,
                    paciente_encontrado=False,
                ),
                alertas=[],
                timestamp=_now_iso(),
                consulta_original=consulta,
            )

        except Exception as exc:
            tb = traceback.format_exc()
            if self._logger:
                self._logger.log_error(
                    exc,
                    context={
                        "consulta": consulta,
                        "component": "MedicalDecisionGraph.run",
                        "stack_trace": tb,
                    },
                )
            return MedicalResponse(
                resposta=GENERIC_ERROR_MESSAGE,
                fontes=ResponseSources(
                    dados_prontuario=[],
                    conhecimento_modelo=False,
                    paciente_encontrado=False,
                ),
                alertas=[],
                timestamp=_now_iso(),
                consulta_original=consulta,
            )

    # ------------------------------------------------------------------
    # Geração do diagrama visual
    # ------------------------------------------------------------------

    def save_diagram(self, output_path: str = "images/decision_flow.png") -> str:
        """Gera e salva o diagrama visual do grafo como imagem PNG.

        Usa a API nativa do LangGraph para gerar o diagrama Mermaid e
        converte para PNG via ``mermaid-py`` ou ``playwright``. Se nenhuma
        biblioteca de renderização estiver disponível, salva o diagrama
        Mermaid como fallback em arquivo de texto.

        Args:
            output_path: Caminho de saída para o arquivo PNG.
                Padrão: ``"images/decision_flow.png"``.

        Returns:
            Caminho do arquivo gerado.

        Raises:
            GraphExecutionError: Se o grafo não foi construído.
        """
        import os

        if not self._is_built:
            raise GraphExecutionError(
                "Grafo não construído. Chame build() antes de save_diagram().",
                context={"output_path": output_path},
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Tenta gerar PNG via API do LangGraph (requer Pillow ou playwright)
        try:
            png_bytes: bytes = self._compiled_graph.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            return output_path
        except Exception:
            pass

        # Fallback 1: salvar diagrama Mermaid como .mmd e gerar PNG com matplotlib
        try:
            mermaid_str: str = self._compiled_graph.get_graph().draw_mermaid()
            self._render_mermaid_with_matplotlib(mermaid_str, output_path)
            return output_path
        except Exception:
            pass

        # Fallback 2: salvar diagrama Mermaid como texto (sem PNG)
        mmd_path = output_path.replace(".png", ".mmd")
        try:
            mermaid_str = self._compiled_graph.get_graph().draw_mermaid()
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid_str)
        except Exception:
            # Último recurso: diagrama estático embutido
            mermaid_str = self._static_mermaid_diagram()
            with open(mmd_path, "w", encoding="utf-8") as f:
                f.write(mermaid_str)

        # Gera PNG estático com matplotlib como último recurso
        try:
            self._render_mermaid_with_matplotlib(mermaid_str, output_path)
        except Exception:
            pass

        return output_path

    def _render_mermaid_with_matplotlib(
        self, mermaid_str: str, output_path: str
    ) -> None:
        """Renderiza um diagrama Mermaid como PNG usando matplotlib.

        Cria uma representação visual simplificada do grafo usando matplotlib,
        adequada para documentação quando bibliotecas de renderização Mermaid
        não estão disponíveis.

        Args:
            mermaid_str: Diagrama em formato Mermaid.
            output_path: Caminho de saída para o PNG.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis("off")
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")

        # Título
        ax.text(
            5, 13.5,
            "Fluxo de Decisão — Assistente Médico Hospitalar",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#2c3e50",
        )

        # Nós e posições
        nodes = [
            (5, 12.5, "START", "#27ae60", "white"),
            (5, 11.0, "Nó 1: Receber Consulta\n(identifica paciente)", "#2980b9", "white"),
            (5, 9.5,  "Nó 2: Recuperar Prontuário\n(RAG via FAISS)", "#2980b9", "white"),
            (5, 8.0,  "Nó 3: Verificar Exames\n(pendentes / alterados)", "#2980b9", "white"),
            (2, 6.0,  "Nó 4: Gerar Alerta\n(prioridade ALTA/MÉDIA)", "#e74c3c", "white"),
            (5, 4.5,  "Nó 5: Gerar Resposta\n(LLM fine-tunada)", "#2980b9", "white"),
            (5, 3.0,  "Nó 6: Formatar Resposta\n(fontes + alertas)", "#2980b9", "white"),
            (5, 1.5,  "END", "#27ae60", "white"),
        ]

        for x, y, label, color, text_color in nodes:
            box = mpatches.FancyBboxPatch(
                (x - 2.2, y - 0.45), 4.4, 0.9,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor="#2c3e50", linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                x, y, label,
                ha="center", va="center", fontsize=8.5,
                color=text_color, fontweight="bold",
            )

        # Setas sequenciais
        arrow_props = dict(arrowstyle="->", color="#2c3e50", lw=1.5)
        for (x1, y1), (x2, y2) in [
            ((5, 12.05), (5, 11.45)),
            ((5, 10.55), (5, 9.95)),
            ((5, 9.05),  (5, 8.45)),
        ]:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=arrow_props)

        # Aresta condicional: verificar_exames → gerar_alerta
        ax.annotate("", xy=(2, 6.45), xytext=(3.8, 7.55),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5))
        ax.text(2.5, 7.1, "exames\npendentes/\nalterados",
                ha="center", va="center", fontsize=7, color="#e74c3c")

        # Aresta condicional: verificar_exames → gerar_resposta (sem alertas)
        ax.annotate("", xy=(5, 4.95), xytext=(5, 7.55),
                    arrowprops=dict(arrowstyle="->", color="#27ae60", lw=1.5,
                                   linestyle="dashed"))
        ax.text(5.8, 6.2, "sem\nalertas", ha="center", va="center",
                fontsize=7, color="#27ae60")

        # Aresta: gerar_alerta → gerar_resposta
        ax.annotate("", xy=(3.8, 4.95), xytext=(2, 5.55),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5))

        # Setas finais
        for (x1, y1), (x2, y2) in [
            ((5, 4.05), (5, 3.45)),
            ((5, 2.55), (5, 1.95)),
        ]:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=arrow_props)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    @staticmethod
    def _static_mermaid_diagram() -> str:
        """Retorna o diagrama Mermaid estático do grafo como string."""
        return """stateDiagram-v2
    [*] --> RecebimentoConsulta

    RecebimentoConsulta : Nó 1 - Recebimento da Consulta (identifica paciente)
    RecuperacaoRAG : Nó 2 - Recuperação via RAG (FAISS + embeddings)
    VerificacaoExames : Nó 3 - Verificação de Exames (pendentes / alterados)
    GeracaoAlerta : Nó 4 - Geração de Alerta (prioridade ALTA/MÉDIA)
    GeracaoResposta : Nó 5 - Geração de Resposta (LLM fine-tunada)
    FormatacaoFinal : Nó 6 - Formatação Final (resposta + fontes)

    RecebimentoConsulta --> RecuperacaoRAG : consulta validada
    RecuperacaoRAG --> VerificacaoExames : prontuários recuperados
    VerificacaoExames --> GeracaoAlerta : exames pendentes/alterados encontrados
    VerificacaoExames --> GeracaoResposta : sem alertas clínicos
    GeracaoAlerta --> GeracaoResposta : alertas gerados
    GeracaoResposta --> FormatacaoFinal : resposta LLM gerada
    FormatacaoFinal --> [*] : MedicalResponse retornada
"""

    # ------------------------------------------------------------------
    # Propriedades
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        """Retorna ``True`` se o grafo foi construído e compilado."""
        return self._is_built

    @property
    def compiled_graph(self) -> Any:
        """Retorna o grafo compilado pelo LangGraph."""
        return self._compiled_graph
