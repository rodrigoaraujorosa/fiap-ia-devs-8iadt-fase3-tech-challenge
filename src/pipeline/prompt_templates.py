"""
Templates de prompt para o Assistente Virtual Médico Hospitalar.

Define os prompts do sistema e templates RAG em Português Brasileiro,
incluindo instruções de segurança: aviso de validação humana e recusa
de prescrições médicas diretas.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Prompt do sistema
# ---------------------------------------------------------------------------

MEDICAL_SYSTEM_PROMPT = """Você é um assistente médico hospitalar especializado.
Responda SEMPRE em Português Brasileiro.
Use as informações do prontuário fornecidas para contextualizar sua resposta.
IMPORTANTE: Toda sugestão clínica requer validação de profissional de saúde habilitado.
Não gere prescrições médicas diretas."""

# ---------------------------------------------------------------------------
# Template RAG
# ---------------------------------------------------------------------------

MEDICAL_RAG_TEMPLATE = """Contexto do prontuário:
{context}

Consulta do médico: {question}

Resposta:"""

# ---------------------------------------------------------------------------
# Mensagem de recusa de prescrição
# ---------------------------------------------------------------------------

PRESCRIPTION_REFUSAL_MESSAGE = (
    "Não é possível gerar prescrições médicas diretas. "
    "A emissão de prescrições está fora do escopo deste assistente e requer "
    "avaliação e assinatura de um profissional de saúde habilitado. "
    "Posso auxiliar com informações clínicas gerais, revisão de prontuários "
    "e sugestões de condutas que devem ser validadas pelo médico responsável."
)

# ---------------------------------------------------------------------------
# Aviso de validação humana (inserido como pós-processamento)
# ---------------------------------------------------------------------------

HUMAN_VALIDATION_WARNING = (
    "\n\n⚠️ AVISO: Esta resposta contém sugestões clínicas que requerem "
    "validação e aprovação de um profissional de saúde habilitado antes de "
    "serem adotadas. O assistente não substitui o julgamento clínico médico."
)

# ---------------------------------------------------------------------------
# Mensagem para paciente não encontrado
# ---------------------------------------------------------------------------

PATIENT_NOT_FOUND_PREFIX = (
    "Não foram encontrados dados de prontuário para o paciente mencionado "
    "na base de dados. A resposta a seguir é baseada exclusivamente no "
    "conhecimento do modelo:\n\n"
)

# ---------------------------------------------------------------------------
# Mensagem de erro genérica
# ---------------------------------------------------------------------------

GENERIC_ERROR_MESSAGE = (
    "Ocorreu um erro interno ao processar sua consulta. "
    "Por favor, tente novamente. Se o problema persistir, "
    "entre em contato com o suporte técnico."
)

# ---------------------------------------------------------------------------
# Palavras-chave para detecção de prescrição
# ---------------------------------------------------------------------------

PRESCRIPTION_KEYWORDS: list[str] = [
    "prescrição",
    "prescrever",
    "prescreva",
    "prescrito",
    "receita médica",
    "receitar",
    "receite",
    "prescribe",
    "prescription",
    "receituário",
    "receituario",
]

# ---------------------------------------------------------------------------
# Palavras-chave para detecção de sugestão clínica
# ---------------------------------------------------------------------------

CLINICAL_SUGGESTION_KEYWORDS: list[str] = [
    "sugiro",
    "sugere",
    "recomendo",
    "recomenda",
    "recomendado",
    "indicado",
    "indica-se",
    "protocolo",
    "conduta",
    "tratamento",
    "medicamento",
    "dose",
    "dosagem",
    "terapia",
    "terapêutica",
    "terapeutica",
    "intervenção",
    "intervencao",
    "procedimento",
    "cirurgia",
    "exame",
    "solicitar",
    "administrar",
    "iniciar",
    "suspender",
    "ajustar",
]
