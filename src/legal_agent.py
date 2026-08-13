"""
legal_agent.py
Core reasoning layer. Takes structured case facts + retrieved legal context
and produces prosecution points, cross-examination points, and handles
follow-up Q&A.
"""

import json
from src.llm_client import LLMClient
from src.rag_engine import LegalRAGEngine
from src.document_processor import CaseDocument


SYSTEM_PROMPT = """You are an assistant supporting a licensed advocate in India. You are NOT
providing legal advice to a member of the public — you are a research and drafting aid for a
qualified lawyer who will independently verify everything before use in court.

Rules:
- Ground every point ONLY in the retrieved legal context provided to you plus the case facts.
  Never invent section numbers, case citations, or amendment dates.
- If the retrieved context doesn't cover something, say so explicitly rather than guessing.
- Distinguish clearly between IPC/CrPC (pre-July 2024) and BNS/BNSS/BSA (post-July 2024)
  provisions where relevant, and flag which regime applies based on the incident date.
- Cross-examination points must be procedurally and evidentially grounded (e.g. delay in FIR
  filing, chain of custody gaps, inconsistencies between FIR and charge sheet, compliance with
  mandatory sections like 41A CrPC / 35 BNSS, witness reliability) — not speculative attacks.
- Output structured, citation-backed points an advocate can act on directly.
"""


class LegalAgent:
    def __init__(self, llm_client: LLMClient = None, rag_engine: LegalRAGEngine = None):
        self.llm = llm_client or LLMClient()
        self.rag = rag_engine or LegalRAGEngine()
        self.session_context = {}

    def analyze_case(self, case_doc: CaseDocument, structured_facts: dict) -> dict:
        section_context = self.rag.retrieve_for_sections(case_doc.sections_cited)
        allegation_context = self.rag.retrieve_for_allegation(
            structured_facts.get("allegation_summary", "") or case_doc.raw_text[:2000]
        )

        retrieved_text = self._format_retrieved_context(section_context, allegation_context)

        user_prompt = f"""
CASE FACTS:
{json.dumps(structured_facts, indent=2)}

SECTIONS CITED IN DOCUMENT: {case_doc.sections_cited}
DOCUMENT TYPE: {case_doc.doc_type}

RETRIEVED LEGAL CONTEXT (statute text, amendment notes, case law):
{retrieved_text}

TASK:
Produce a JSON object with exactly these keys:

"applicable_law_summary": brief note on which regime applies (IPC/CrPC vs BNS/BNSS) and why,
"prosecution_points": [ {{ "point": str, "grounded_in": str, "strength": "strong|moderate|weak" }} ],
"cross_examination_points": [ {{ "point": str, "target": "FIR|charge_sheet|witness|procedure", "grounded_in": str, "suggested_question": str }} ],
"procedural_checks": [ {{ "check": str, "status": "compliant|non-compliant|unclear", "note": str }} ],
"gaps_in_retrieved_context": [ str ]

Return ONLY the JSON object, no preamble.
"""
        response = self.llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=3000)
        analysis = self._safe_json_parse(response)

        self.session_context = {
            "case_doc": case_doc,
            "structured_facts": structured_facts,
            "retrieved_context": retrieved_text,
            "analysis": analysis,
        }
        return analysis

    def ask_followup(self, question: str, chat_history: list[dict] = None) -> str:
        if not self.session_context:
            return "No case loaded yet. Please analyze a case document first."

        extra_context = self.rag.retrieve(question, k=4)
        extra_text = "\n\n".join(r["content"] for r in extra_context)

        history_text = ""
        if chat_history:
            for turn in chat_history[-6:]:
                history_text += f"{turn['role'].upper()}: {turn['content']}\n"

        user_prompt = f"""
CASE CONTEXT (already analyzed):
{json.dumps(self.session_context['analysis'], indent=2)}

ADDITIONAL RETRIEVED LEGAL CONTEXT FOR THIS QUESTION:
{extra_text}

CONVERSATION SO FAR:
{history_text}

ADVOCATE'S QUESTION:
{question}

Answer directly and specifically, grounded in the retrieved context and case facts above.
If drafting a question for cross-examination, phrase it as it would actually be put to a witness in court.
"""
        return self.llm.complete(SYSTEM_PROMPT, user_prompt, max_tokens=1500)

    def _format_retrieved_context(self, section_context: dict, allegation_context: list) -> str:
        parts = []
        for section, results in section_context.items():
            parts.append(f"--- Context for cited section: {section} ---")
            for r in results:
                parts.append(r["content"])
        parts.append("--- Context from fact-pattern semantic search ---")
        for r in allegation_context:
            parts.append(r["content"])
        return "\n\n".join(parts)

    def _safe_json_parse(self, text: str) -> dict:
        cleaned = text.strip().strip("```json").strip("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM output as JSON", "raw_output": text}
