from __future__ import annotations

"""LLMProxy v2 test-harness — regression corpora seeds (Phase T / WU5).

STATIC test fixtures: real fleet structured-output schemas + a chat-loop slice,
captured as literal data so Phases 3-5 can grade the proxy's structured/repair
and chat paths against a known-good, self-consistent corpus.

These fixtures are COPIED SHAPES from the live agent code (each case cites its
source file:line). They deliberately DO NOT import agent code — importing the
agents pulls the whole stack and breaks under the isolated test runner. When a
schema was large or awkward to reproduce exactly, a faithful REPRESENTATIVE
subset is captured and marked as such.

This module is DATA ONLY — it does not drive the proxy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StructuredCase:
    """One structured-output regression fixture.

    - ``kind`` is "gbnf" (a GBNF grammar string constrains the shape) or
      "json_schema" (a JSON-Schema dict constrains the shape).
    - ``schema`` holds the GBNF string or the JSON-schema dict.
    - ``prompt`` is a representative user prompt: the grammar/schema constrains
      SHAPE, but the prompt must still describe the desired output.
    - ``valid_output`` is a JSON string satisfying the schema.
    - ``invalid_output`` is a JSON string that PARSES but VIOLATES the schema
      (or is truncated) — the Phase-3 repair target.
    """

    name: str
    source: str
    kind: str  # "gbnf" | "json_schema"
    schema: Any
    prompt: str
    valid_output: str
    invalid_output: str


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — mail-agent email-triage classifier (GBNF).
# Source: originfleet/originfleet/agents/mail-agent/grammars/classify_triage.gbnf
#   + enums in originfleet/originfleet/agents/mail-agent/triage_taxonomy.py:44-65
#   + payload build in triage_classifier.py:341-375 (mode="triage").
# The grammar emits the three decision axes + self-assessed confidence + reason.
# ─────────────────────────────────────────────────────────────────────────────

_TRIAGE_GBNF = r'''# GBNF grammar for the mail-agent email-triage classifier.
# copied shape of grammars/classify_triage.gbnf
root ::= (
        "{" ws
        "\"domain\"" ws ":" ws domain ws ","
        ws "\"action\"" ws ":" ws action ws ","
        ws "\"urgency\"" ws ":" ws urgency ws ","
        ws "\"confidence\"" ws ":" ws conf ws ","
        ws "\"reason\"" ws ":" ws qstr ws
        "}"
        )
domain ::= (
          "\"flight_itinerary\"" | "\"lodging\"" | "\"ground_travel\""
        | "\"receipt\"" | "\"bill\"" | "\"delivery\"" | "\"appointment\""
        | "\"event\"" | "\"contact\"" | "\"news_article\"" | "\"personal_note\""
        | "\"family_note\"" | "\"security\"" | "\"financial_notice\""
        | "\"insurance_eob\"" | "\"mail_digest\"" | "\"utility_notice\""
        | "\"generic\""
        )
action ::= "\"pay\"" | "\"reply\"" | "\"decide\"" | "\"schedule\"" | "\"verify\"" | "\"none\""
urgency ::= "\"overdue\"" | "\"due\"" | "\"upcoming\"" | "\"info\""
conf ::= "\"high\"" | "\"medium\"" | "\"low\""
qstr  ::= "\"" qchar* "\""
qchar ::= [^"\\\x00-\x1f] | "\\" esc
esc   ::= ["\\/bfnrt] | "u" hex hex hex hex
hex   ::= [0-9a-fA-F]
ws    ::= [ \t\n\r]*
'''

TRIAGE_DOMAINS = [
    "flight_itinerary", "lodging", "ground_travel", "receipt", "bill",
    "delivery", "appointment", "event", "contact", "news_article",
    "personal_note", "family_note", "security", "financial_notice",
    "insurance_eob", "mail_digest", "utility_notice", "generic",
]
TRIAGE_ACTIONS = ["pay", "reply", "decide", "schedule", "verify", "none"]
TRIAGE_URGENCIES = ["overdue", "due", "upcoming", "info"]
TRIAGE_CONFIDENCES = ["high", "medium", "low"]

_TRIAGE_CASE = StructuredCase(
    name="mail-agent_email_triage_gbnf",
    source="originfleet/originfleet/agents/mail-agent/grammars/classify_triage.gbnf"
           " + triage_taxonomy.py:44-65",
    kind="gbnf",
    schema=_TRIAGE_GBNF,
    prompt=(
        "Classify this email on three axes and self-assess.\n"
        "domain = what it is about (one of: flight_itinerary, lodging, "
        "ground_travel, receipt, bill, delivery, appointment, event, contact, "
        "news_article, personal_note, family_note, security, financial_notice, "
        "insurance_eob, mail_digest, utility_notice, generic).\n"
        "action = what the user must do (pay|reply|decide|schedule|verify|none).\n"
        "urgency = when it matters (overdue|due|upcoming|info).\n"
        "confidence = high|medium|low. reason = one line.\n\n"
        "EMAIL: From: billing@example-utility.test  Subject: Your Example Utility bill is "
        "ready — $142.18 due Jul 15. Autopay is OFF on this account.\n\n"
        "Return ONLY the JSON object."
    ),
    valid_output=(
        '{"domain": "bill", "action": "pay", "urgency": "due", '
        '"confidence": "high", "reason": "Utility invoice with an amount due Jul 15, autopay off."}'
    ),
    # VIOLATION: `domain` "utility_bill" is not in the enum, `urgency` "soon" is
    # not in the enum. Parses as JSON, fails the grammar's closed vocab.
    invalid_output=(
        '{"domain": "utility_bill", "action": "pay", "urgency": "soon", '
        '"confidence": "high", "reason": "Utility bill due."}'
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — mail-agent triage in "understand" (escalation) mode (GBNF+extra).
# Source: triage_classifier.py:216-240 (TriageVerdict), :363-368 (understand mode
#   grammar = classify_triage_understand.gbnf), parse_verdict :277-324.
# The understand grammar additionally emits summary / recommended_action /
# priority_reason. Represented here as a superset schema case (representative).
# ─────────────────────────────────────────────────────────────────────────────

_TRIAGE_UNDERSTAND_GBNF = r'''# representative of grammars/classify_triage_understand.gbnf
# superset of the cheap triage grammar: same 5 axes PLUS three understanding fields.
root ::= (
        "{" ws
        "\"domain\"" ws ":" ws domain ws ","
        ws "\"action\"" ws ":" ws action ws ","
        ws "\"urgency\"" ws ":" ws urgency ws ","
        ws "\"confidence\"" ws ":" ws conf ws ","
        ws "\"reason\"" ws ":" ws qstr ws ","
        ws "\"summary\"" ws ":" ws qstr ws ","
        ws "\"recommended_action\"" ws ":" ws qstr ws ","
        ws "\"priority_reason\"" ws ":" ws qstr ws
        "}"
        )
domain ::= "\"bill\"" | "\"security\"" | "\"appointment\"" | "\"personal_note\"" | "\"generic\""
action ::= "\"pay\"" | "\"reply\"" | "\"decide\"" | "\"schedule\"" | "\"verify\"" | "\"none\""
urgency ::= "\"overdue\"" | "\"due\"" | "\"upcoming\"" | "\"info\""
conf ::= "\"high\"" | "\"medium\"" | "\"low\""
qstr  ::= "\"" qchar* "\""
qchar ::= [^"\\\x00-\x1f] | "\\" esc
esc   ::= ["\\/bfnrt] | "u" hex hex hex hex
hex   ::= [0-9a-fA-F]
ws    ::= [ \t\n\r]*
'''

_TRIAGE_UNDERSTAND_CASE = StructuredCase(
    name="mail-agent_email_triage_understand_gbnf",
    source="originfleet/originfleet/agents/mail-agent/triage_classifier.py:216-240,363-368",
    kind="gbnf",
    schema=_TRIAGE_UNDERSTAND_GBNF,
    prompt=(
        "This email escalated for a deeper read. Emit the same domain/action/"
        "urgency/confidence/reason axes AND a short summary, a recommended_action, "
        "and a priority_reason.\n\n"
        "EMAIL: From: security@example-bank.test  Subject: New sign-in to your account from "
        "a device we don't recognize. Review activity now.\n\n"
        "Return ONLY the JSON object."
    ),
    valid_output=(
        '{"domain": "security", "action": "verify", "urgency": "upcoming", '
        '"confidence": "high", "reason": "Unrecognized sign-in alert on a bank account.", '
        '"summary": "The bank flagged a sign-in from an unrecognized device.", '
        '"recommended_action": "Review recent account activity and confirm it was you.", '
        '"priority_reason": "Account-security event on a financial account."}'
    ),
    # VIOLATION: truncated mid-string (unterminated JSON) — a classic streaming
    # cutoff. Fails to parse; Phase-3 repair target.
    invalid_output=(
        '{"domain": "security", "action": "verify", "urgency": "upcoming", '
        '"confidence": "high", "reason": "Unrecognized sign-in alert", '
        '"summary": "The bank flagged a sign-in from an unrecogniz'
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — knowledge_v3 refinery "frame extraction" (json_schema).
# Source: originfleet/originfleet/agents/knowledge_v3/refinery/frame_extract.py
#   _SYSTEM :48-71 (the output contract) + _normalize :147-164 (the persisted shape).
# The in-repo gold standard: NO fixed schema is handed to the model — it derives
# the KIND and coins new component types (open vocab). We represent the OUTPUT
# CONTRACT as a permissive json_schema (open component-kinds via additionalProperties)
# so Phase-3 can validate the persisted-frame shape without closing the vocab.
# ─────────────────────────────────────────────────────────────────────────────

_FRAME_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["kind", "is_new_kind", "spine", "status", "components"],
    "properties": {
        "kind": {"type": "string", "minLength": 1},
        "is_new_kind": {"type": "boolean"},
        "why_this_kind": {"type": "string"},
        # SPINE is open: identity facts vary per kind (who/when/where for a
        # journey; what/for-whom/how-much for a charge). Keys are dynamic.
        "spine": {
            "type": "object",
            "properties": {
                "answers": {"type": "object"},
                "absent": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "status": {
            "type": "string",
            # STATUS is a soft modality enum (kinds that have one).
            "enum": ["proposed", "booked", "confirmed", "completed",
                     "cancelled", "unknown"],
        },
        # COMPONENTS: open vocab — the model coins new `type`s (is_new_kind /
        # novel). We require grounding (evidence) + a type, but never close the type.
        "components": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"type": "string", "minLength": 1},
                    "label": {"type": "string"},
                    "details": {},
                    "novel": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "additionalProperties": True,
            },
        },
        "noise_excluded": {"type": "array", "items": {"type": "string"}},
        "coherence_check": {"type": "object"},
    },
    "additionalProperties": True,
}

_FRAME_CASE = StructuredCase(
    name="knowledge_v3_frame_extract_json_schema",
    source="originfleet/originfleet/agents/knowledge_v3/refinery/frame_extract.py:48-71,147-164",
    kind="json_schema",
    schema=_FRAME_JSON_SCHEMA,
    prompt=(
        "You build knowledge from a source. NO schema is given — you derive it. "
        "Decide the KIND of thing this is (reuse a known kind if it fits, else "
        "coin a new snake_case name and set is_new_kind=true). Fill the SPINE "
        "(the few stable identity facts a human needs to identify it; mark ABSENT "
        "what the source doesn't support — never invent). Discover the COMPONENTS "
        "this instance actually has, each {type, label, details, novel, evidence}, "
        "grounded in the source. Set STATUS (proposed|booked|confirmed|completed|"
        "cancelled|unknown). Exclude marketing/boilerplate as noise. Add a "
        "coherence_check.\n\n"
        "SOURCE: Northwind Air confirmation QQ7X2R — Jordan Rivera, SFO→JFK Jul 12 6:05am flight "
        "NW2412, seat 14C, Economy. Earn 2,500 miles! Terms apply.\n\n"
        "Output ONLY the JSON object."
    ),
    valid_output=(
        '{"kind": "flight_journey", "is_new_kind": false, '
        '"why_this_kind": "A booked airline itinerary for one passenger.", '
        '"spine": {"answers": {"who": {"value": "Jordan Rivera", "status": "stated"}, '
        '"when": {"value": "Jul 12 6:05am", "status": "stated"}, '
        '"from": {"value": "SFO", "status": "stated"}, '
        '"to": {"value": "JFK", "status": "stated"}}, "absent": ["return_leg"]}, '
        '"status": "confirmed", '
        '"components": [{"type": "flight_segment", "label": "NW2412 SFO-JFK", '
        '"details": {"flight": "NW2412", "seat": "14C", "cabin": "Economy"}, '
        '"novel": false, "evidence": "SFO\\u2192JFK Jul 12 6:05am flight NW2412, seat 14C"}], '
        '"noise_excluded": ["Earn 2,500 miles! Terms apply."], '
        '"coherence_check": {"makes_sense": true, "gaps": [], "contradictions": []}}'
    ),
    # VIOLATION: `status` "reserved" is not in the modality enum, and a component
    # is missing its required `type`. Parses as JSON, fails the schema.
    invalid_output=(
        '{"kind": "flight_journey", "is_new_kind": false, '
        '"spine": {"answers": {}, "absent": []}, '
        '"status": "reserved", '
        '"components": [{"label": "NW2412 SFO-JFK", "evidence": "flight NW2412"}]}'
    ),
)


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — investor trade-proposal array (json_schema).
# Source: originfleet/originfleet/agents/investor/perception/proposals.py
#   Proposal :66-86, _SYSTEM_PROMPT :43-55, _parse_one :129-164 (validation rules).
# The model outputs a JSON ARRAY of trade IDEAS (never orders/sizes). Each item:
#   symbol (ticker), direction long|short|flat, thesis, confidence 0..1, source_refs.
# ─────────────────────────────────────────────────────────────────────────────

_INVESTOR_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "required": ["symbol", "direction", "thesis"],
        "properties": {
            "symbol": {"type": "string", "minLength": 1},
            "direction": {"type": "string", "enum": ["long", "short", "flat"]},
            "thesis": {"type": "string", "minLength": 1},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "source_refs": {"type": "array", "items": {"type": "string"}},
        },
        # explicitly reject size/order fields the risk system owns.
        "not": {
            "anyOf": [
                {"required": ["size"]},
                {"required": ["quantity"]},
                {"required": ["notional"]},
                {"required": ["weight"]},
                {"required": ["price"]},
            ]
        },
        "additionalProperties": True,
    },
}

_INVESTOR_CASE = StructuredCase(
    name="investor_trade_proposals_json_schema",
    source="originfleet/originfleet/agents/investor/perception/proposals.py:43-86,129-164",
    kind="json_schema",
    schema=_INVESTOR_JSON_SCHEMA,
    prompt=(
        "You are a markets analyst that reads recent news and proposes EQUITY-ETF "
        "trade IDEAS — never orders, never position sizes. Output ONLY a JSON array. "
        "Each element has EXACTLY: symbol (a ticker from the allowed universe), "
        "direction (long|short|flat), thesis (1-2 sentences grounding it in the "
        "news), confidence (0..1, calibrated), source_refs (short citation strings). "
        "Do NOT include size, quantity, notional, weight, price, or order fields.\n\n"
        "Allowed universe (use ONLY these tickers): SPY, QQQ, XLE, XLF, GLD.\n\n"
        "Recent news/context:\n"
        "- Fed holds rates; dot-plot signals two cuts later this year.\n"
        "- Crude slips 3% on demand-outlook downgrade.\n\n"
        "Propose the JSON array of trade IDEAS now (or [] if nothing is actionable)."
    ),
    valid_output=(
        '[{"symbol": "XLE", "direction": "short", '
        '"thesis": "Crude down 3% on a demand downgrade pressures energy names near-term.", '
        '"confidence": 0.55, "source_refs": ["Crude slips 3% on demand-outlook downgrade"]}, '
        '{"symbol": "QQQ", "direction": "long", '
        '"thesis": "A dovish dot-plot signaling two cuts is a tailwind for long-duration tech.", '
        '"confidence": 0.6, "source_refs": ["Fed holds rates; dot-plot signals two cuts"]}]'
    ),
    # VIOLATION: `direction` "buy" is not in the enum, `confidence` 1.5 is out of
    # [0,1], and a forbidden `size` field is present. Parses as JSON, fails the schema.
    invalid_output=(
        '[{"symbol": "XLE", "direction": "buy", '
        '"thesis": "Energy looks weak.", "confidence": 1.5, "size": 100, '
        '"source_refs": ["crude down"]}]'
    ),
)


STRUCTURED_CASES: List[StructuredCase] = [
    _TRIAGE_CASE,
    _TRIAGE_UNDERSTAND_CASE,
    _FRAME_CASE,
    _INVESTOR_CASE,
]


# ─────────────────────────────────────────────────────────────────────────────
# CHAT_LOOP_CASES — representative orchestrator inner-loop chat request bodies.
# Representative of the orchestrator tool-calling inner loop; the leading
# byte-stable block (persona + loop contract + tool catalog) then a user turn,
# then assistant tool-calls + tool results across iterations.
# See originfleet/originfleet/orchestrator/prompt_builder.py:14-34 for the
# leading/trailing split; these are SYNTHESIZED to look like real inner-loop turns.
# ─────────────────────────────────────────────────────────────────────────────

_LOOP_SYSTEM_LEADING = (
    "You are Sidekick, the household's assistant. You run a tool-calling loop: on each "
    "turn either call exactly one tool to gather what you need, or answer the user "
    "directly when you have enough. Tools available:\n"
    "- knowledge_recall: retrieve stored facts about people, trips, and things.\n"
    "- temporal_context: current time, place, weather, and daily awareness.\n"
    "- research: local-first factual lookup + web/news search.\n"
    "Prefer one decisive tool call over guessing. Never fabricate a tool result."
)

# Case A — single-turn, tool not yet called (iteration 1).
_CHAT_CASE_A: Dict[str, Any] = {
    "name": "loop_iter1_user_only",
    "model": "orchestrator-reasoner",
    "messages": [
        {"role": "system", "content": _LOOP_SYSTEM_LEADING},
        {"role": "user", "content": "When is Robin's next dentist appointment?"},
    ],
    "max_tokens": 512,
    "temperature": 0.2,
}

# Case B — iteration 2: assistant issued a tool call, tool result came back.
_CHAT_CASE_B: Dict[str, Any] = {
    "name": "loop_iter2_after_tool_result",
    "model": "orchestrator-reasoner",
    "messages": [
        {"role": "system", "content": _LOOP_SYSTEM_LEADING},
        {"role": "user", "content": "When is Robin's next dentist appointment?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_kr_01",
                    "type": "function",
                    "function": {
                        "name": "knowledge_recall",
                        "arguments": '{"query": "Robin dentist appointment upcoming"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_kr_01",
            "content": (
                '{"hits": [{"fact": "Robin has a dental cleaning at Example Dental on '
                'Jul 9 at 3:30pm", "source": "email:example-dental:2026-06-20"}]}'
            ),
        },
    ],
    "max_tokens": 512,
    "temperature": 0.2,
}

# Case C — a two-tool chain (temporal then research), iteration 3.
_CHAT_CASE_C: Dict[str, Any] = {
    "name": "loop_iter3_two_tool_chain",
    "model": "orchestrator-reasoner",
    "messages": [
        {"role": "system", "content": _LOOP_SYSTEM_LEADING},
        {"role": "user", "content": "Do I need a jacket for my walk this afternoon?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_tc_01",
                    "type": "function",
                    "function": {
                        "name": "temporal_context",
                        "arguments": '{"want": ["weather", "place"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_tc_01",
            "content": (
                '{"place": "Riverton", "weather": {"temp_f": 52, "conditions": '
                '"breezy, clouds building", "precip_chance": 0.4}}'
            ),
        },
        {
            "role": "assistant",
            "content": (
                "It's 52F and breezy in Riverton with clouds building and a 40% chance "
                "of rain this afternoon — I'd take a light jacket."
            ),
        },
    ],
    "max_tokens": 512,
    "temperature": 0.3,
}

# Case D — grammar-constrained inner sub-call (routing decision), extra_body carries a grammar.
_CHAT_CASE_D: Dict[str, Any] = {
    "name": "loop_routing_grammar_subcall",
    "model": "gemma-router",
    "messages": [
        {
            "role": "system",
            "content": (
                "Route the user's message to exactly one handler. Emit "
                'ONLY {"handler": "...", "confidence": "high|medium|low"}.'
            ),
        },
        {"role": "user", "content": "play something mellow for the evening"},
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "extra_body": {
        "grammar": (
            'root ::= "{" ws "\\"handler\\"" ws ":" ws handler ws "," ws '
            '"\\"confidence\\"" ws ":" ws conf ws "}"\n'
            'handler ::= "\\"dj\\"" | "\\"knowledge\\"" | "\\"temporal\\"" | "\\"research\\"" | "\\"chat\\""\n'
            'conf ::= "\\"high\\"" | "\\"medium\\"" | "\\"low\\""\n'
            'ws ::= [ \\t\\n\\r]*'
        )
    },
}

# Case E — a longer summary/carryover turn (trailing dynamic block heavy).
_CHAT_CASE_E: Dict[str, Any] = {
    "name": "loop_summary_carryover_turn",
    "model": "orchestrator-reasoner",
    "messages": [
        {"role": "system", "content": _LOOP_SYSTEM_LEADING},
        {
            "role": "system",
            "content": (
                "[rolling summary] Jordan asked about weekend travel; you recalled the "
                "Jul 12 Delta SFO->JFK flight and the Example Dental appointment for Robin.\n"
                "[self_status] knowledge_recall: ok  temporal_context: ok\n"
                "[carryover] user is planning around Robin's schedule this week."
            ),
        },
        {"role": "user", "content": "Given all that, what should I prioritize tomorrow?"},
    ],
    "max_tokens": 768,
    "temperature": 0.4,
}


CHAT_LOOP_CASES: List[Dict[str, Any]] = [
    _CHAT_CASE_A,
    _CHAT_CASE_B,
    _CHAT_CASE_C,
    _CHAT_CASE_D,
    _CHAT_CASE_E,
]
