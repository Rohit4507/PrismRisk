"""Zero-hallucination adverse-action notice generator.

Strategy: an (optional) LLM renders plain-English prose from STRUCTURED evidence
facts. Every numeric token in the output must appear verbatim in the allowed set;
if any number is ungrounded, the LLM text is rejected and a deterministic,
number-safe template is emitted instead. Hallucinated numbers can never reach
the applicant.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


@dataclass(frozen=True)
class EvidenceFacts:
    pd_pct: str            # e.g. "8.4%"
    tau_pct: str           # e.g. "5.0%"  (approval cutoff)
    current_loan: str      # e.g. "$4,500"
    target_loan: str       # e.g. "$3,200"
    new_pd_pct: str        # e.g. "3.6%"
    reason_1: str
    reason_2: str
    reason_3: str
    decision: str = "AUTO_DENY"  # AUTO_DENY | GREY_ZONE_REVIEW
    applicant_ref: str = "applicant"  # never a real name in logs/notice

    def allowed_numbers(self) -> set[str]:
        """All numeric tokens present in the structured facts (commas stripped)."""
        text = " ".join(
            [
                self.pd_pct,
                self.tau_pct,
                self.current_loan,
                self.target_loan,
                self.new_pd_pct,
                self.reason_1,
                self.reason_2,
                self.reason_3,
                self.applicant_ref,  # greeting may carry digits (e.g. case_id) — never ungrounded
            ]
        )
        return {m.group().replace(",", "") for m in NUMBER_RE.finditer(text)}


@dataclass
class NoticeResult:
    text: str
    grounding_passed: bool
    fallback_triggered: bool
    ungrounded_numbers: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[str]:
    return [m.group() for m in NUMBER_RE.finditer(text)]


def find_ungrounded_numbers(text: str, allowed: set[str]) -> list[str]:
    """Return numeric tokens in `text` that do not appear in `allowed`.

    Comparison normalizes thousands separators and trailing '%'.
    """
    norm_allowed = {a.replace(",", "").rstrip("%") for a in allowed}
    bad = []
    for tok in extract_numbers(text):
        norm = tok.replace(",", "").rstrip("%")
        if norm not in norm_allowed:
            bad.append(tok)
    return bad


def deterministic_letter(facts: EvidenceFacts) -> str:
    """Always-grounded template. Numbers come only from `facts`."""
    return (
        f"Dear {facts.applicant_ref},\n\n"
        f"Thank you for your application. Our review model estimates your probability "
        f"of default at {facts.pd_pct}, above the {facts.tau_pct} approval threshold. "
        f"The main factors were: {facts.reason_1}; {facts.reason_2}; {facts.reason_3}.\n\n"
        f"One action associated with a lower model-estimated risk is reducing the "
        f"requested amount from {facts.current_loan} to {facts.target_loan} "
        f"(estimated new PD {facts.new_pd_pct}). "
        f"This is a model decision, not a final legal determination; you may ask for "
        f"manual review or provide additional documentation.\n\n"
        f"PrismRisk Credit Review"
    )


def llm_prompt(facts: EvidenceFacts) -> str:
    return (
        "Write a 3-sentence plain-English credit decision letter using ONLY these exact "
        "numbers and no others:\n"
        f"- decision: {facts.decision}\n"
        f"- estimated default probability: {facts.pd_pct}\n"
        f"- approval threshold: {facts.tau_pct}\n"
        f"- current requested loan: {facts.current_loan}\n"
        f"- suggested amount: {facts.target_loan}\n"
        f"- estimated new default probability: {facts.new_pd_pct}\n"
        f"- reasons: {facts.reason_1}; {facts.reason_2}; {facts.reason_3}\n"
        "Do not invent any other numbers, dates, rates, or amounts."
    )


def render_notice(
    facts: EvidenceFacts,
    llm: Callable[[str], str] | None = None,
) -> NoticeResult:
    """Render the notice; fall back to the safe template on any ungrounded number."""
    allowed = facts.allowed_numbers()
    if llm is None:
        text = deterministic_letter(facts)
        return NoticeResult(text, grounding_passed=True, fallback_triggered=False)

    try:
        candidate = llm(llm_prompt(facts))
    except Exception:
        candidate = ""
    ungrounded = find_ungrounded_numbers(candidate, allowed) if candidate else ["<llm-failed>"]
    if ungrounded:
        return NoticeResult(
            deterministic_letter(facts),
            grounding_passed=False,
            fallback_triggered=True,
            ungrounded_numbers=ungrounded,
        )
    return NoticeResult(candidate, grounding_passed=True, fallback_triggered=False)
