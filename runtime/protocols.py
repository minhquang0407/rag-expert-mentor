import json
import re
from typing import Any, Dict, Optional

from core.schemas import AgentResult, CriticReport


def clean_think_tags(text: str) -> str:
    """Remove reasoning traces emitted inside <think>...</think> blocks."""
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_markdown_fences(text: str) -> str:
    """Remove surrounding Markdown code fences from model output."""
    if not text:
        return ""
    pattern = r"^\s*```(?:json)?\s*(.*?)\s*```\s*$"
    match = re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def normalize_latex_markdown(text: str) -> str:
    """Normalize common malformed LLM LaTeX into Streamlit-friendly Markdown math."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"\\\[\s*", "$$\n", cleaned)
    cleaned = re.sub(r"\s*\\\]", "\n$$", cleaned)
    cleaned = re.sub(r"\\\(\s*", "$", cleaned)
    cleaned = re.sub(r"\s*\\\)", "$", cleaned)

    # Convert raw LaTeX matrix environments that were emitted without delimiters.
    cleaned = re.sub(r"(?<!\$)(\\begin\{(?:b|p|v|B|V)?matrix\})", r"$$\n\1", cleaned)
    cleaned = re.sub(r"(\\end\{(?:b|p|v|B|V)?matrix\})(?!\$)", r"\1\n$$", cleaned)

    # Repair responses that start mid-matrix, e.g. 'd_1 & 0 ... \\ ... \end{pmatrix} $$'.
    if "\\end{pmatrix}" in cleaned and "\\begin{pmatrix}" not in cleaned:
        matrix_start = re.search(r"(?m)^[^\n]*&[^\n]*(?:\\\\|\\cdots|\\ddots)", cleaned)
        if matrix_start:
            idx = matrix_start.start()
            prefix = cleaned[:idx].rstrip()
            matrix_body = cleaned[idx:].lstrip()
            matrix_body = re.sub(r"\$\$\s*$", "", matrix_body).strip()
            cleaned = f"{prefix}\n\n$$\n\\begin{{pmatrix}}\n{matrix_body}\n\\end{{pmatrix}}\n$$"

    # Ensure display math delimiters sit on their own lines for Streamlit/KaTeX.
    cleaned = re.sub(r"(?<!\$)\$\$(?!\$)", "\n$$\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def extract_json_object(text: str) -> str:
    """Extract the first likely JSON object from an LLM response."""
    cleaned = strip_markdown_fences(clean_think_tags(text))
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start:end + 1]
    return cleaned


def repair_truncated_json(text: str) -> str:
    """Best-effort repair for simple truncated JSON object responses."""
    candidate = extract_json_object(text)
    if not candidate:
        return candidate

    if candidate.count('"') % 2 != 0:
        candidate += '"'

    open_braces = candidate.count("{")
    close_braces = candidate.count("}")
    if open_braces > close_braces:
        candidate += "}" * (open_braces - close_braces)

    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")
    if open_brackets > close_brackets:
        candidate += "]" * (open_brackets - close_brackets)

    return candidate


def _sanitize_json_string_content(text: str) -> str:
    """Escape raw control chars and invalid backslash escapes inside JSON strings."""
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}
    escaped = []
    in_string = False
    idx = 0

    while idx < len(text):
        char = text[idx]

        if char == '"':
            escaped.append(char)
            # Toggle only when quote is not escaped by an odd number of backslashes.
            backslash_count = 0
            lookbehind = idx - 1
            while lookbehind >= 0 and text[lookbehind] == "\\":
                backslash_count += 1
                lookbehind -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
            idx += 1
            continue

        if in_string and char in {"\n", "\r", "\t"}:
            escaped.append(char.encode("unicode_escape").decode("ascii"))
            idx += 1
            continue

        if in_string and char == "\\":
            next_char = text[idx + 1] if idx + 1 < len(text) else ""
            if next_char and next_char in valid_escapes:
                escaped.append(char)
            else:
                # Convert LaTeX-ish invalid JSON escapes like \c, \d, \( into literal backslashes.
                escaped.append("\\\\")
            idx += 1
            continue

        escaped.append(char)
        idx += 1

    return "".join(escaped)


def parse_json_object(text: str) -> Dict[str, Any]:
    """Parse an LLM JSON object with cleaning and best-effort repair."""
    candidate = extract_json_object(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            repaired = repair_truncated_json(_sanitize_json_string_content(candidate))
            parsed = json.loads(repaired, strict=False)

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object from model output.")
    return parsed


def agent_result_to_event(result: AgentResult) -> Dict[str, Any]:
    """Convert an AgentResult into a Streamlit-compatible runtime event."""
    return {
        "type": "agent_result",
        "agent": result.agent_name,
        "content": result.message.content,
        "success": result.success,
        "error": result.error,
        "metadata": result.message.metadata,
    }


def critic_report_to_event(report: CriticReport) -> Dict[str, Any]:
    """Convert a CriticReport into a Streamlit-compatible runtime event."""
    return {
        "type": "critic_report",
        "status": report.status,
        "reviewed_agent": report.reviewed_agent,
        "issues": report.issues,
        "send_back_to": report.send_back_to,
        "confidence": report.confidence,
    }
