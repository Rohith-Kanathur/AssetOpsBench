from __future__ import annotations

import io
import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable

from llm import LLMBackend

from .models import (
    EvidenceBundle,
    EvidenceCandidate,
    EvidenceSnippet,
    RetrievalAction,
    RetrievalDiagnostics,
)
from .prompts import RETRIEVAL_METADATA_JUDGE_PROMPT, RETRIEVAL_QUERY_PLAN_PROMPT
from .utils import parse_llm_json

_log = logging.getLogger(__name__)

_ARXIV_COOLDOWN_SECONDS = 3.1
_ARXIV_BASE_URL = "http://export.arxiv.org/api/query?"
_ARXIV_HEADERS = {"User-Agent": "AssetOpsBench/1.0 (mailto:admin@example.com)"}
_MAX_STEPS = 5
_MAX_QUERIES_PER_STEP = 2
_MAX_METADATA_RESULTS = 6
_MAX_CANDIDATE_POOL = 8
_TOP_PDF_DOWNLOADS = 3
_MAX_PDF_PAGES = 5
_MAX_SNIPPETS_PER_DOC = 3

_STOPWORDS = {
    "a",
    "an",
    "and",
    "asset",
    "class",
    "equipment",
    "for",
    "in",
    "industrial",
    "machine",
    "of",
    "on",
    "or",
    "plant",
    "system",
    "the",
    "to",
}

_DIAGNOSTIC_KEYWORDS = [
    "condition monitoring",
    "diagnostic",
    "diagnostics",
    "fault",
    "fault diagnosis",
    "failure",
    "failure analysis",
    "degradation",
    "health",
    "inspection",
    "maintenance",
    "monitoring",
    "prognostic",
    "prognostics",
    "reliability",
]

_STANDARD_KEYWORDS = [
    "iec",
    "ieee",
    "iso",
    "condition assessment",
    "maintenance strategy",
    "reliability centered maintenance",
]

_ML_NEGATIVE_KEYWORDS = [
    "attention mechanism",
    "bert",
    "deep learning",
    "federated learning",
    "foundation model",
    "language model",
    "llm",
    "mamba",
    "neural network",
    "nlp",
    "time series transformer",
    "transformer architecture",
    "vision transformer",
]

_OFF_TARGET_KEYWORDS = [
    "communications",
    "control system",
    "control systems",
    "cyber attack",
    "cyber security",
    "cyberattack",
    "cybersecurity",
    "data architecture",
    "false data injection",
    "market",
    "networking",
    "smart grid",
]

_PHYSICAL_REASON_MARKERS = [
    "physical asset focused",
    "physical equipment focused",
]

_OFF_TARGET_REASON_MARKERS = [
    "not physical asset focused",
    "communications",
    "control system",
    "cyberattack",
    "cybersecurity",
    "data architecture",
    "generic ml",
    "indirect relevance",
    "market",
    "networking",
    "not directly",
    "not entirely physical asset focused",
    "other asset family",
    "somewhat indirect",
    "somewhat relevant",
    "smart-grid",
    "system paper",
    "transformer architecture",
]


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _default_canonical_asset_name(asset_name: str) -> str:
    return _normalise_text(asset_name.replace("_", " "))


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        clean = item.strip()
        if clean and clean not in seen:
            ordered.append(clean)
            seen.add(clean)
    return ordered


def _asset_tokens(asset_name: str) -> list[str]:
    return [
        token
        for token in _tokenise(asset_name)
        if len(token) > 2 and token not in _STOPWORDS
    ]


class _ArxivExecutor:
    """Single-entry executor that enforces a cooldown before every ArXiv request."""

    def __init__(self, cooldown_seconds: float = _ARXIV_COOLDOWN_SECONDS) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._last_request_at: float | None = None
        self.metadata_requests = 0
        self.pdf_requests = 0
        self._ctx = ssl.create_default_context()
        self._ctx.check_hostname = False
        self._ctx.verify_mode = ssl.CERT_NONE

    def _wait_for_cooldown(self) -> None:
        if self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.cooldown_seconds:
            time.sleep(self.cooldown_seconds - elapsed)

    def _open(self, url: str, timeout: int) -> bytes:
        self._wait_for_cooldown()
        req = urllib.request.Request(url, headers=_ARXIV_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout, context=self._ctx) as response:
            data = response.read()
        self._last_request_at = time.monotonic()
        return data

    def fetch_metadata(
        self,
        query: str,
        max_results: int = _MAX_METADATA_RESULTS,
    ) -> list[EvidenceCandidate]:
        safe_query = urllib.parse.quote(query)
        url = f"{_ARXIV_BASE_URL}search_query={safe_query}&start=0&max_results={max_results}"
        self.metadata_requests += 1

        try:
            data = self._open(url, timeout=10)
        except urllib.error.HTTPError as exc:
            _log.warning("HTTP error fetching ArXiv metadata for %r: %s", query, exc)
            return []
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch ArXiv metadata for %r: %s", query, exc)
            return []

        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            _log.warning("Failed to parse ArXiv XML for %r: %s", query, exc)
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        candidates: list[EvidenceCandidate] = []
        for entry in root.findall("atom:entry", ns):
            raw_id = entry.findtext("atom:id", default="", namespaces=ns).strip()
            arxiv_id = raw_id.rsplit("/", 1)[-1] if raw_id else ""
            title = entry.findtext("atom:title", default="No Title", namespaces=ns)
            summary = entry.findtext("atom:summary", default="No Summary", namespaces=ns)
            published = entry.findtext("atom:published", default="", namespaces=ns).strip() or None

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                href = link.attrib.get("href")
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = href
                    if pdf_url and not pdf_url.endswith(".pdf"):
                        pdf_url += ".pdf"
                    break

            candidates.append(
                EvidenceCandidate(
                    arxiv_id=arxiv_id or title.strip(),
                    title=title.strip().replace("\n", " "),
                    summary=summary.strip().replace("\n", " "),
                    query=query,
                    pdf_url=pdf_url,
                    published=published,
                )
            )
        return candidates

    def fetch_pdf_text(self, pdf_url: str, max_pages: int = _MAX_PDF_PAGES) -> str:
        self.pdf_requests += 1
        try:
            pdf_bytes = self._open(pdf_url, timeout=15)
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages: list[str] = []
            for index, page in enumerate(reader.pages):
                if index >= max_pages:
                    break
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            return "\n".join(pages)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Failed to fetch or parse ArXiv PDF %s: %s", pdf_url, exc)
            return ""


def _default_queries(canonical_asset_name: str) -> list[str]:
    canonical = _default_canonical_asset_name(canonical_asset_name)
    return [
        f"{canonical} condition monitoring",
        f"{canonical} fault diagnosis",
    ]


def _render_queries(queries: list[str]) -> str:
    return "\n".join(f"- {query}" for query in queries) if queries else "(none yet)"


def _render_results_summary(candidates: list[EvidenceCandidate]) -> str:
    if not candidates:
        return "(no judged results yet)"
    lines = []
    for candidate in candidates[:5]:
        lines.append(
            f"- [{candidate.judge_score}/10] {candidate.arxiv_id} | {candidate.title} | {candidate.judge_reason}"
        )
    return "\n".join(lines)


def _summarise_metadata_for_judge(candidates: list[EvidenceCandidate]) -> str:
    payload = [
        {
            "arxiv_id": candidate.arxiv_id,
            "title": candidate.title,
            "summary": candidate.summary,
            "query": candidate.query,
            "published": candidate.published,
        }
        for candidate in candidates
    ]
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _coerce_action(
    parsed: object,
    asset_name: str,
    canonical_asset_name: str,
    step_number: int,
    has_candidates: bool,
) -> RetrievalAction:
    fallback_canonical = canonical_asset_name or _default_canonical_asset_name(asset_name)
    fallback_queries = _default_queries(fallback_canonical)

    if not isinstance(parsed, dict):
        if has_candidates and step_number > 1:
            return RetrievalAction(
                action="finish",
                reason="Fallback finish because the retrieval action could not be parsed.",
                canonical_asset_name=fallback_canonical,
            )
        return RetrievalAction(
            action="search",
            reason="Fallback search because the retrieval action could not be parsed.",
            canonical_asset_name=fallback_canonical,
            queries=fallback_queries,
        )

    action = str(parsed.get("action", "search")).strip().lower()
    if action not in {"search", "finish"}:
        action = "finish" if has_candidates and step_number > 1 else "search"

    action_obj = RetrievalAction(
        action=action,
        reason=str(parsed.get("reason", "")).strip() or "No reason provided.",
        canonical_asset_name=str(parsed.get("canonical_asset_name") or fallback_canonical).strip() or fallback_canonical,
        queries=[
            str(query).strip()
            for query in parsed.get("queries", [])
            if str(query).strip()
        ][:_MAX_QUERIES_PER_STEP],
        selected_ids=[
            str(arxiv_id).strip()
            for arxiv_id in parsed.get("selected_ids", [])
            if str(arxiv_id).strip()
        ],
    )

    if action_obj.action == "search" and not action_obj.queries:
        action_obj.queries = fallback_queries

    if action_obj.action == "finish":
        action_obj.queries = []

    return action_obj


def _judge_fallback(
    candidate: EvidenceCandidate,
    canonical_asset_name: str,
) -> tuple[int, str]:
    text = _normalise_text(f"{candidate.title} {candidate.summary}")
    asset_phrase = _normalise_text(canonical_asset_name)
    asset_tokens = _asset_tokens(canonical_asset_name)

    score = 2
    reasons: list[str] = []

    if asset_phrase and asset_phrase in text:
        score += 4
        reasons.append("physical asset focused; mentions canonical asset")
    else:
        hits = sum(1 for token in asset_tokens if token in text)
        if hits:
            score += min(3, hits)
            reasons.append("physical asset focused; partial asset overlap")

    if any(keyword in text for keyword in _DIAGNOSTIC_KEYWORDS):
        score += 2
        reasons.append("physical asset focused; diagnostics or reliability terms")

    if any(keyword in text for keyword in _OFF_TARGET_KEYWORDS):
        score -= 3
        reasons.append("not physical asset focused; system, cyber, networking, or smart-grid context")

    if any(keyword in text for keyword in _ML_NEGATIVE_KEYWORDS):
        score -= 2
        reasons.append("not physical asset focused; contains generic ML framing")

    score = max(1, min(10, score))
    if not reasons:
        reasons.append("not physical asset focused; fallback metadata relevance estimate")
    reason = "; ".join(reasons)
    return score, reason


def _judge_metadata_batch(
    asset_name: str,
    canonical_asset_name: str,
    candidates: list[EvidenceCandidate],
    llm: LLMBackend,
) -> list[EvidenceCandidate]:
    if not candidates:
        return []

    prompt = RETRIEVAL_METADATA_JUDGE_PROMPT.format(
        asset_name=asset_name,
        canonical_asset_name=canonical_asset_name,
        metadata_entries_json=_summarise_metadata_for_judge(candidates),
    )
    response = llm.generate(prompt)
    parsed, _ = parse_llm_json(response)

    scored_by_id: dict[str, tuple[int, str]] = {}
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            arxiv_id = str(entry.get("arxiv_id", "")).strip()
            if not arxiv_id:
                continue
            raw_score = entry.get("score_1_to_10", 0)
            try:
                score = int(raw_score)
            except (TypeError, ValueError):
                score = 0
            score = max(1, min(10, score)) if score else 0
            reason = str(entry.get("reason", "")).strip() or "No judge reason provided."
            if score:
                scored_by_id[arxiv_id] = (score, reason)

    judged: list[EvidenceCandidate] = []
    for candidate in candidates:
        score_reason = scored_by_id.get(candidate.arxiv_id)
        if score_reason is None:
            score_reason = _judge_fallback(candidate, canonical_asset_name)
        candidate.judge_score = score_reason[0]
        candidate.judge_reason = score_reason[1]
        judged.append(candidate)
    return judged


def _merge_pool(
    pool: dict[str, EvidenceCandidate],
    judged_candidates: list[EvidenceCandidate],
) -> None:
    for candidate in judged_candidates:
        existing = pool.get(candidate.arxiv_id)
        if existing is None or candidate.judge_score > existing.judge_score:
            pool[candidate.arxiv_id] = candidate


def _sorted_candidates(pool: dict[str, EvidenceCandidate]) -> list[EvidenceCandidate]:
    return sorted(
        pool.values(),
        key=lambda candidate: (candidate.judge_score, candidate.title.lower()),
        reverse=True,
    )


def _fallback_retry_queries(
    canonical_asset_name: str,
    query_history: list[str],
) -> list[str]:
    canonical = _default_canonical_asset_name(canonical_asset_name)
    candidates = [
        f"{canonical} condition monitoring",
        f"{canonical} fault diagnosis",
        f"{canonical} condition assessment",
        f"{canonical} maintenance",
        f"{canonical} degradation",
        f"{canonical} physical measurements",
    ]
    return [
        query
        for query in _unique_preserve_order(candidates)
        if query not in query_history
    ][:_MAX_QUERIES_PER_STEP]


def _candidate_reason_flags(reason: str) -> tuple[bool, bool]:
    normalized = _normalise_text(reason)
    is_off_target = any(marker in normalized for marker in _OFF_TARGET_REASON_MARKERS)
    is_physical = (
        not is_off_target
        and any(marker in normalized for marker in _PHYSICAL_REASON_MARKERS)
    )
    return is_physical, is_off_target


def _candidate_focus_label(candidate: EvidenceCandidate) -> str:
    is_physical, is_off_target = _candidate_reason_flags(candidate.judge_reason)
    if candidate.judge_score >= 8 and is_physical:
        return "accepted: physical asset focused"
    if is_off_target or candidate.judge_score <= 4:
        return "rejected: not physical asset focused"
    return "mixed: uncertain physical asset focus"


def _evaluate_candidate_pool(candidates: list[EvidenceCandidate]) -> tuple[bool, str]:
    if not candidates:
        return False, "No judged candidates are available yet."

    top_candidates = candidates[:5]
    strong_physical = 0
    off_target = 0

    for candidate in top_candidates:
        is_physical, is_off_target = _candidate_reason_flags(candidate.judge_reason)
        if candidate.judge_score >= 8 and is_physical:
            strong_physical += 1
        elif is_off_target or candidate.judge_score <= 4:
            off_target += 1

    if strong_physical < 2:
        return (
            False,
            f"Only {strong_physical} top candidates are clearly physical asset focused; need at least 2 before finishing early.",
        )

    if off_target >= strong_physical:
        return (
            False,
            "Top candidates are still dominated by system, cyber, networking, or ML-context papers.",
        )

    return (
        True,
        f"Top pool contains {strong_physical} clearly physical-asset-focused papers.",
    )


def _extract_snippet_text(text: str, canonical_asset_name: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""

    keywords = _unique_preserve_order(
        [canonical_asset_name, *_DIAGNOSTIC_KEYWORDS, *_STANDARD_KEYWORDS]
    )
    lowered = compact.lower()
    windows: list[tuple[int, int]] = []
    for keyword in keywords:
        norm = _normalise_text(keyword)
        if not norm:
            continue
        idx = lowered.find(norm)
        if idx == -1:
            continue
        start = max(0, idx - 220)
        end = min(len(compact), idx + len(norm) + 220)
        windows.append((start, end))
        if len(windows) >= _MAX_SNIPPETS_PER_DOC:
            break

    if not windows:
        return compact[:900]

    snippets = [compact[start:end].strip() for start, end in windows if compact[start:end].strip()]
    return "\n...\n".join(snippets[:_MAX_SNIPPETS_PER_DOC])


def _select_final_candidates(
    pool: dict[str, EvidenceCandidate],
    preferred_ids: list[str],
) -> tuple[list[EvidenceCandidate], list[str]]:
    ordered_pool = _sorted_candidates(pool)
    ordered_by_id = {candidate.arxiv_id: candidate for candidate in ordered_pool}

    selected: list[EvidenceCandidate] = []
    selected_ids: list[str] = []
    for arxiv_id in preferred_ids:
        candidate = ordered_by_id.get(arxiv_id)
        if candidate and arxiv_id not in selected_ids:
            selected.append(candidate)
            selected_ids.append(arxiv_id)

    for candidate in ordered_pool:
        if candidate.arxiv_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.append(candidate.arxiv_id)
        if len(selected) >= _TOP_PDF_DOWNLOADS:
            break

    return selected[:_TOP_PDF_DOWNLOADS], selected_ids[:_TOP_PDF_DOWNLOADS]


def _render_step_log(
    step_number: int,
    action: RetrievalAction,
    new_queries: list[str],
    fetched_candidates: list[EvidenceCandidate],
    pool: dict[str, EvidenceCandidate],
    canonical_asset_name: str,
) -> str:
    lines = [
        f"Step: {step_number}/{_MAX_STEPS}",
        f"Action: {action.action}",
        f"Canonical Asset: {canonical_asset_name}",
        f"Reason: {action.reason}",
        "",
        "Queries:",
    ]
    if new_queries:
        lines.extend(f"- {query}" for query in new_queries)
    else:
        lines.append("(no new queries)")

    lines.extend(["", "Fetched Metadata:"])
    if fetched_candidates:
        for candidate in fetched_candidates:
            lines.append(
                f"- [{candidate.judge_score}/10] {candidate.title}"
            )
            lines.append(f"  id: {candidate.arxiv_id}")
            lines.append(f"  query: {candidate.query}")
            lines.append(f"  assessment: {_candidate_focus_label(candidate)}")
            lines.append(f"  reason: {candidate.judge_reason}")
    else:
        lines.append("(no metadata fetched)")

    lines.extend(["", "Current Top Candidates:"])
    top_candidates = _sorted_candidates(pool)[:5]
    if top_candidates:
        for candidate in top_candidates:
            lines.append(
                f"- [{candidate.judge_score}/10] {candidate.arxiv_id} | {candidate.title}"
            )
            lines.append(f"  assessment: {_candidate_focus_label(candidate)}")
            lines.append(f"  reason: {candidate.judge_reason}")
    else:
        lines.append("(no judged candidates in pool)")

    focused, focus_reason = _evaluate_candidate_pool(top_candidates)
    lines.extend(
        [
            "",
            f"Physical Asset Focused: {'yes' if focused else 'no'}",
            f"Focus Summary: {focus_reason}",
        ]
    )

    return "\n".join(lines)


def _render_final_log(
    bundle: EvidenceBundle,
    selected_candidates: list[EvidenceCandidate],
) -> str:
    focused, focus_reason = _evaluate_candidate_pool(bundle.candidates)
    lines = [
        f"Asset: {bundle.asset_name}",
        f"Canonical Asset: {bundle.canonical_asset_name}",
        f"Steps Run: {bundle.diagnostics.steps_run}",
        f"Finish Reason: {bundle.diagnostics.finish_reason}",
        f"Metadata Requests: {bundle.diagnostics.metadata_requests}",
        f"PDF Requests: {bundle.diagnostics.pdf_requests}",
        f"Cooldown: {bundle.diagnostics.cooldown_seconds:.1f}s",
        f"Physical Asset Focused: {'yes' if focused else 'no'}",
        "",
        "Query History:",
    ]
    if bundle.query_history:
        lines.extend(f"- {query}" for query in bundle.query_history)
    else:
        lines.append("(none)")

    lines.extend(["", "Top Judged Candidates:"])
    if bundle.candidates:
        for candidate in bundle.candidates:
            lines.append(
                f"- [{candidate.judge_score}/10] {candidate.arxiv_id} | {candidate.title}"
            )
            lines.append(f"  query: {candidate.query}")
            lines.append(f"  reason: {candidate.judge_reason}")
    else:
        lines.append("(none)")

    lines.extend(["", "Selected PDFs:"])
    if selected_candidates:
        for candidate in selected_candidates:
            lines.append(
                f"- {candidate.arxiv_id} | {candidate.title}"
            )
            lines.append(f"  pdf: {candidate.pdf_url or '(no pdf url)'}")
    else:
        lines.append("(none)")

    lines.extend(["", f"Final Pool Assessment: {focus_reason}"])

    return "\n".join(lines)


def retrieve_asset_evidence(
    asset_name: str,
    server_desc: dict[str, str],
    llm: LLMBackend,
    log_writer: Callable[[str, str], None] | None = None,
) -> EvidenceBundle:
    executor = _ArxivExecutor(cooldown_seconds=_ARXIV_COOLDOWN_SECONDS)
    canonical_asset_name = _default_canonical_asset_name(asset_name)
    query_history: list[str] = []
    candidate_pool: dict[str, EvidenceCandidate] = {}
    finish_reason = "Finished after reaching the step limit."
    selected_ids: list[str] = []
    steps_run = 0

    for step_number in range(1, _MAX_STEPS + 1):
        steps_run = step_number
        current_summary = _render_results_summary(_sorted_candidates(candidate_pool))
        previous_queries = _render_queries(query_history)
        prompt = RETRIEVAL_QUERY_PLAN_PROMPT.format(
            asset_name=asset_name,
            step_number=step_number,
            max_steps=_MAX_STEPS,
            canonical_asset_name=canonical_asset_name,
            tool_descriptions=json.dumps(server_desc, indent=2),
            previous_queries=previous_queries,
            current_results_summary=current_summary,
        )
        response = llm.generate(prompt)
        parsed, _ = parse_llm_json(response)
        action = _coerce_action(
            parsed=parsed,
            asset_name=asset_name,
            canonical_asset_name=canonical_asset_name,
            step_number=step_number,
            has_candidates=bool(candidate_pool),
        )
        canonical_asset_name = action.canonical_asset_name or canonical_asset_name

        if action.action == "finish" and candidate_pool:
            pool_focused, focus_reason = _evaluate_candidate_pool(
                _sorted_candidates(candidate_pool)
            )
            if pool_focused or step_number == _MAX_STEPS:
                finish_reason = (
                    action.reason
                    if pool_focused
                    else f"{action.reason} Finalizing at step limit. {focus_reason}"
                )
                selected_ids = action.selected_ids
                if log_writer:
                    log_writer(
                        f"retrieval_step_{step_number:02d}",
                        _render_step_log(
                            step_number=step_number,
                            action=action,
                            new_queries=[],
                            fetched_candidates=[],
                            pool=candidate_pool,
                            canonical_asset_name=canonical_asset_name,
                        ),
                    )
                break

            action = RetrievalAction(
                action="search",
                reason=f"Early finish blocked. {focus_reason}",
                canonical_asset_name=canonical_asset_name,
                queries=_fallback_retry_queries(canonical_asset_name, query_history),
            )

        new_queries = [
            query
            for query in _unique_preserve_order(action.queries)
            if query not in query_history
        ][: _MAX_QUERIES_PER_STEP]

        if action.action == "search" and not new_queries:
            new_queries = _fallback_retry_queries(canonical_asset_name, query_history)

        if not new_queries:
            if candidate_pool:
                finish_reason = action.reason or "No new queries remained; keeping the best-so-far pool."
                if log_writer:
                    log_writer(
                        f"retrieval_step_{step_number:02d}",
                        _render_step_log(
                            step_number=step_number,
                            action=RetrievalAction(
                                action="finish",
                                reason=finish_reason,
                                canonical_asset_name=canonical_asset_name,
                            ),
                            new_queries=[],
                            fetched_candidates=[],
                            pool=candidate_pool,
                            canonical_asset_name=canonical_asset_name,
                        ),
                    )
                break
            new_queries = _default_queries(canonical_asset_name)

        query_history.extend(new_queries)
        fetched_candidates: list[EvidenceCandidate] = []
        for query in new_queries:
            fetched_candidates.extend(executor.fetch_metadata(query))

        judged_candidates = _judge_metadata_batch(
            asset_name=asset_name,
            canonical_asset_name=canonical_asset_name,
            candidates=fetched_candidates,
            llm=llm,
        )
        _merge_pool(candidate_pool, judged_candidates)

        if log_writer:
            log_writer(
                f"retrieval_step_{step_number:02d}",
                _render_step_log(
                    step_number=step_number,
                    action=action,
                    new_queries=new_queries,
                    fetched_candidates=judged_candidates,
                    pool=candidate_pool,
                    canonical_asset_name=canonical_asset_name,
                ),
            )

    top_candidates = _sorted_candidates(candidate_pool)[:_MAX_CANDIDATE_POOL]
    selected_candidates, selected_candidate_ids = _select_final_candidates(
        candidate_pool,
        selected_ids,
    )

    snippets: list[EvidenceSnippet] = []
    for candidate in selected_candidates:
        source = "summary"
        text = candidate.summary
        if candidate.pdf_url:
            pdf_text = executor.fetch_pdf_text(candidate.pdf_url)
            if pdf_text.strip():
                source = "pdf"
                text = pdf_text
        snippet_text = _extract_snippet_text(text, canonical_asset_name)
        if not snippet_text:
            continue
        snippets.append(
            EvidenceSnippet(
                arxiv_id=candidate.arxiv_id,
                title=candidate.title,
                url=candidate.pdf_url,
                source=source,
                text=snippet_text,
            )
        )

    bundle = EvidenceBundle(
        asset_name=asset_name,
        canonical_asset_name=canonical_asset_name,
        query_history=query_history,
        selected_candidate_ids=selected_candidate_ids,
        candidates=top_candidates,
        snippets=snippets,
        diagnostics=RetrievalDiagnostics(
            steps_run=steps_run,
            finish_reason=finish_reason,
            metadata_requests=executor.metadata_requests,
            pdf_requests=executor.pdf_requests,
            cooldown_seconds=_ARXIV_COOLDOWN_SECONDS,
        ),
    )

    if log_writer:
        log_writer("retrieval_summary", _render_final_log(bundle, selected_candidates))

    return bundle
