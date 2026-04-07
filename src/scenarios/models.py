from typing import Any
from pydantic import BaseModel, Field


class AssetProfile(BaseModel):
    asset_name: str
    description: str
    sensor_mappings: dict[str, str] = Field(default_factory=dict)
    known_failure_modes: list[str] = Field(default_factory=list)
    relevant_tools: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    iso_standards: list[str] = Field(default_factory=list)


class RetrievalAction(BaseModel):
    action: str = "search"
    reason: str = ""
    canonical_asset_name: str
    queries: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)


class EvidenceCandidate(BaseModel):
    arxiv_id: str
    title: str
    summary: str
    query: str
    pdf_url: str | None = None
    published: str | None = None
    judge_score: int = 0
    judge_reason: str = ""


class EvidenceSnippet(BaseModel):
    arxiv_id: str
    title: str
    url: str | None = None
    source: str
    text: str


class RetrievalDiagnostics(BaseModel):
    steps_run: int = 1
    finish_reason: str = ""
    metadata_requests: int = 0
    pdf_requests: int = 0
    cooldown_seconds: float = 3.1


class EvidenceBundle(BaseModel):
    asset_name: str
    canonical_asset_name: str
    query_history: list[str] = Field(default_factory=list)
    selected_candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    snippets: list[EvidenceSnippet] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics = Field(default_factory=RetrievalDiagnostics)


class ScenarioBudget(BaseModel):
    total_scenarios: int = 50
    allocation: dict[str, int] = Field(default_factory=dict)
    reasoning: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.allocation:
            types = ["iot", "fmsr", "tsfm", "wo", "multiagent"]
            per_type = self.total_scenarios // len(types)
            remainder = self.total_scenarios % len(types)

            self.allocation = {t: per_type for t in types}
            if remainder > 0:
                self.allocation[types[0]] += remainder


class Scenario(BaseModel):
    id: str
    type: str
    text: str
    category: str
    characteristic_form: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
