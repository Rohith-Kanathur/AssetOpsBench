from typing import Any
from pydantic import BaseModel, Field


class AssetProfile(BaseModel):
    asset_name: str
    description: str
    sensor_mappings: dict[str, str] = Field(default_factory=dict)
    known_failure_modes: list[str] = Field(default_factory=list)
    relevant_tools: dict[str, list[dict]] = Field(default_factory=dict)
    iso_standards: list[str] = Field(default_factory=list)


class ScenarioBudget(BaseModel):
    total_scenarios: int = 50
    allocation: dict[str, int] = Field(default_factory=dict)
    reasoning: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.allocation:
            # Equal distribution as a default
            types = ["iot", "fmsr", "tsfm", "wo", "multiagent"]
            per_type = self.total_scenarios // len(types)
            remainder = self.total_scenarios % len(types)
            
            self.allocation = {t: per_type for t in types}
            # Add remainder to first type
            if remainder > 0:
                self.allocation[types[0]] += remainder


class Scenario(BaseModel):
    id: str
    type: str  # e.g., iot, fmsr, wo, multiagent
    text: str
    category: str
    characteristic_form: str

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
