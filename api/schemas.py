
from typing import Any
from pydantic import BaseModel, Field


class FlowRequest(BaseModel):
    features: dict[str, Any]


class BatchFlowRequest(BaseModel):
    flows: list[dict[str, Any]] = Field(
        min_length=1,
        max_length=256
    )
