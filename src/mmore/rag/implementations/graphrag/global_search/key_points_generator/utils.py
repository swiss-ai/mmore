from pydantic import BaseModel, Field


class KeyPointInfo(BaseModel):
    description: str = Field(description="The description of the key point")
    score: float = Field(description="The score of the key point")


class KeyPointsResult(BaseModel):
    points: list[KeyPointInfo] = Field(description="the points")
