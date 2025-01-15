from dataclasses import dataclass


@dataclass
class CommunityReport:
    id: str
    title: str
    summary: str
    rank: float
    weight: float
    content: str
