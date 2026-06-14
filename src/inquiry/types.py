"""心智漫游问题卡类型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptCard:
    """一张可用于引导用户思考的问题卡。"""

    id: str
    kind: str
    title: str
    prompt: str
    context: str = ""
    tags: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    twists: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "PromptCard":
        return cls(
            id=str(data["id"]).strip(),
            kind=str(data["kind"]).strip(),
            title=str(data["title"]).strip(),
            prompt=str(data["prompt"]).strip(),
            context=str(data.get("context") or "").strip(),
            tags=[str(item).strip() for item in data.get("tags", []) if str(item).strip()],
            followups=[
                str(item).strip() for item in data.get("followups", []) if str(item).strip()
            ],
            twists=[str(item).strip() for item in data.get("twists", []) if str(item).strip()],
        )
