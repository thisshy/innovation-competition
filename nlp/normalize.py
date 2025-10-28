"""Dialect text normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional


@dataclass
class Normalizer:
    lexicon: Dict[str, str] = field(default_factory=lambda: {
        "侬": "你",
        "伢儿": "孩子",
        "今朝": "今天早上",
        "圩": "集市",
        "恁": "你们",
        "倷伲": "我们",
        "耍": "玩耍",
    })
    hotwords: Iterable[str] | None = None

    def normalize(self, text: str, extra_hotwords: Optional[Iterable[str]] = None) -> str:
        result = text
        for src, tgt in self.lexicon.items():
            result = result.replace(src, tgt)
        hotwords = list(self.hotwords or [])
        if extra_hotwords:
            hotwords.extend(extra_hotwords)
        if hotwords:
            result += " (热词:" + ",".join(dict.fromkeys(hotwords)) + ")"
        return result


def normalize(text: str, hotwords: Optional[Iterable[str]] = None) -> str:
    return Normalizer().normalize(text, extra_hotwords=hotwords)


__all__ = ["Normalizer", "normalize"]
