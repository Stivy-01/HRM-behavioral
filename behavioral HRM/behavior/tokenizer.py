from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

_CANON_MAP = {
    "Side by side Contact, opposite way": "SIDE_BY_SIDE_CONTACT_OPPOSITE",
    "Side by side Contact": "SIDE_BY_SIDE_CONTACT",
    "Oral-genital Contact": "ORAL_GENITAL_CONTACT",
    "Oral-oral Contact": "ORAL_ORAL_CONTACT",
    "Approach rear": "APPROACH_REAR",
    "Get away": "GET_AWAY",
    "Social approach": "SOCIAL_APPROACH",
    "Social escape": "SOCIAL_ESCAPE",
    "Break contact": "BREAK_CONTACT",
    "Contact": "CONTACT",
    "Move in contact": "MOVE_IN_CONTACT",
    "Stop in contact": "STOP_IN_CONTACT",
    "Move isolated": "MOVE_ISOLATED",
    "Stop isolated": "STOP_ISOLATED",
    "Rear in contact": "REAR_IN_CONTACT",
    "Rearing": "REARING",
    "SAP": "SAP",
    "Huddling": "HUDDLING",
    "Center_Zone": "CENTER_ZONE",
    "Periphery_Zone": "PERIPHERY_ZONE",
}


def canonicalize_event(name: str) -> str:
    if name in _CANON_MAP:
        return _CANON_MAP[name]
    norm = []
    for ch in name:
        if ch.isalnum():
            norm.append(ch.upper())
        else:
            norm.append("_")
    tok = "".join(norm)
    while "__" in tok:
        tok = tok.replace("__", "_")
    return tok.strip("_")


PAD_TOKEN = "PAD"
EOS_TOKEN = "EOS"


@dataclass
class Vocab:
    token_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_token: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.id_to_token and not self.token_to_id:
            self.add_token(PAD_TOKEN)
            self.add_token(EOS_TOKEN)

    def add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.id_to_token)
        self.id_to_token.append(token)
        self.token_to_id[token] = idx
        return idx

    def get_id(self, token: str) -> int:
        return self.token_to_id.get(token, self.add_token(token))


@dataclass
class DurationBinner:
    edges: List[float] = field(default_factory=lambda: [0.37, 0.70, 1.57, 3.80, 9.17, 16.57, 64.40])

    def bin_index(self, seconds: float) -> int:
        for i, e in enumerate(self.edges, start=1):
            if seconds <= e:
                return i
        return len(self.edges) + 1

    def token(self, seconds: float) -> str:
        return f"DUR_BIN_{self.bin_index(seconds)}"


@dataclass
class CircadianConfig:
    lights_on_hour: int = 7
    lights_off_hour: int = 19
    hour_bin_size: int = 6

    def is_night(self, hour: int) -> bool:
        if self.lights_on_hour < self.lights_off_hour:
            return not (self.lights_on_hour <= hour < self.lights_off_hour)
        return self.lights_off_hour <= hour < self.lights_on_hour

    def day_night_token(self, hour: int) -> str:
        return "NIGHT" if self.is_night(hour) else "DAY"

    def hour_bin_token(self, hour: int) -> str:
        b = (hour // self.hour_bin_size) * self.hour_bin_size
        return f"HOUR_BIN_{b}_{b + self.hour_bin_size}"


@dataclass
class TokenizationConfig:
    include_partner_genotype: bool = True
    include_hour_bin: bool = True


def build_event_tokens(*, vocab: Vocab, event_name: str, is_dyad: bool, is_active: Optional[bool],
                       duration_seconds: float, partner_genotype: Optional[str],
                       duration_binner: DurationBinner, config: TokenizationConfig) -> List[int]:
    tokens: List[int] = []
    base = canonicalize_event(event_name)
    if is_dyad:
        assert is_active is not None
        role = "ACT" if is_active else "PAS"
        tokens.append(vocab.get_id(f"{base}_{role}"))
        if config.include_partner_genotype and partner_genotype:
            tokens.append(vocab.get_id(f"PARTNER_{partner_genotype.upper()}"))
    else:
        tokens.append(vocab.get_id(base))
    tokens.append(vocab.get_id(duration_binner.token(duration_seconds)))
    return tokens


def window_context_tokens(*, vocab: Vocab, setup: str, self_genotype: str, hour: int,
                          circadian: CircadianConfig, config: TokenizationConfig) -> List[int]:
    ctx = [
        vocab.get_id(f"SETUP_{setup.upper()}"),
        vocab.get_id(f"SELF_{self_genotype.upper()}"),
        vocab.get_id(circadian.day_night_token(hour)),
    ]
    if config.include_hour_bin:
        ctx.append(vocab.get_id(circadian.hour_bin_token(hour)))
    return ctx


