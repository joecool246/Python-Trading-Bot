"""Provenance and point-in-time joins for research-only financial events."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

EVENT_FEATURES = ["event_count", "event_sentiment", "event_direction", "event_uncertainty",
                  "event_risk", "event_surprise", "event_guidance", "event_corporate"]
KINDS = {"sec_filing", "earnings", "news", "disclosure", "synthetic"}
EVENT_TYPES = {"earnings", "guidance", "risk", "corporate", "other"}


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Source:
    symbol: str
    kind: str
    url: str
    published_at: str
    observed_at: str
    rights: str
    text: str

    def __post_init__(self):
        if self.kind not in KINDS or not re.fullmatch(r"[A-Z][A-Z0-9.-]*", self.symbol):
            raise ValueError("Invalid source kind or symbol")
        url = urlsplit(self.url)
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("Use a public HTTPS reference without credentials or query parameters")
        if self.kind == "sec_filing" and url.hostname != "www.sec.gov":
            raise ValueError("SEC filings require a canonical SEC archive reference")
        if not isinstance(self.rights, str) or not self.rights.strip():
            raise ValueError("Document processing rights must be recorded")
        if not isinstance(self.text, str) or not 1 <= len(self.text) <= 12000:
            raise ValueError("Provide a source excerpt of 1 to 12,000 characters")
        if timestamp(self.observed_at) < timestamp(self.published_at):
            raise ValueError("Source cannot be observed before publication")

    @property
    def identity(self) -> str:
        return sha256(json.dumps([self.symbol, self.kind, self.url, timestamp(self.published_at).isoformat(),
                                  self.text], ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class Interpretation:
    sentiment: int
    direction: int
    earnings_surprise: int | None
    guidance_change: int | None
    material_risk: bool
    event_type: str
    uncertainty: float
    evidence: str

    @classmethod
    def parse(cls, payload: dict, source: Source):
        required = set(cls.__annotations__)
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("Unexpected extraction fields")
        for key in ("sentiment", "direction", "earnings_surprise", "guidance_change"):
            value = payload[key]
            if key in {"earnings_surprise", "guidance_change"} and value is None:
                continue
            if type(value) is not int or value not in {-1, 0, 1}:
                raise ValueError("Extraction direction must be -1, 0 or 1")
        if type(payload["material_risk"]) is not bool or payload["event_type"] not in EVENT_TYPES:
            raise ValueError("Invalid extracted event type or risk flag")
        uncertainty = payload["uncertainty"]
        evidence = payload["evidence"]
        if (type(uncertainty) not in {int, float} or not math.isfinite(uncertainty)
                or not 0 <= uncertainty <= 1 or not isinstance(evidence, str)
                or not 15 <= len(evidence) <= 350 or evidence not in source.text):
            raise ValueError("Extraction must include a verbatim source quote and bounded uncertainty")
        return cls(**payload)


@dataclass(frozen=True)
class Event:
    source: Source
    interpretation: Interpretation
    extracted_at: str
    provider: str
    model: str

    def __post_init__(self):
        if timestamp(self.extracted_at) < timestamp(self.source.observed_at):
            raise ValueError("Extraction cannot predate source observation")
        if (self.source.kind == "synthetic") != (self.provider == "fixture"):
            raise ValueError("Synthetic source and fixture provenance must agree")
        if self.provider not in {"fixture", "openai", "anthropic"} or not self.model:
            raise ValueError("Missing or invalid model provenance")

    @property
    def available(self) -> datetime:
        return timestamp(self.extracted_at)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict):
        source = Source(**row["source"])
        return cls(source, Interpretation.parse(row["interpretation"], source),
                   row["extracted_at"], row["provider"], row["model"])


def read_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_events(path: str) -> list[Event]:
    return [Event.from_dict(row) for row in read_jsonl(path)]


def unique(events: list[Event]) -> list[Event]:
    # Earliest valid extraction of an exact document/version wins. Corrections
    # are distinct versions and become visible only at their own completion time.
    seen = set()
    result = []
    for event in sorted(events, key=lambda item: (item.available, item.provider, item.model, item.source.identity)):
        identity = event.source.identity
        if identity not in seen:
            result.append(event)
            seen.add(identity)
    return result


def event_features(index: pd.DatetimeIndex, events: list[Event], max_age_days: int = 7) -> pd.DataFrame:
    if type(max_age_days) is not int or not 1 <= max_age_days <= 90:
        raise ValueError("Event age must be 1 to 90 calendar days")
    # Freeze news before 13:00 New York, including scheduled early closes.
    # Completed prices are still read after the market close by the simulator.
    cutoffs = (index.tz_convert("UTC").tz_localize(None).normalize()
               + pd.Timedelta(hours=13)).tz_localize("America/New_York")
    output = np.zeros((len(index), len(EVENT_FEATURES)))
    next_version = {}
    for event in reversed(unique(events)):
        reference = event.source.symbol, event.source.url
        replacement = next_version.get(reference)
        next_version[reference] = event.available
        published = pd.Timestamp(timestamp(event.source.published_at))
        first = cutoffs.searchsorted(pd.Timestamp(event.available), side="right")
        last = cutoffs.searchsorted(published + pd.Timedelta(days=max_age_days), side="right")
        if replacement is not None:
            last = min(last, cutoffs.searchsorted(pd.Timestamp(replacement), side="right"))
        if first >= last:
            continue
        item = event.interpretation
        signal = 1 - item.uncertainty
        values = [1, signal * item.sentiment, signal * item.direction, item.uncertainty,
                  int(item.material_risk), signal * (item.earnings_surprise or 0),
                  signal * (item.guidance_change or 0), int(item.event_type == "corporate")]
        output[first:last] += values
    output[:, 1:] /= np.maximum(output[:, :1], 1)
    return pd.DataFrame(output, index=index, columns=EVENT_FEATURES)
