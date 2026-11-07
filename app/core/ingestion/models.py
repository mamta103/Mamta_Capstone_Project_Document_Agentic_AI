from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentSection:
    text: str
    metadata: dict


class DocumentLoadError(Exception):
    """Raised with a user-safe message when a file cannot be loaded."""


@dataclass
class IngestionResult:
    sections: list[DocumentSection] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
