from dataclasses import dataclass


@dataclass(frozen=True)
class PromptContext:
    """Input payload for a prompt template."""

    case_name: str
    citation: str
    opinion_text: str
