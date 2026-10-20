from __future__ import annotations

from src.p01_preparation._02_prompt_creation.base import PromptBuilder
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_a import (
    CHUNK_A_CITATION_PATTERN,
    PromptChunkA,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_b import PromptChunkB
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_c import PromptChunkC
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_d import PromptChunkD
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_strategies import (
    ChunkStrategyKind,
    CitationChunkStrategy,
    make_chunk_strategy,
)
from src.p01_preparation._02_prompt_creation.chunk_prompts.chunk_base import (
    CHUNK_STRATEGY,
    CHUNK_STRATEGY_KIND,
)
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_a import PromptA
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_b import PromptB
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_c import PromptC
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_d import PromptD
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_e import PromptE
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_f import PromptF
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_g import PromptG
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_h import PromptH
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_i import PromptI
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_i_v2 import PromptIV2
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_i_v3 import PromptIV3
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_j import PromptJ
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_j_v1 import PromptJV1
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_j_v2 import PromptJV2
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_j_v3 import PromptJV3
from src.p01_preparation._02_prompt_creation.single_prompts.prompt_j_v4 import PromptJV4

_ALLOWED = (
    "a, b, c, d, e, f, g, h, i, i_v2, i_v3, j, j_v1, j_v2, j_v3, j_v4, "
    "chunk_a, chunk_b, chunk_c, chunk_d"
)


def _resolve_chunk_strategy(
    chunk_strategy: CitationChunkStrategy | None,
    chunk_strategy_kind: ChunkStrategyKind | None,
) -> CitationChunkStrategy:
    if chunk_strategy is not None:
        return chunk_strategy
    if chunk_strategy_kind is not None:
        return make_chunk_strategy(
            chunk_strategy_kind,
            words_before=100,
            words_after=100,
            sentences_before=5,
            sentences_after=5,
        )
    return CHUNK_STRATEGY


def build_prompt_builder(
    prompt_name: str,
    *,
    chunk_strategy_kind: ChunkStrategyKind | None = None,
    chunk_strategy: CitationChunkStrategy | None = None,
) -> PromptBuilder:
    normalized = prompt_name.strip().lower()
    strategy = _resolve_chunk_strategy(chunk_strategy, chunk_strategy_kind)
    if normalized in {"a", "prompt_a"}:
        return PromptA()
    if normalized in {"b", "prompt_b"}:
        return PromptB()
    if normalized in {"c", "prompt_c"}:
        return PromptC()
    if normalized in {"d", "prompt_d"}:
        return PromptD()
    if normalized in {"e", "prompt_e"}:
        return PromptE()
    if normalized in {"f", "prompt_f"}:
        return PromptF()
    if normalized in {"g", "prompt_g"}:
        return PromptG()
    if normalized in {"h", "prompt_h"}:
        return PromptH()
    if normalized in {"i", "prompt_i"}:
        return PromptI()
    if normalized in {"i_v2", "prompt_i_v2"}:
        return PromptIV2()
    if normalized in {"i_v3", "prompt_i_v3"}:
        return PromptIV3()
    if normalized in {"j", "prompt_j"}:
        return PromptJ()
    if normalized in {"j_v1", "prompt_j_v1"}:
        return PromptJV1()
    if normalized in {"j_v2", "prompt_j_v2"}:
        return PromptJV2()
    if normalized in {"j_v3", "prompt_j_v3"}:
        return PromptJV3()
    if normalized in {"j_v4", "prompt_j_v4"}:
        return PromptJV4()
    if normalized in {"chunk_a", "prompt_chunk_a"}:
        return PromptChunkA(citation_pattern=CHUNK_A_CITATION_PATTERN, chunk_strategy=strategy)
    if normalized in {"chunk_b", "prompt_chunk_b"}:
        return PromptChunkB(chunk_strategy=strategy)
    if normalized in {"chunk_c", "prompt_chunk_c"}:
        return PromptChunkC(chunk_strategy=strategy)
    if normalized in {"chunk_d", "prompt_chunk_d"}:
        return PromptChunkD()
    raise ValueError(
        f"Unsupported prompt style {prompt_name!r}. Allowed: {_ALLOWED}. "
        f"Default chunk strategy kind: {CHUNK_STRATEGY_KIND!r}."
    )
