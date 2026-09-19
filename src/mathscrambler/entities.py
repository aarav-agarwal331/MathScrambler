"""Scenario swaps: new names, objects and places for a blueprint's entities (Step C).

The `fast` role proposes; `check_entity_swap` decides. A proposal that drops a
mathematically loaded word, introduces a number, or leaves an entity unchanged
is rejected with a reason, and the reasons go back to the model on the one
retry. Running out of accepted proposals never fails a run: the numbers still
change, so a variant with its original scenery is a weaker variant, not a
broken one — it is recorded as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from mathscrambler import prompts
from mathscrambler.blueprint import Blueprint
from mathscrambler.config import Role
from mathscrambler.ollama_client import OllamaClient, StructuredCallError, TimingRecord
from mathscrambler.sampler import check_entity_swap

# Asked for on top of what is still needed, so one round of rejections rarely
# forces a second call.
SPARE_PROPOSALS = 2

# `fast` first, as the spec assigns; the reasoner only when the fast model's
# proposals keep failing the guard (a 7B model tends to change the place and
# leave "adult tickets" as it was). Cheaper than settling for original scenery.
PROPOSER_ROLES: tuple[Role, ...] = ("fast", "reasoner")
CALLS_PER_ROLE = 2


class EntityReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # No `pattern` here: a regex in the schema made the fast model's engine
    # emit nothing at all (observed live). `{E1}` is accepted as a spelling of
    # `E1` in `_judge` instead — same slot, unambiguous.
    slot: str
    replacement: str


class EntityProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replacements: list[EntityReplacement]


class EntityProposals(BaseModel):
    """The `fast` role's reply: several complete scenery swaps."""

    model_config = ConfigDict(extra="forbid")

    proposals: list[EntityProposal]


@dataclass(frozen=True)
class EntityResult:
    maps: list[dict[str, str]]  # exactly n entries, one per variant
    accepted: int  # distinct proposals that passed the guard
    rejected: list[str] = field(default_factory=list)
    timings: list[TimingRecord] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Every variant got its own accepted proposal (no cycling, no originals)."""
        return self.accepted >= len(self.maps)


def _judge(bp: Blueprint, proposal: EntityProposal) -> tuple[dict[str, str] | None, str | None]:
    """The proposal as a slot map, or why it is unusable."""
    originals = bp.original_entities()
    mapping: dict[str, str] = {}
    for item in proposal.replacements:
        slot = item.slot.strip().strip("{}").strip()  # the template's `{E1}` names the same slot
        if slot not in originals:
            return None, f"{item.slot!r} is not an entity slot"
        if slot in mapping:
            return None, f"{slot} was given twice"
        why = check_entity_swap(originals[slot], item.replacement)
        if why:
            return None, f"{slot} {originals[slot]!r} -> {item.replacement!r}: {why}"
        mapping[slot] = item.replacement.strip()
    if missing := sorted(set(originals) - set(mapping)):
        return None, f"proposal left {', '.join(missing)} unchanged"
    return mapping, None


async def propose_entities(
    client: OllamaClient,
    bp: Blueprint,
    *,
    n: int,
    roles: tuple[Role, ...] = PROPOSER_ROLES,
    calls_per_role: int = CALLS_PER_ROLE,
) -> EntityResult:
    """`n` entity maps for `bp`, one per variant.

    Each role in `roles` gets up to `calls_per_role` tries (the rejections of
    the previous try fed back) before the next role is asked. Short of accepted
    proposals at the end, the accepted ones are reused in rotation; with none
    at all, every variant keeps the original scenery. Either shortfall is
    visible in `accepted` and `rejected`.
    """
    if not bp.entities:
        return EntityResult(maps=[{} for _ in range(n)], accepted=n)

    accepted: list[dict[str, str]] = []
    rejected: list[str] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    timings: list[TimingRecord] = []
    transcript: list[dict[str, Any]] = []
    system = {"role": "system", "content": prompts.load("entity_swap")}

    for role in roles:
        for _call in range(calls_per_role):
            user = {
                "role": "user",
                "content": prompts.render(
                    "entity_swap_user",
                    count=n - len(accepted) + SPARE_PROPOSALS,
                    template_md=bp.template_md,
                    entities=bp.entities,
                    rejected=rejected[-6:],
                ),
            }
            transcript.append({"role": "user", "proposer": role, "content": user["content"]})
            try:
                result = await client.structured(role, [system, user], EntityProposals)
            except StructuredCallError as e:
                # This role cannot produce the shape at all; asking it again with
                # feedback about proposals it never made would only burn time.
                rejected.append(f"no valid JSON from the {role} model: {e}")
                break
            timings.extend(result.timings)
            rejected.extend(f"{role} reply rejected by the client: {why}" for why in result.errors)
            transcript.append({"role": "assistant", "content": result.value.model_dump_json(indent=2)})
            for proposal in result.value.proposals:
                mapping, why = _judge(bp, proposal)
                if mapping is None:
                    rejected.append(why or "rejected")
                    continue
                key = tuple(sorted(mapping.items()))
                if key in seen:
                    rejected.append(f"duplicate proposal {dict(key)}")
                    continue
                seen.add(key)
                accepted.append(mapping)
                if len(accepted) >= n:
                    break
            if len(accepted) >= n:
                break
        if len(accepted) >= n:
            break

    maps = list(accepted[:n])
    while len(maps) < n:
        maps.append(accepted[len(maps) % len(accepted)] if accepted else bp.original_entities())
    return EntityResult(
        maps=maps, accepted=len(accepted), rejected=rejected, timings=timings, transcript=transcript
    )
