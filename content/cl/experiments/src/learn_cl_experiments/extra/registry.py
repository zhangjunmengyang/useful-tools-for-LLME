from __future__ import annotations

from ..extra_core import ExtraExperiment
from . import (
    budget,
    buffer,
    capacity,
    compose,
    conflict,
    disagree,
    distill,
    eligible,
    ewcmem,
    evolve,
    gendream,
    graduate,
    keepfail,
    longtail,
    onpolicy,
    ortho,
    plastic,
    rollback,
    route,
    selfedit,
    seqedit,
    shadow,
    skill,
    sleep,
    stale,
    surprise,
    tombstone,
    unplug,
)

EXTRAS: dict[str, ExtraExperiment] = {
    item.EXPERIMENT.extra_id: item.EXPERIMENT
    for item in (
        unplug,
        distill,
        skill,
        selfedit,
        conflict,
        evolve,
        route,
        capacity,
        sleep,
        surprise,
        seqedit,
        onpolicy,
        ortho,
        ewcmem,
        plastic,
        graduate,
        buffer,
        gendream,
        stale,
        shadow,
        eligible,
        budget,
        tombstone,
        longtail,
        compose,
        disagree,
        keepfail,
        rollback,
    )
}


def get_extra(extra_id: str) -> ExtraExperiment:
    try:
        return EXTRAS[extra_id]
    except KeyError as error:
        raise KeyError(f"Unknown extra experiment: {extra_id}") from error
