from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core import LessonExperiment


Coordinate = tuple[int, int, int]


def _rank(dp: int, cp: int, ep: int, cp_size: int, ep_size: int) -> int:
    return ((dp * cp_size) + cp) * ep_size + ep


def _mesh(dp_size: int, cp_size: int, ep_size: int) -> dict[int, Coordinate]:
    return {
        _rank(dp, cp, ep, cp_size, ep_size): (dp, cp, ep)
        for dp in range(dp_size)
        for cp in range(cp_size)
        for ep in range(ep_size)
    }


def _axis_groups(
    mesh: dict[int, Coordinate],
    axis: int,
) -> list[list[int]]:
    grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for rank, coordinate in mesh.items():
        fixed = tuple(
            value for index, value in enumerate(coordinate) if index != axis
        )
        grouped[fixed].append(rank)
    return [
        sorted(group) for _, group in sorted(grouped.items())
    ]


def _gradient_sum(
    weight: float,
    examples: list[tuple[float, float]],
) -> tuple[float, int]:
    gradient = sum((weight * feature - target) * feature for feature, target in examples)
    return gradient, len(examples)


def _run() -> dict[str, Any]:
    meshes = {
        "A_fsdp": _mesh(8, 1, 1),
        "B_fsdp_ep": _mesh(2, 1, 4),
        "C_ep_cp": _mesh(1, 2, 4),
    }
    c_mesh = meshes["C_ep_cp"]
    cp_groups = _axis_groups(c_mesh, axis=1)
    ep_groups = _axis_groups(c_mesh, axis=2)
    group_intersections = [
        len(set(cp_group) & set(ep_group))
        for cp_group in cp_groups
        for ep_group in ep_groups
    ]

    rank_losses = [
        [1.0, 3.0],
        [10.0],
    ]
    global_loss = (
        sum(sum(losses) for losses in rank_losses)
        / sum(len(losses) for losses in rank_losses)
    )
    mean_of_rank_means = sum(
        sum(losses) / len(losses) for losses in rank_losses
    ) / len(rank_losses)

    router_records: list[dict[str, Any]] = [
        {"selected": 0, "probabilities": [0.70, 0.10, 0.10, 0.10]},
        {"selected": 1, "probabilities": [0.15, 0.60, 0.15, 0.10]},
        {"selected": 1, "probabilities": [0.05, 0.80, 0.10, 0.05]},
        {"selected": 3, "probabilities": [0.15, 0.10, 0.20, 0.55]},
        {"selected": 0, "probabilities": [0.90, 0.03, 0.03, 0.04]},
        {"selected": 2, "probabilities": [0.05, 0.10, 0.75, 0.10]},
    ]
    router_shards = [router_records[::2], router_records[1::2]]

    def aggregate(
        records: list[dict[str, Any]],
    ) -> tuple[list[int], list[float]]:
        counts = [0, 0, 0, 0]
        probability_sums = [0.0, 0.0, 0.0, 0.0]
        for record in records:
            selected = int(record["selected"])
            probabilities = [float(value) for value in record["probabilities"]]
            counts[selected] += 1
            for expert, probability in enumerate(probabilities):
                probability_sums[expert] += probability
        return counts, probability_sums

    global_counts, global_probability_sums = aggregate(router_records)
    selected_probability_total = sum(
        float(record["probabilities"][int(record["selected"])])
        for record in router_records
    )
    shard_stats = [aggregate(shard) for shard in router_shards]
    reduced_counts = [
        sum(stats[0][expert] for stats in shard_stats) for expert in range(4)
    ]
    reduced_probability_sums = [
        sum(stats[1][expert] for stats in shard_stats) for expert in range(4)
    ]
    router_aux = 4 * sum(
        (count / len(router_records))
        * (probability_sum / len(router_records))
        for count, probability_sum in zip(
            global_counts,
            global_probability_sums,
        )
    )

    examples = [
        (1.0, 2.0),
        (2.0, 1.0),
        (3.0, 4.0),
        (4.0, 3.0),
        (5.0, 7.0),
    ]
    example_shards = [examples[::2], examples[1::2]]
    weight = 0.5
    learning_rate = 0.1
    reference_sum, reference_count = _gradient_sum(weight, examples)
    reference_update = weight - learning_rate * reference_sum / reference_count
    shard_gradients = [
        _gradient_sum(weight, shard) for shard in example_shards
    ]
    reduced_sum = sum(gradient for gradient, _ in shard_gradients)
    reduced_count = sum(count for _, count in shard_gradients)
    distributed_update = weight - learning_rate * reduced_sum / reduced_count

    full_sequence = list(range(12))
    context_shards = [full_sequence[:6], full_sequence[6:]]
    recombined_sequence = [
        token for shard in context_shards for token in shard
    ]
    dispatched_tokens = [
        (token_id, int(record["selected"]))
        for token_id, record in enumerate(router_records)
    ]

    checks = {
        "all canonical meshes contain eight unique ranks": all(
            len(mesh) == 8 and set(mesh) == set(range(8))
            for mesh in meshes.values()
        ),
        "EP and CP groups are orthogonal in the combined mesh": (
            all(size <= 1 for size in group_intersections)
            and sorted(cp_groups) == [[0, 4], [1, 5], [2, 6], [3, 7]]
            and sorted(ep_groups) == [[0, 1, 2, 3], [4, 5, 6, 7]]
        ),
        "global valid-token loss differs from the wrong mean of rank means": (
            global_loss != mean_of_rank_means
        ),
        "all-reduced router sufficient statistics match the global reference": (
            reduced_counts == global_counts
            and all(
                abs(left - right) < 1e-12
                for left, right in zip(
                    reduced_probability_sums,
                    global_probability_sums,
                )
            )
        ),
        "router_probability_sums_include_the_full_softmax_for_every_token": (
            all(
                abs(sum(record["probabilities"]) - 1.0) < 1e-12
                and int(record["selected"])
                == max(
                    range(4),
                    key=lambda expert: record["probabilities"][expert],
                )
                for record in router_records
            )
            and abs(sum(global_probability_sums) - len(router_records)) < 1e-12
            and sum(global_probability_sums) > selected_probability_total
        ),
        "sharded gradient sums reproduce the single-process update": (
            abs(distributed_update - reference_update) < 1e-12
        ),
        "context shards reconstruct the original token order": (
            recombined_sequence == full_sequence
        ),
        "expert dispatch accounts for each logical token exactly once": (
            sorted(token_id for token_id, _ in dispatched_tokens)
            == list(range(len(router_records)))
        ),
    }

    return {
        "summary": (
            "在单进程内精确计算三套八 rank mesh、EP/CP 正交分组、全局 loss"
            " 归一化、router sufficient statistics 和分片梯度更新；这些是"
            "分布式数值语义检查，不是 NCCL 吞吐结果。"
        ),
        "metrics": {
            "mesh_coordinates": {
                name: {
                    str(rank): list(coordinate)
                    for rank, coordinate in mesh.items()
                }
                for name, mesh in meshes.items()
            },
            "cp_groups": cp_groups,
            "ep_groups": ep_groups,
            "global_valid_token_loss": global_loss,
            "incorrect_mean_of_rank_means": mean_of_rank_means,
            "router_counts": global_counts,
            "router_probability_sums": global_probability_sums,
            "selected_probability_total": selected_probability_total,
            "router_aux_without_alpha": router_aux,
            "single_process_updated_weight": reference_update,
            "sharded_updated_weight": distributed_update,
        },
        "checks": checks,
    }


LESSON = LessonExperiment(
    lesson_id="13",
    title="8 卡分布式训练：FSDP2、EP、CP 与数值一致性",
    question="不同并行轴怎样保持正交，并让全局 loss 与参数更新等价于单进程参考？",
    run=_run,
)
