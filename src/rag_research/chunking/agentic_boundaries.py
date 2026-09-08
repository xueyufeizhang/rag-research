from fractions import Fraction


def validate_boundaries(
    boundaries: list[tuple[int, int]],
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
    allow_short_final: bool = True,
) -> None:
    if min_sentences <= 0:
        raise ValueError("agentic min sentences must be positive")
    if max_sentences < min_sentences:
        raise ValueError(
            "agentic max sentences must be greater than or equal to min sentences"
        )
    validate_boundary_structure(boundaries, sentence_count)

    for index, (start, end) in enumerate(boundaries):
        size = end - start + 1
        is_final = index == len(boundaries) - 1
        if size > max_sentences:
            raise ValueError(f"chunk size {size} exceeds maximum {max_sentences}")
        if size < min_sentences and (not is_final or not allow_short_final):
            raise ValueError(
                f"chunk size {size} is below minimum {min_sentences}"
            )


def validate_boundary_structure(
    boundaries: list[tuple[int, int]],
    sentence_count: int,
) -> None:
    if sentence_count <= 0:
        raise ValueError("sentence count must be positive")
    if not boundaries:
        raise ValueError("no chunk boundaries returned")

    expected_start = 1
    for start, end in boundaries:
        if type(start) is not int or type(end) is not int:
            raise ValueError("boundary start and end indexes must be integers")
        if start != expected_start:
            raise ValueError(
                f"expected chunk to start at {expected_start}, got {start}"
            )
        if start < 1 or end < start or end > sentence_count:
            raise ValueError(
                f"invalid boundary ({start}, {end}) for {sentence_count} sentences"
            )
        expected_start = end + 1

    if boundaries[-1][1] != sentence_count:
        raise ValueError(
            f"last chunk ends at {boundaries[-1][1]}, expected {sentence_count}"
        )


def project_boundaries(
    boundaries: list[tuple[int, int]],
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
    allow_short_final: bool = True,
) -> list[tuple[int, int]]:
    """Project model boundaries onto hard size limits with minimal movement."""
    if min_sentences <= 0:
        raise ValueError("agentic min sentences must be positive")
    if max_sentences < min_sentences:
        raise ValueError(
            "agentic max sentences must be greater than or equal to min sentences"
        )
    validate_boundary_structure(boundaries, sentence_count)

    if all(
        (end - start + 1) <= max_sentences
        and (
            index == len(boundaries) - 1
            and allow_short_final
            or (end - start + 1) >= min_sentences
        )
        for index, (start, end) in enumerate(boundaries)
    ):
        return list(boundaries)

    proposed_chunk_count = len(boundaries)
    feasible_chunk_counts = [
        chunk_count
        for chunk_count in range(1, sentence_count + 1)
        if (
            (
                (chunk_count - 1) * min_sentences + 1
                if allow_short_final
                else chunk_count * min_sentences
            )
            <= sentence_count
            <= chunk_count * max_sentences
        )
    ]
    if not feasible_chunk_counts:
        raise ValueError(
            "sentence count cannot be partitioned under the configured "
            "agentic minimum and maximum"
        )

    projected_chunk_count = min(
        feasible_chunk_counts,
        key=lambda count: (abs(count - proposed_chunk_count), count),
    )
    proposed_endpoints = [0, *(end for _, end in boundaries)]

    target_endpoints: list[Fraction] = []
    for boundary_index in range(1, projected_chunk_count):
        model_position = Fraction(
            boundary_index * proposed_chunk_count,
            projected_chunk_count,
        )
        left_index = model_position.numerator // model_position.denominator
        fraction = model_position - left_index
        left_endpoint = proposed_endpoints[left_index]
        right_endpoint = proposed_endpoints[left_index + 1]
        target_endpoints.append(
            Fraction(left_endpoint)
            + fraction * (right_endpoint - left_endpoint)
        )

    minimum_final_size = 1 if allow_short_final else min_sentences

    def feasible_previous_ends(chunk_index: int) -> range:
        remaining_chunks = projected_chunk_count - chunk_index
        minimum_remaining = (
            (remaining_chunks - 1) * min_sentences + minimum_final_size
        )
        earliest = max(
            chunk_index * min_sentences,
            sentence_count - remaining_chunks * max_sentences,
        )
        latest = min(
            chunk_index * max_sentences,
            sentence_count - minimum_remaining,
        )
        return range(earliest, latest + 1)

    # Store only the next endpoint for each state. With equal total costs the
    # smaller next endpoint gives the same lexicographically smallest path as
    # comparing complete endpoint tuples in the recursive formulation.
    choices: list[dict[int, int]] = [
        {} for _ in range(projected_chunk_count)
    ]
    final_index = projected_chunk_count - 1
    next_costs = {
        previous_end: Fraction(0)
        for previous_end in feasible_previous_ends(final_index)
    }
    choices[final_index] = {
        previous_end: sentence_count for previous_end in next_costs
    }
    for chunk_index in range(final_index - 1, -1, -1):
        costs: dict[int, Fraction] = {}
        for previous_end in feasible_previous_ends(chunk_index):
            best: tuple[Fraction, int] | None = None
            for current_end in range(
                previous_end + min_sentences,
                min(previous_end + max_sentences, sentence_count - 1) + 1,
            ):
                tail_cost = next_costs.get(current_end)
                if tail_cost is None:
                    continue
                boundary_cost = abs(
                    Fraction(current_end) - target_endpoints[chunk_index]
                )
                candidate = (boundary_cost + tail_cost, current_end)
                if best is None or candidate < best:
                    best = candidate
            if best is not None:
                costs[previous_end] = best[0]
                choices[chunk_index][previous_end] = best[1]
        next_costs = costs

    if 0 not in next_costs:
        raise ValueError(
            "failed to project agentic boundaries onto configured limits"
        )

    projected: list[tuple[int, int]] = []
    previous_end = 0
    for chunk_choices in choices:
        current_end = chunk_choices[previous_end]
        projected.append((previous_end + 1, current_end))
        previous_end = current_end
    return projected


def strict_partition_is_feasible(
    sentence_count: int,
    *,
    min_sentences: int,
    max_sentences: int,
) -> bool:
    if sentence_count <= 0:
        return False
    return any(
        chunk_count * min_sentences <= sentence_count
        <= chunk_count * max_sentences
        for chunk_count in range(1, sentence_count + 1)
    )


def rebalance_document_boundaries(
    boundaries: list[tuple[int, int]],
    *,
    sentence_count: int,
    min_sentences: int,
    max_sentences: int,
) -> list[tuple[int, int]]:
    """Repair short chunks while preserving semantic boundaries when possible."""
    validate_boundary_structure(boundaries, sentence_count)
    if any(end - start + 1 > max_sentences for start, end in boundaries):
        raise ValueError(
            "document-level rebalancing received an oversized chunk"
        )
    if not strict_partition_is_feasible(
        sentence_count,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    ):
        return project_boundaries(
            boundaries,
            sentence_count=sentence_count,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            allow_short_final=True,
        )

    endpoints = [end for _, end in boundaries]
    while True:
        previous_end = 0
        sizes: list[int] = []
        for endpoint in endpoints:
            sizes.append(endpoint - previous_end)
            previous_end = endpoint

        short_index = next(
            (
                index
                for index, size in enumerate(sizes)
                if size < min_sentences
            ),
            None,
        )
        if short_index is None:
            break

        merge_candidates: list[tuple[int, int, str]] = []
        if short_index > 0:
            merged_size = sizes[short_index - 1] + sizes[short_index]
            if merged_size <= max_sentences:
                merge_candidates.append((merged_size, 0, "left"))
        if short_index + 1 < len(sizes):
            merged_size = sizes[short_index] + sizes[short_index + 1]
            if merged_size <= max_sentences:
                merge_candidates.append((merged_size, 1, "right"))

        if merge_candidates:
            _, _, merge_side = min(merge_candidates)
            boundary_to_remove = (
                short_index - 1 if merge_side == "left" else short_index
            )
            del endpoints[boundary_to_remove]
            continue

        needed = min_sentences - sizes[short_index]
        left_available = (
            sizes[short_index - 1] - min_sentences
            if short_index > 0
            else 0
        )
        right_available = (
            sizes[short_index + 1] - min_sentences
            if short_index + 1 < len(sizes)
            else 0
        )
        if left_available + right_available < needed:
            return project_boundaries(
                boundaries,
                sentence_count=sentence_count,
                min_sentences=min_sentences,
                max_sentences=max_sentences,
                allow_short_final=False,
            )

        take_from_left = min(left_available, needed)
        take_from_right = needed - take_from_left
        if take_from_left:
            endpoints[short_index - 1] -= take_from_left
        if take_from_right:
            endpoints[short_index] += take_from_right

    rebalanced: list[tuple[int, int]] = []
    previous_end = 0
    for endpoint in endpoints:
        rebalanced.append((previous_end + 1, endpoint))
        previous_end = endpoint

    validate_boundaries(
        rebalanced,
        sentence_count=sentence_count,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
        allow_short_final=False,
    )
    return rebalanced
