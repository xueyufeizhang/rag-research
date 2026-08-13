import argparse
import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data/raw/a_christmas_carol.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/evaluation/carol_canonical.json"


# Sentence ranges are 1-based and inclusive. ``supports`` contains 1-based
# indices into each question's gold_answer_points list.
EVIDENCE_RANGES = {
    "carol_easy_001": [(279, 279, [1]), (269, 270, [3]), (351, 356, [2, 4])],
    "carol_easy_002": [(13, 13, [1]), (102, 105, [2]), (790, 801, [3]), (1543, 1550, [4])],
    "carol_easy_003": [(137, 150, [1, 2, 4]), (944, 963, [2, 4]), (1508, 1521, [3])],
    "carol_easy_004": [(533, 556, [1, 2, 3]), (586, 590, [2, 3]), (595, 608, [4])],
    "carol_easy_005": [(1021, 1045, [1, 2, 3, 4])],
    "carol_easy_006": [(1378, 1392, [1]), (1435, 1451, [2]), (1471, 1492, [3]), (1505, 1521, [3]), (1543, 1551, [4])],
    "carol_medium_001": [(80, 96, [1, 2]), (120, 150, [3]), (165, 184, [4])],
    "carol_medium_002": [(165, 184, [1, 2, 3]), (848, 858, [4])],
    "carol_medium_003": [(302, 319, [1, 2, 3]), (335, 356, [2, 3, 4])],
    "carol_medium_004": [(466, 477, [1, 2, 4]), (479, 504, [3, 4])],
    "carol_medium_005": [(615, 648, [1, 2]), (656, 683, [3, 4])],
    "carol_medium_006": [(771, 775, [1]), (790, 805, [1, 2]), (843, 858, [2, 4]), (863, 880, [3]), (887, 888, [1])],
    "carol_medium_007": [(919, 925, [1, 3]), (936, 963, [1, 3]), (987, 1005, [2, 3]), (1505, 1521, [4])],
    "carol_medium_008": [(1048, 1070, [1, 2]), (1076, 1094, [2]), (1342, 1370, [3, 4])],
    "carol_hard_001": [(337, 356, [1]), (466, 477, [2]), (499, 504, [2]), (619, 648, [2]), (675, 683, [2]), (771, 775, [3]), (790, 801, [3]), (843, 858, [3]), (887, 888, [3]), (1021, 1045, [3]), (1076, 1094, [4]), (1161, 1215, [4]), (1342, 1370, [4, 5]), (1378, 1392, [5])],
    "carol_hard_002": [(595, 608, [1]), (619, 648, [2]), (675, 683, [2]), (771, 775, [3]), (790, 801, [3]), (843, 858, [3, 4]), (887, 888, [3])],
    "carol_hard_003": [(80, 105, [1, 2]), (709, 733, [3]), (771, 775, [4]), (838, 845, [4]), (887, 888, [4]), (1412, 1418, [5])],
    "carol_hard_004": [(790, 801, [1]), (848, 858, [1, 2]), (1259, 1277, [3]), (1294, 1302, [3]), (1322, 1326, [3]), (1547, 1550, [4])],
    "carol_hard_005": [(165, 184, [1]), (302, 339, [2, 4]), (1021, 1045, [3, 4])],
    "carol_hard_006": [(1076, 1109, [1]), (1161, 1215, [2, 4]), (1259, 1302, [3]), (1322, 1326, [3]), (1342, 1370, [4, 5])],
    "carol_hard_007": [(843, 880, [1, 2, 3, 4])],
    "carol_hard_008": [(80, 105, [1]), (181, 184, [1]), (335, 339, [2]), (595, 608, [3]), (1543, 1551, [4])],
    "carol_hard_009": [(466, 477, [1, 4]), (499, 504, [1, 4]), (619, 648, [2, 4]), (675, 683, [2, 4]), (790, 801, [3, 4]), (848, 858, [3, 4])],
    "carol_hard_010": [(1378, 1392, [1]), (1435, 1451, [2]), (1471, 1492, [3]), (1505, 1521, [4]), (1543, 1551, [5])],
}


# These labels describe the story facts, not the entities emitted by any one
# chunking/extraction run. Alias handling can therefore be evaluated separately.
GRAPH_GOLD = {
    "carol_easy_001": (["Jacob Marley", "Scrooge", "Marley's Chain"], ["Jacob Marley||Scrooge", "Jacob Marley||Marley's Chain"]),
    "carol_easy_002": (["Bob Cratchit", "Scrooge", "Tiny Tim"], ["Bob Cratchit||Scrooge", "Bob Cratchit||Tiny Tim", "Scrooge||Tiny Tim"]),
    "carol_easy_003": (["Fred", "Scrooge"], ["Fred||Scrooge"]),
    "carol_easy_004": (["Fezziwig", "Scrooge", "Dick Wilkins"], ["Fezziwig||Scrooge", "Dick Wilkins||Fezziwig"]),
    "carol_easy_005": (["Ignorance", "Want", "Ghost of Christmas Present", "Mankind"], ["Ghost of Christmas Present||Ignorance", "Ghost of Christmas Present||Want", "Ignorance||Mankind", "Mankind||Want"]),
    "carol_easy_006": (["Scrooge", "Bob Cratchit", "Tiny Tim", "Fred", "Prize Turkey", "Charity Collector"], ["Prize Turkey||Scrooge", "Bob Cratchit||Scrooge", "Fred||Scrooge", "Scrooge||Tiny Tim", "Charity Collector||Scrooge"]),
    "carol_medium_001": (["Scrooge", "Fred", "Charity Collectors"], ["Fred||Scrooge", "Charity Collectors||Scrooge"]),
    "carol_medium_002": (["Scrooge", "Charity Collectors", "The Poor", "Prisons", "Union Workhouses", "Ghost of Christmas Present", "Tiny Tim"], ["Charity Collectors||Scrooge", "Prisons||Scrooge", "Scrooge||Union Workhouses", "Ghost of Christmas Present||Scrooge", "Scrooge||Tiny Tim"]),
    "carol_medium_003": (["Jacob Marley", "Scrooge", "Mankind"], ["Jacob Marley||Scrooge", "Jacob Marley||Mankind"]),
    "carol_medium_004": (["Ghost of Christmas Past", "Scrooge", "Ali Baba", "Fan"], ["Ghost of Christmas Past||Scrooge", "Ali Baba||Scrooge", "Fan||Scrooge"]),
    "carol_medium_005": (["Belle", "Scrooge", "Belle's Husband", "Belle's Family"], ["Belle||Scrooge", "Belle||Belle's Husband", "Belle||Belle's Family"]),
    "carol_medium_006": (["Scrooge", "Bob Cratchit", "Tiny Tim", "Mrs. Cratchit"], ["Bob Cratchit||Tiny Tim", "Bob Cratchit||Mrs. Cratchit", "Scrooge||Tiny Tim"]),
    "carol_medium_007": (["Fred", "Scrooge", "Fred's Household"], ["Fred||Scrooge", "Fred||Fred's Household"]),
    "carol_medium_008": (["Scrooge", "Ghost of Christmas Yet to Come", "Scrooge's Grave"], ["Ghost of Christmas Yet to Come||Scrooge", "Ghost of Christmas Yet to Come||Scrooge's Grave", "Scrooge||Scrooge's Grave"]),
    "carol_hard_001": (["Scrooge", "Jacob Marley", "Ghost of Christmas Past", "Ghost of Christmas Present", "Ghost of Christmas Yet to Come"], ["Jacob Marley||Scrooge", "Ghost of Christmas Past||Scrooge", "Ghost of Christmas Present||Scrooge", "Ghost of Christmas Yet to Come||Scrooge"]),
    "carol_hard_002": (["Scrooge", "Fezziwig", "Belle", "Bob Cratchit", "Tiny Tim"], ["Fezziwig||Scrooge", "Belle||Scrooge", "Bob Cratchit||Tiny Tim", "Scrooge||Tiny Tim"]),
    "carol_hard_003": (["Scrooge", "Ghost of Christmas Present", "Bob Cratchit", "Tiny Tim"], ["Ghost of Christmas Present||Scrooge", "Bob Cratchit||Tiny Tim", "Scrooge||Tiny Tim"]),
    "carol_hard_004": (["Tiny Tim", "Scrooge", "Bob Cratchit", "Ghost of Christmas Present"], ["Bob Cratchit||Tiny Tim", "Scrooge||Tiny Tim", "Ghost of Christmas Present||Tiny Tim"]),
    "carol_hard_005": (["Scrooge", "Jacob Marley", "Mankind", "Ignorance", "Want", "Ghost of Christmas Present", "Charity Collectors", "The Poor"], ["Charity Collectors||Scrooge", "Jacob Marley||Mankind", "Ghost of Christmas Present||Ignorance", "Ghost of Christmas Present||Want", "Ignorance||Mankind", "Mankind||Want"]),
    "carol_hard_006": (["Scrooge", "Businessmen", "Old Joe", "Charwoman", "Bob Cratchit", "Tiny Tim", "Ghost of Christmas Yet to Come", "Scrooge's Grave"], ["Businessmen||Scrooge", "Charwoman||Old Joe", "Bob Cratchit||Tiny Tim", "Ghost of Christmas Yet to Come||Scrooge", "Scrooge||Scrooge's Grave"]),
    "carol_hard_007": (["Bob Cratchit", "Mrs. Cratchit", "Scrooge", "Tiny Tim"], ["Bob Cratchit||Mrs. Cratchit", "Bob Cratchit||Tiny Tim", "Bob Cratchit||Scrooge"]),
    "carol_hard_008": (["Scrooge", "Jacob Marley", "Mankind", "Fezziwig", "Bob Cratchit"], ["Jacob Marley||Scrooge", "Jacob Marley||Mankind", "Fezziwig||Scrooge", "Bob Cratchit||Scrooge"]),
    "carol_hard_009": (["Scrooge", "Ghost of Christmas Past", "Belle", "Tiny Tim", "Bob Cratchit"], ["Ghost of Christmas Past||Scrooge", "Belle||Scrooge", "Bob Cratchit||Tiny Tim", "Scrooge||Tiny Tim"]),
    "carol_hard_010": (["Scrooge", "Bob Cratchit", "Tiny Tim", "Fred", "Prize Turkey", "Charity Collector"], ["Prize Turkey||Scrooge", "Bob Cratchit||Scrooge", "Fred||Scrooge", "Scrooge||Tiny Tim", "Charity Collector||Scrooge"]),
}


def split_sentences(source: str) -> list[dict]:
    sentences = []
    for match in re.finditer(r".*?(?:[.!?](?=\s)|\Z)", source, re.DOTALL):
        raw = match.group()
        if not raw.strip():
            continue
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        start = match.start() + leading
        end = match.end() - trailing
        text = source[start:end]
        sentences.append({"start": start, "end": end, "text": text})
    return sentences


def normalize_relation(relation: str) -> str:
    parts = [part.strip() for part in relation.split("||")]
    if len(parts) != 2:
        raise ValueError(f"invalid relation: {relation}")
    return "||".join(sorted(parts))


def build(source_path: Path, canonical_template_path: Path) -> dict:
    source = source_path.read_text(encoding="utf-8")
    sentences = split_sentences(source)
    canonical_template = json.loads(canonical_template_path.read_text(encoding="utf-8"))
    questions = canonical_template.get("questions", [])
    question_ids = {question["id"] for question in questions}

    if question_ids != set(EVIDENCE_RANGES):
        missing = sorted(question_ids - set(EVIDENCE_RANGES))
        extra = sorted(set(EVIDENCE_RANGES) - question_ids)
        raise ValueError(f"evidence coverage mismatch; missing={missing}, extra={extra}")
    if question_ids != set(GRAPH_GOLD):
        missing = sorted(question_ids - set(GRAPH_GOLD))
        extra = sorted(set(GRAPH_GOLD) - question_ids)
        raise ValueError(f"graph gold coverage mismatch; missing={missing}, extra={extra}")

    canonical_questions = []
    for question in questions:
        question_id = question["id"]
        answer_point_count = len(question.get("gold_answer_points", []))
        evidence_spans = []
        covered_answer_points = set()

        for evidence_index, (sentence_start, sentence_end, supports) in enumerate(
            EVIDENCE_RANGES[question_id], start=1
        ):
            if sentence_start < 1 or sentence_end > len(sentences) or sentence_start > sentence_end:
                raise ValueError(f"invalid sentence range for {question_id}: {sentence_start}-{sentence_end}")
            if any(index < 1 or index > answer_point_count for index in supports):
                raise ValueError(f"invalid support index for {question_id}: {supports}")

            first = sentences[sentence_start - 1]
            last = sentences[sentence_end - 1]
            span_text = source[first["start"]:last["end"]]
            covered_answer_points.update(supports)
            evidence_spans.append({
                "evidence_id": f"{question_id}_e{evidence_index:02d}",
                "sentence_start": sentence_start,
                "sentence_end": sentence_end,
                "char_start": first["start"],
                "char_end": last["end"],
                "supports_answer_points": supports,
                "text_sha256": hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                "text_preview": re.sub(r"\s+", " ", span_text).strip()[:240],
            })

        expected_supports = set(range(1, answer_point_count + 1))
        if covered_answer_points != expected_supports:
            raise ValueError(
                f"answer-point coverage mismatch for {question_id}: "
                f"covered={sorted(covered_answer_points)}, expected={sorted(expected_supports)}"
            )

        gold_entities, gold_relations = GRAPH_GOLD[question_id]
        canonical_questions.append({
            "id": question_id,
            "difficulty": question.get("difficulty"),
            "question_type": question.get("question_type"),
            "question": question["question"],
            "gold_entities": gold_entities,
            "gold_relations": sorted({normalize_relation(relation) for relation in gold_relations}),
            "gold_answer_points": question.get("gold_answer_points", []),
            "expected_keyword_groups": question.get("expected_keyword_groups", []),
            "gold_evidence_spans": evidence_spans,
        })

    return {
        "schema_version": "1.0",
        "dataset": "A Christmas Carol canonical retrieval gold",
        "source": {
            "path": str(source_path.relative_to(PROJECT_ROOT)),
            "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "character_count": len(source),
            "sentence_count": len(sentences),
        },
        "segmentation": {
            "sentence_ids_are_one_based": True,
            "sentence_end_is_inclusive": True,
            "char_offsets_are_zero_based_half_open": True,
            "rule": "Shortest text ending in ., !, or ? followed by whitespace; final residual text is one sentence; surrounding whitespace is excluded.",
        },
        "annotation": {
            "gold_unit": "source evidence span",
            "chunk_independent": True,
            "evidence_policy": "Each answer point must be supported by at least one source span. Spans identify the relevant scene rather than a particular chunk boundary.",
            "graph_label_policy": "Entity and relation names describe canonical story facts and do not copy the output of any extraction run.",
        },
        "entity_aliases": {
            "Ebenezer Scrooge": "Scrooge",
            "Marley": "Jacob Marley",
            "Marley's Ghost": "Jacob Marley",
            "The Chain": "Marley's Chain",
            "Chain": "Marley's Chain",
            "The Ghost of Christmas Yet to Come": "Ghost of Christmas Yet to Come",
            "Ghost of Christmas Future": "Ghost of Christmas Yet to Come",
            "The Prize Turkey": "Prize Turkey",
            "The Grave": "Scrooge's Grave",
            "Master of the House": "Belle's Husband",
            "Mankind's Welfare": "Mankind",
        },
        "questions": canonical_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk-independent canonical gold for A Christmas Carol.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Existing canonical file supplying question text and answer points.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    canonical = build(args.source.resolve(), args.template.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(canonical, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(canonical['questions'])} questions to {args.output}")


if __name__ == "__main__":
    main()
