from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/processed/paper_corpus_v1.parquet"
)
OUTPUT_PATH = Path(
    "outputs/reports/corpus_coverage_v1.csv"
)


def main() -> None:
    frame = pd.read_parquet(INPUT_PATH)

    fields = [
        "title",
        "description",
        "culture",
        "period",
        "object_type",
        "material",
        "technique",
        "provenance",
        "classification",
        "license",
        "image_url",
        "object_url",
    ]

    rows: list[dict[str, object]] = []

    for museum in ["ALL", "MET", "CMA"]:
        subset = (
            frame
            if museum == "ALL"
            else frame.loc[frame["museum"] == museum]
        )

        for field in fields:
            values = (
                subset[field]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            non_empty = values.ne("").sum()
            total = len(subset)

            rows.append(
                {
                    "museum": museum,
                    "field": field,
                    "non_empty": int(non_empty),
                    "total": total,
                    "coverage": (
                        non_empty / total
                        if total
                        else 0.0
                    ),
                }
            )

    report = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(report.to_string(index=False))
    print(f"\nReporte: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()