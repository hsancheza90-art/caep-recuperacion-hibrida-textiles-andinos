from __future__ import annotations

from pathlib import Path

import pandas as pd


SOURCES = {
    "MET_inventario": Path(
        "data/source/met/met_inventario_base.csv"
    ),
    "CMA_candidatos": Path(
        "data/source/cma/cma_andes_textiles_candidates.csv"
    ),
}


def main() -> None:
    for name, path in SOURCES.items():
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )

        print("\n" + "=" * 88)
        print(name)
        print("=" * 88)
        print(f"Filas: {len(frame)}")
        print(f"Columnas: {len(frame.columns)}")

        for index, column in enumerate(
            frame.columns,
            start=1,
        ):
            values = (
                frame[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            non_empty = values.ne("").sum()
            coverage = (
                non_empty / len(frame)
                if len(frame)
                else 0.0
            )

            print(
                f"{index:02d}. {column}: "
                f"{non_empty}/{len(frame)} "
                f"({coverage:.1%})"
            )


if __name__ == "__main__":
    main()