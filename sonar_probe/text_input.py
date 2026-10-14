"""TSV reader for text extraction."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from sonar_probe.errors import BadSpecError
from sonar_probe.types import Lang, TextItem


def read_text_tsv(path: Path, default_lang: Lang | None = None) -> Iterator[TextItem]:
    with Path(path).open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or "text_id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise BadSpecError(f"{path}: must have columns 'text_id' and 'text'")
        has_lang_col = "lang" in reader.fieldnames
        for row in reader:
            lang = (row.get("lang") if has_lang_col else None) or default_lang
            if not lang:
                raise BadSpecError(
                    f"{path}:{reader.line_num}: no lang column and no default --lang"
                )
            yield TextItem(text_id=row["text_id"], text=row["text"], lang=lang)
