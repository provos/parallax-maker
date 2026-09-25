"""Write the combined JSON Schema of the public API models to a file.

The Svelte frontend generates its TypeScript types from this file (see the
``gen:api-types`` script planned in ``docs/svelte-migration/ARCHITECTURE.md``).

Usage::

    python -m parallax_maker.api.schema_export OUT.json
"""

from __future__ import annotations

import sys

from .schemas import dump_schema


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(
            "usage: python -m parallax_maker.api.schema_export OUT.json",
            file=sys.stderr,
        )
        return 2
    dump_schema(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
