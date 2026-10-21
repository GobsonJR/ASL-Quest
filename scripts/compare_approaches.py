"""Compare recognizers evaluated by scripts/evaluate_webcam.py on the SAME unseen-room webcam rows.

    venv311\\Scripts\\python.exe scripts\\compare_approaches.py outputs\\evaluation_webcam\\resnet_current.json outputs\\evaluation_webcam\\resnet_webcam.json outputs\\evaluation_webcam\\landmarks.json

Prints one table (accuracy, macro-F1, rejection, accuracy among accepted, stability, latency), the per-letter recall of
every candidate, and the letters where candidates differ most. It refuses to compare runs that used different test rooms,
because that would not be the same test. It does NOT pick a winner for you: read the whole table, especially wrong
confirmations (a recognizer that confidently confirms a wrong letter is worse than one that abstains).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def pct(x: float | None) -> str:
    return "  -  " if x is None else f"{100 * x:5.1f}"


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sys.argv[1:]]
    if len({tuple(r["rooms"]) for r in runs}) > 1 or len({r["split"] for r in runs}) > 1:
        sys.exit("these runs used different rooms/splits, so they are not the same test: " + str([(r["name"], r["rooms"], r["split"]) for r in runs]))
    names = [r["name"] for r in runs]
    print(f"test rooms: {runs[0]['rooms']}   split: {runs[0]['split']}   scored frames: {[r['overall']['n_scored'] for r in runs]}\n")
    width = max(len(n) for n in names) + 2
    print(f"{'':32s}" + "".join(f"{n:>{width}s}" for n in names))

    def row(label: str, getter) -> None:
        cells = []
        for r in runs:
            try:
                cells.append(getter(r))
            except (KeyError, TypeError):
                cells.append("  -  ")
        print(f"{label:32s}" + "".join(f"{c:>{width}s}" for c in cells))

    row("accuracy %", lambda r: pct(r["overall"]["accuracy"]))
    row("macro-F1 % (letters present)", lambda r: pct(r["overall"]["macro_f1_over_present_letters"]))
    row("rejected as uncertain %", lambda r: pct(r["overall"]["reject_rate"]))
    row("accuracy among accepted %", lambda r: pct(r["overall"]["accuracy_among_accepted"]))
    row("hold: confirmed correct %", lambda r: pct(r["stability"]["confirmed_correct"]))
    row("hold: confirmed WRONG %", lambda r: pct(r["stability"]["confirmed_wrong"]))
    row("hold: never confirmed %", lambda r: pct(r["stability"]["never_confirmed"]))
    row("hold: median s to confirm", lambda r: f"{r['stability']['median_seconds_to_confirm']:.1f}")
    row("latency median ms", lambda r: f"{r['latency_ms']['median']:.1f}")

    for title, key in (("hand", "by_hand"), ("background", "by_background"), ("room", "by_room"), ("distance", "by_distance"), ("lighting", "by_lighting")):
        values = sorted({g for r in runs for g in r.get(key, {})})
        if len(values) > 1:
            print(f"\naccuracy % by {title}:")
            for g in values:
                row(f"  {g}", lambda r, g=g: pct(r[key][g]["accuracy"]) + f" (n={r[key][g]['n']})")

    letters = [l for l, v in runs[0]["overall"]["per_letter"].items() if v["support"]]
    print("\nper-letter recall %:")
    print(f"{'letter':8s}{'n':>4s}" + "".join(f"{n:>{width}s}" for n in names))
    for l in letters:
        print(f"{l:8s}{runs[0]['overall']['per_letter'][l]['support']:>4d}" + "".join(f"{pct(r['overall']['per_letter'][l]['recall']):>{width}s}" for r in runs))
    if len(runs) >= 2:
        gaps = sorted(letters, key=lambda l: -(max(r["overall"]["per_letter"][l]["recall"] for r in runs) - min(r["overall"]["per_letter"][l]["recall"] for r in runs)))[:6]
        print("\nletters where the candidates differ most:", gaps)
    print("\nMost common wrong predictions (first run):", {l: v["top_wrong"] for l, v in runs[0]["overall"]["per_letter"].items() if v["top_wrong"]})


if __name__ == "__main__":
    main()
