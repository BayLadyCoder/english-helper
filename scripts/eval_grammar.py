"""Run the grammar checker on tricky sentences against the local model and score it.

Each case lists phrases the corrected text must contain and phrases it must not.
Run with: uv run python scripts/eval_grammar.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import check_grammar  # noqa: E402

CASES = [
    (
        "The marketing team have spent weeks developing the new campaign.",
        ["team has spent"],
        ["have spent"],
    ),
    ("The committee are meeting tomorrow.", ["committee is meeting"], ["are meeting"]),
    ("Each of the students have a laptop.", ["students has"], ["have a laptop"]),
    ("The list of items are on the desk.", ["items is"], ["are on"]),
    ("Neither the manager nor the employees was ready.", ["employees were"], ["was ready"]),
    ("There is many reasons to stay.", ["There are many"], ["There is many"]),
    ("She don't like coffee.", ["doesn't"], ["don't"]),
    ("I have went there before.", ["have gone"], ["have went"]),
    ("If I would have known, I would have came earlier.", ["If I had known", "have come"], ["have came"]),
    ("Me and him went to the meeting.", ["He and I"], ["Me and him"]),
    ("Between you and I, the plan won't work.", ["you and me", "won't work"], ["you and I"]),
    ("Less people came than we expected.", ["Fewer people"], ["Less people"]),
    ("The report, along with the slides, were sent yesterday.", ["slides, was sent"], ["were sent"]),
    (
        "Yesterday I go to office. The marketing team have spent weeks developing the new "
        "campaign and the results was great.",
        ["I went", "team has spent", "results were"],
        ["have spent", "results was"],
    ),
    (
        "The marketing team have spent weeks developing the new campaign, and they is very "
        "proud of it.",
        ["team has spent", "they are"],
        ["have spent", "they is"],
    ),
    (
        "Our company have hired three new engineers, but the group are still too small "
        "for the project.",
        ["company has hired", "group is"],
        ["have hired", "group are"],
    ),
    ("The new campaign launches next week.", ["The new campaign launches next week."], []),
    ("Our team has finished the project ahead of schedule.", ["Our team has finished"], []),
]


def main() -> None:
    """Print each result and the overall pass count."""
    passed = 0
    for text, required, forbidden in CASES:
        corrected = check_grammar(text)["corrected_text"]
        ok = all(p.casefold() in corrected.casefold() for p in required) and not any(
            p.casefold() in corrected.casefold() for p in forbidden
        )
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {text}\n      -> {corrected}")
    print(f"\n{passed}/{len(CASES)} passed")


if __name__ == "__main__":
    main()
