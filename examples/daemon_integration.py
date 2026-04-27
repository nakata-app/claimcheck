"""Pipeline.from_daemon — encoder lives in a long-lived `adaptmem serve`.

Use this when claimcheck + halluguard + a third service would otherwise
each load their own MiniLM. The daemon serves all three.

Two terminals:

  Terminal 1 — daemon:
    pip install "adaptmem[server]"
    adaptmem serve --port 7800 --base-model all-MiniLM-L6-v2

  Terminal 2 — this script:
    pip install -e ../adaptmem ../halluguard -e ".[dev]"
    pip install requests
    python examples/daemon_integration.py
"""
from claimcheck import Pipeline


def main() -> None:
    documents = [
        "Apollo 11 landed on the Moon on July 20, 1969.",
        "Neil Armstrong walked first; Buzz Aldrin followed.",
        "The Apollo program ran from 1961 to 1972 with six successful crewed lunar landings.",
    ]

    # `from_daemon` calls /healthz first, so misconfig fails loudly here.
    pipeline = Pipeline.from_daemon(
        documents=documents,
        daemon_url="http://127.0.0.1:7800",
        enable_nli=True,  # NLI verifier still runs in-process
    )

    cases = [
        ("Apollo 11 landed in 1969 and Armstrong walked first.", "When did Apollo 11 land?", "grounded"),
        ("Apollo 11 landed in 1955 and Yuri Gagarin walked first.", "When did Apollo 11 land?", "fully wrong"),
        ("Apollo 11 landed in 1969. The astronauts also visited Mars on the same trip.", "Apollo 11?", "mixed"),
    ]
    for answer, question, label in cases:
        v = pipeline.check(answer, question=question, profile=True)
        timing = v.timing or {}
        print(
            f"[{label:>11}] trust={v.trust_score:.2f}  "
            f"{timing.get('total_ms', 0):.0f}ms  {answer!r}"
        )
        for c in v.flagged_claims:
            print(f"             flagged: {c}")


if __name__ == "__main__":
    main()
