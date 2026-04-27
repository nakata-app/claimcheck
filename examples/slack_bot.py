"""Slack bot: verify a teammate's AI-generated message before they ship it.

Pattern: someone posts an AI-generated answer to a channel; the bot
re-checks it against your team's documents (Notion exports, internal
wiki dumps, Confluence) and replies with a trust score + flagged
claims. Reduces "Claude said X but our docs actually say Y" embarrassment.

Slack-side wiring uses `slack_bolt` (the official Slack SDK for
Python). The verify path stays portable — swap slack_bolt for any
chat platform's SDK.

Install:

    pip install -e ../adaptmem ../halluguard -e .
    pip install slack_bolt

Run (local dev with Socket Mode — no public URL needed):

    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    python examples/slack_bot.py
"""
from __future__ import annotations

import os
from typing import Any

from claimcheck import Pipeline


# ---- One-time setup: build the pipeline from your team's docs --------------

def build_pipeline() -> Pipeline:
    """Load your team's reference corpus.

    In production this would read from a Notion export / S3 dump /
    Postgres dump. The shape needed is just `list[str]` of document
    bodies + `list[{query, relevant_ids}]` for the supervised pairs.
    """
    documents = [
        # Replace these with your real team docs.
        "Engineering on-call rotates weekly. Each shift starts Monday 09:00 TR time.",
        "Pull requests need at least one approval; security-sensitive changes need two.",
        "We use PostgreSQL 15 in production; the staging cluster runs 16 for testing.",
        "Refunds over $500 require manager approval. Smaller refunds: any agent.",
        "Customer support SLA: first response within 2 business hours.",
    ]
    labelled = [
        {"query": "When does on-call start?", "relevant_ids": ["doc0"]},
        {"query": "How many PR approvals do we need?", "relevant_ids": ["doc1"]},
        {"query": "What Postgres version do we run?", "relevant_ids": ["doc2"]},
        {"query": "When does a refund need manager approval?", "relevant_ids": ["doc3"]},
    ]
    return Pipeline.from_corpus(documents, labelled, train=True, enable_nli=True, device="cpu")


# ---- Slack handlers ------------------------------------------------------

def make_app(pipeline: Pipeline) -> Any:
    try:
        from slack_bolt import App
    except ImportError as e:
        raise SystemExit(
            "Slack bot example needs slack_bolt. Run `pip install slack_bolt`."
        ) from e

    app = App(token=os.environ.get("SLACK_BOT_TOKEN", "xoxb-placeholder"))

    @app.command("/factcheck")
    def handle_factcheck(ack: Any, command: Any, respond: Any) -> None:
        """Slash command: /factcheck <claim> — runs claimcheck and posts the verdict."""
        ack()
        claim = command.get("text", "").strip()
        if not claim:
            respond("Usage: `/factcheck <claim>` — e.g. `/factcheck Postgres 16 is in production`")
            return

        verdict = pipeline.check(claim)
        emoji = "✅" if verdict.trust_score >= 0.7 else ("⚠️" if verdict.trust_score >= 0.4 else "❌")
        lines = [f"{emoji} *trust_score* `{verdict.trust_score:.2f}`"]
        if verdict.flagged_claims:
            lines.append("\n*Flagged in our docs:*")
            for c in verdict.flagged_claims:
                lines.append(f"• {c}")
        if verdict.supported_claims:
            lines.append("\n*Supported by our docs:*")
            for c in verdict.supported_claims[:3]:
                lines.append(f"• {c}")
        respond("\n".join(lines))

    @app.event("app_mention")
    def handle_mention(event: Any, say: Any) -> None:
        """When mentioned, treat the message text as a claim and verify."""
        text = event.get("text", "")
        # Strip the bot mention prefix — Slack inserts `<@U12345>`.
        claim = text.split(">", 1)[-1].strip() if ">" in text else text
        if not claim:
            return
        verdict = pipeline.check(claim)
        say(
            text=(
                f"trust={verdict.trust_score:.2f}, "
                f"{len(verdict.flagged_claims)} flagged, "
                f"{len(verdict.supported_claims)} supported"
            ),
            thread_ts=event.get("ts"),
        )

    return app


def main() -> None:
    try:
        from slack_bolt.adapter.socket_mode import SocketModeHandler
    except ImportError as e:
        raise SystemExit("`pip install slack_bolt`") from e

    pipeline = build_pipeline()
    app = make_app(pipeline)
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()


if __name__ == "__main__":
    main()
