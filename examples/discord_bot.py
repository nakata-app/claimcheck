"""Discord bot: verify a claim against your team's documents.

Mirror of `slack_bot.py` for Discord. Server admins post a slash command
(`/factcheck`) and the bot replies with a trust score + flagged claims.

Uses the official `discord.py` library. The verification logic is the
same as the Slack flavour — only the SDK + event hooks change.

Install:

    pip install -e ../adaptmem ../halluguard -e .
    pip install "discord.py>=2.4"

Run:

    export DISCORD_BOT_TOKEN=...   # from https://discord.com/developers/applications
    python examples/discord_bot.py
"""
from __future__ import annotations

import os
from typing import Any

from claimcheck import Pipeline


# ---- One-time setup: build the pipeline from your team's docs --------------

def build_pipeline() -> Pipeline:
    """Replace these documents with your real team / community knowledge base.

    Common shapes that fit straight into this list:
    - exported Notion / Confluence / Google Docs
    - your community wiki dump
    - product FAQ entries
    - moderation rules (so the bot can verify "is this allowed by the rules?")
    """
    documents = [
        "Server rule #1: be kind. No personal attacks; warnings escalate to mute then ban.",
        "Server rule #2: NSFW content only in the #adult channel; tagged posts elsewhere are removed.",
        "Self-promotion is allowed in #showcase only. Drop-and-leave links anywhere else are removed.",
        "Verified contributors get a green name and access to #verified-only. Apply via /verify.",
        "Bot commands: /factcheck verifies a claim, /search finds prior discussion, /report flags a message.",
    ]
    labelled = [
        {"query": "What's the rule on self-promotion?", "relevant_ids": ["doc2"]},
        {"query": "How do I become verified?", "relevant_ids": ["doc3"]},
        {"query": "Where can NSFW go?", "relevant_ids": ["doc1"]},
    ]
    return Pipeline.from_corpus(documents, labelled, train=True, enable_nli=True, device="cpu")


# ---- Discord bot wiring --------------------------------------------------

def make_bot(pipeline: Pipeline) -> Any:
    try:
        import discord
        from discord import app_commands
    except ImportError as e:
        raise SystemExit(
            'Discord example needs `discord.py>=2.4`. Run `pip install "discord.py>=2.4"`.'
        ) from e

    intents = discord.Intents.default()
    intents.message_content = True
    bot = discord.Client(intents=intents)
    tree = app_commands.CommandTree(bot)

    @tree.command(name="factcheck", description="Verify a claim against the server's documented rules.")
    @app_commands.describe(claim="The claim or message to verify")
    async def factcheck(interaction: Any, claim: str) -> None:
        verdict = pipeline.check(claim)
        if verdict.trust_score >= 0.7:
            emoji = "✓"
        elif verdict.trust_score >= 0.4:
            emoji = "!"
        else:
            emoji = "✗"
        embed = discord.Embed(
            title=f"{emoji} trust = {verdict.trust_score:.2f}",
            description=f"Claim: `{claim}`",
            color=0x2ECC71 if verdict.trust_score >= 0.7 else (0xF39C12 if verdict.trust_score >= 0.4 else 0xE74C3C),
        )
        if verdict.flagged_claims:
            embed.add_field(
                name="Flagged",
                value="\n".join(f"• {c}" for c in verdict.flagged_claims[:5]),
                inline=False,
            )
        if verdict.supported_claims:
            embed.add_field(
                name="Supported",
                value="\n".join(f"• {c}" for c in verdict.supported_claims[:3]),
                inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @bot.event
    async def on_ready() -> None:
        await tree.sync()
        print(f"logged in as {bot.user}")

    return bot


def main() -> None:
    pipeline = build_pipeline()
    bot = make_bot(pipeline)
    bot.run(os.environ["DISCORD_BOT_TOKEN"])


if __name__ == "__main__":
    main()
