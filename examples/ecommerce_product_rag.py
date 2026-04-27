"""E-commerce product RAG: verify AI's answer against your product catalog.

Common shape: customer asks a question about your products in natural
language ("yaz için pastel bir elbise var mı?"), an LLM generates a
recommendation, claimcheck makes sure the LLM didn't invent a product
or attribute that doesn't exist in your catalog.

This example uses a tiny synthetic catalog. In production swap it for
your own (Shopify export / Postgres dump / Stripe products / etc.).

Install:

    pip install -e ../adaptmem ../halluguard -e .

Run:

    python examples/ecommerce_product_rag.py
"""
from __future__ import annotations

from claimcheck import Pipeline


# ---- Catalog: replace with your real product data ------------------------

CATALOG = [
    # Each entry is one product, expressed in natural language so
    # claimcheck can match user questions and AI answers against it.
    "SKU-1001: Pastel sarı yazlık elbise, %100 pamuk, midi boy, 36-42 beden, 599₺.",
    "SKU-1002: Lacivert keten gömlek, uzun kollu, slim fit, S/M/L/XL, 449₺.",
    "SKU-1003: Beyaz spor ayakkabı, tabanı kauçuk, deri üst, 36-45 numara, 1299₺.",
    "SKU-1004: Siyah deri ceket, biker stil, omuz vatkalı, S/M/L, 2499₺.",
    "SKU-1005: Krem rengi triko hırka, V yaka, akrilik karışım, tek beden, 379₺.",
    "SKU-1006: Pembe yün kazak, balıkçı yaka, %30 yün, S/M/L, 549₺.",
]

LABELLED = [
    {"query": "yazlık elbise var mı?", "relevant_ids": ["doc0"]},
    {"query": "spor ayakkabı seçenekleri", "relevant_ids": ["doc2"]},
    {"query": "kış için sıcak triko", "relevant_ids": ["doc4", "doc5"]},
    {"query": "deri ürünler", "relevant_ids": ["doc3"]},
]


def main() -> None:
    pipeline = Pipeline.from_corpus(
        CATALOG,
        LABELLED,
        train=True,
        enable_nli=True,
        device="cpu",
    )

    # Three test customer interactions.
    cases = [
        # (customer_question, ai_generated_answer, expected_outcome)
        (
            "Yaz için pastel renkli elbise var mı?",
            "Evet, SKU-1001 pastel sarı yazlık elbisemiz var. %100 pamuk, midi boy, 599₺.",
            "fully grounded",
        ),
        (
            "Çocuk bedeni mavi elbiseniz var mı?",
            # AI hallucinates a product that isn't in the catalog.
            "Evet, SKU-1099 çocuk bedeni mavi elbisemiz var, 299₺ fiyatla.",
            "hallucinated SKU + price",
        ),
        (
            "Spor ayakkabı fiyatı kaç?",
            # AI gets the SKU right but invents a wrong price.
            "Beyaz spor ayakkabımız (SKU-1003) 1299₺. Aynı zamanda 999₺'lik gri model de var.",
            "first claim ok, second invented",
        ),
    ]

    for question, answer, expected in cases:
        v = pipeline.check(answer, question=question)
        if v.trust_score >= 0.7:
            label = "OK"
        elif v.trust_score >= 0.4:
            label = "WARN"
        else:
            label = "BLOCK"
        print(f"[{label:>5}] trust={v.trust_score:.2f}  ({expected})")
        print(f"        Q: {question!r}")
        print(f"        A: {answer!r}")
        for c in v.flagged_claims:
            print(f"  flagged: {c}")
        for c in v.supported_claims:
            print(f"  ok     : {c}")
        print()


if __name__ == "__main__":
    main()
