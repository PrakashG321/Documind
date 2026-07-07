"""
Retrieval evaluation for DocuMind.

Two layers of retrieval quality, both objective (no human judgment):

1. FILE-LEVEL — "Hit Rate @ k": did the correct doc file appear in the top-k?
   Coarse: a file can rank #1 on its intro chunk while the chunk that actually
   contains the answer is buried far lower.

2. ANSWER-LEVEL — "Answer-in-Context @ k": does the concatenated top-k context
   we would send to the LLM actually contain the answer text (a distinctive
   marker string)? This is the metric that matters — the LLM can only answer
   from what's in the context — and it exposes the buried-chunk problem.

Each question carries a `gold_source` (the file) and an `answer_marker`
(a distinctive string that must appear in context, e.g. `--invokable`).
"""

from query import build_pipeline

eval_set = [
    {"question": "How do I create a single-action (invokable) controller?", "gold_source": "controllers.md", "answer_marker": "--invokable"},
    {"question": "How do I register middleware in Laravel?", "gold_source": "middleware.md", "answer_marker": "make:middleware"},
    {"question": "How do I validate an incoming form request?", "gold_source": "validation.md", "answer_marker": "->validate("},
    {"question": "How do I define a route with a required parameter?", "gold_source": "routing.md", "answer_marker": "{id}"},
    {"question": "How do I define a one-to-many relationship in Eloquent?", "gold_source": "eloquent-relationships.md", "answer_marker": "hasMany"},
    {"question": "How do I create a database migration?", "gold_source": "migrations.md", "answer_marker": "Schema::create"},
    {"question": "How do I send an email using a mailable?", "gold_source": "mail.md", "answer_marker": "make:mail"},
    {"question": "How do I display a variable in a Blade template?", "gold_source": "blade.md", "answer_marker": "{{ $"},
    {"question": "How does Laravel protect against CSRF attacks?", "gold_source": "csrf.md", "answer_marker": "@csrf"},
    {"question": "How do I add a where clause using the query builder?", "gold_source": "queries.md", "answer_marker": "->where("},
    {"question": "How do I store an item in the cache?", "gold_source": "cache.md", "answer_marker": "Cache::put"},
    {"question": "How do I schedule an artisan command to run daily?", "gold_source": "scheduling.md", "answer_marker": "->daily("},
]

# Retrieve this many chunks per question, then derive both metrics for smaller k.
MAX_K = 20
K_VALUES = [3, 5, 10, 15, 20]


def evaluate(store, item):
    """Retrieve top-MAX_K once; return (gold_rank, marker_rank).

    gold_rank   = 1-based rank where the gold file first appears (or None)
    marker_rank = 1-based rank of the first chunk containing the answer marker;
                  answer-in-context @ k is then simply (marker_rank <= k).
    """
    results = store.similarity_search(item["question"], k=MAX_K)
    gold_rank = None
    marker_rank = None
    for i, doc in enumerate(results, start=1):
        if gold_rank is None and doc.metadata["source"] == item["gold_source"]:
            gold_rank = i
        if marker_rank is None and item["answer_marker"] in doc.page_content:
            marker_rank = i
    return gold_rank, marker_rank


def main():
    store, _ = build_pipeline()  # only the retriever is needed here

    gold_ranks, marker_ranks = [], []
    print(f"{'file':>4} {'ans':>4}  {'gold source':<26}  question")
    print("-" * 84)
    for item in eval_set:
        g, m = evaluate(store, item)
        gold_ranks.append(g)
        marker_ranks.append(m)
        gs = str(g) if g is not None else "MISS"
        ms = str(m) if m is not None else "MISS"
        print(f"{gs:>4} {ms:>4}  {item['gold_source']:<26}  {item['question']}")

    total = len(eval_set)

    def rate_at_k(ranks, k):
        return sum(1 for r in ranks if r is not None and r <= k)

    print("\n=== File-level Hit Rate @ k (did the right FILE appear?) ===")
    for k in K_VALUES:
        h = rate_at_k(gold_ranks, k)
        print(f"  @{k:<3}: {h}/{total} = {h / total:.0%}")

    print("\n=== Answer-in-Context @ k (is the ANSWER text in the context?) ===")
    for k in K_VALUES:
        h = rate_at_k(marker_ranks, k)
        print(f"  @{k:<3}: {h}/{total} = {h / total:.0%}")

    mrr = sum((1 / r) for r in gold_ranks if r is not None) / total
    print(f"\nFile-level MRR (over top-{MAX_K}): {mrr:.3f}")


if __name__ == "__main__":
    main()
