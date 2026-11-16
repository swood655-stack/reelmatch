"""
recommender.py
--------------
The recommendation engine. Three approaches:

1. Content-based  - TF-IDF vectors of each movie's text "soup" (genres,
                    keywords, director, cast, plot), compared with cosine
                    similarity. "Movies made of similar ingredients."
2. Sentiment-based - compares the emotional tone of plot summaries
                    (VADER compound, positive and negative scores).
                    "Movies that feel similar."
3. Hybrid (default) - a weighted blend of the two, plus a small boost for
                    quality (a vote-weighted rating), so a well-matched but
                    badly reviewed movie doesn't top the list.

Every recommendation comes with plain-English reasons ("Same director:
Christopher Nolan", "Shared themes: time travel, heist").
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hybrid weights. Content matters most; tone refines; quality breaks ties.
WEIGHTS = {"content": 0.65, "sentiment": 0.25, "quality": 0.10}

METHODS = {
    "Hybrid (recommended)": "hybrid",
    "Content-based (story, cast & crew)": "content",
    "Mood-based (sentiment of the plot)": "sentiment",
}


def weighted_rating(df, m=None):
    """IMDb-style weighted rating: pulls movies with few votes toward the
    average, so a 9.0 from 12 votes doesn't beat an 8.4 from 20,000."""
    v, r = df["vote_count"].astype(float), df["vote_average"].astype(float)
    c = r.mean() if len(r) else 6.5
    m = m if m is not None else max(v.quantile(0.6) if len(v) else 100, 50)
    return (v / (v + m)) * r + (m / (v + m)) * c


class Recommender:
    def __init__(self, pool):
        """Fit the TF-IDF model on the movie pool once (it's cached by the app)."""
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                          min_df=1, max_df=0.6, sublinear_tf=True)
        self.vectorizer.fit(pool["soup"])
        # Typical number of votes in the pool, used for the quality score.
        self.m = max(float(pool["vote_count"].quantile(0.6)), 50.0)

    # -- individual scores -------------------------------------------------
    def content_scores(self, seed_soup, candidates):
        seed_vec = self.vectorizer.transform([seed_soup])
        cand_vec = self.vectorizer.transform(candidates["soup"])
        return cosine_similarity(seed_vec, cand_vec).ravel()

    @staticmethod
    def sentiment_similarity(target, candidates):
        """1 = identical tone, 0 = opposite. `target` holds compound, pos, neg."""
        cols = ["compound", "pos", "neg"]
        t = np.array([target[c] for c in cols], dtype=float)
        x = candidates[cols].to_numpy(dtype=float)
        # compound spans -1..1 (range 2), pos/neg span 0..1, so scale compound by 0.5
        scale = np.array([0.5, 1.0, 1.0])
        dist = np.sqrt((((x - t) * scale) ** 2).sum(axis=1))
        return 1 - dist / np.sqrt(3)

    def quality_scores(self, candidates):
        wr = weighted_rating(candidates, self.m)
        return (wr / 10).to_numpy()

    def text_scores(self, text, candidates):
        """How well each movie matches free-text keywords (for the Ask page)."""
        if not text.strip():
            return np.zeros(len(candidates))
        return self.content_scores(text.lower(), candidates)

    # -- main entry point ----------------------------------------------------
    def recommend(self, seed, candidates, method="hybrid", n=10):
        """Rank `candidates` by similarity to the `seed` movie (a row/Series).

        Returns the top-n candidates with score columns and a 'reasons' list.
        """
        cands = candidates[candidates["id"] != seed["id"]].copy()
        if cands.empty:
            return cands

        cands["content_score"] = self.content_scores(seed["soup"], cands)
        cands["sentiment_score"] = self.sentiment_similarity(seed, cands)
        cands["quality_score"] = self.quality_scores(cands)

        if method == "content":
            cands["score"] = cands["content_score"]
        elif method == "sentiment":
            # Pure tone matching would surface random movies with the same
            # mood, so keep a light content signal (shared genres) in the mix.
            genre_overlap = cands["genres"].apply(lambda g: _jaccard(g, seed["genres"]))
            cands["score"] = 0.75 * cands["sentiment_score"] + 0.25 * genre_overlap
        else:
            # Content scores are usually 0-0.4, so rescale to 0-1 before blending.
            c = cands["content_score"]
            c_norm = c / c.max() if c.max() > 0 else c
            cands["score"] = (WEIGHTS["content"] * c_norm
                              + WEIGHTS["sentiment"] * cands["sentiment_score"]
                              + WEIGHTS["quality"] * cands["quality_score"])

        top = cands.sort_values("score", ascending=False).head(n).copy()
        top["reasons"] = top.apply(lambda r: explain(seed, r), axis=1)
        top["match"] = _to_match_percent(top["score"], method)
        return top

    def rank_for_query(self, candidates, keywords="", mood=None, n=12):
        """Rank movies for a natural-language request: keyword match + mood
        match + quality. Used when there is no seed movie."""
        cands = candidates.copy()
        if cands.empty:
            return cands
        text = self.text_scores(keywords, cands)
        cands["text_score"] = text / text.max() if text.max() > 0 else text
        if mood:
            target = {"compound": 0.8, "pos": 0.3, "neg": 0.0} if mood == "positive" \
                else {"compound": -0.8, "pos": 0.0, "neg": 0.3}
            cands["sentiment_score"] = self.sentiment_similarity(target, cands)
        else:
            cands["sentiment_score"] = 0.0
        cands["quality_score"] = self.quality_scores(cands)

        w_text = 0.5 if keywords.strip() else 0.0
        w_mood = 0.35 if mood else 0.0
        w_quality = 1.0 - w_text - w_mood
        cands["score"] = (w_text * cands["text_score"] + w_mood * cands["sentiment_score"]
                          + w_quality * cands["quality_score"])
        top = cands.sort_values("score", ascending=False).head(n).copy()
        top["reasons"] = top.apply(lambda r: explain_query(r, keywords, mood), axis=1)
        top["match"] = (top["score"] * 100).clip(0, 99).round().astype(int)
        return top


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------
def _jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def _to_match_percent(scores, method):
    """Turn raw scores into a friendly 0-99% 'match' number."""
    s = scores.astype(float)
    if method == "content":
        s = s / s.max() if s.max() > 0 else s
    return (s * 100).clip(0, 99).round().astype(int)


def explain(seed, cand):
    """Plain-English reasons why `cand` was recommended for `seed`."""
    reasons = []
    directors = set(seed["director"]) & set(cand["director"])
    if directors:
        reasons.append("Same director: " + ", ".join(sorted(directors)))
    cast = [c for c in cand["cast"] if c in set(seed["cast"])]
    if cast:
        reasons.append("Shared cast: " + ", ".join(cast[:3]))
    genres = [g for g in cand["genres"] if g in set(seed["genres"])]
    if genres:
        reasons.append("Both are " + " / ".join(genres[:3]))
    keywords = [k for k in cand["keywords"] if k in set(seed["keywords"])]
    if keywords:
        reasons.append("Shared themes: " + ", ".join(keywords[:4]))
    if seed["mood"] == cand["mood"]:
        reasons.append(f"Similar tone: both read as {cand['mood'].lower()}")
    if not reasons:
        reasons.append("Similar plot language and overall feel")
    return reasons


def explain_query(cand, keywords, mood):
    reasons = []
    if keywords:
        kw = set(keywords.split())
        hits = [k for k in cand["keywords"] if set(k.lower().split()) & kw]
        hits += [w for w in kw if w in str(cand["overview"]).lower() and w not in " ".join(hits)]
        if hits:
            reasons.append("Matches: " + ", ".join(dict.fromkeys(hits[:4])))
    if mood:
        reasons.append(f"Tone: {cand['mood'].lower()} (score {cand['compound']:+.2f})")
    reasons.append(f"Rated {cand['vote_average']:.1f} by {int(cand['vote_count']):,} people")
    return reasons


