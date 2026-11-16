"""
nlp.py
------
All natural language processing:

1. build_soup      - combines a movie's text features into one document
                     for TF-IDF (used by content-based filtering).
2. sentiment_scores / mood_label
                   - VADER sentiment analysis of plot summaries. Each movie
                     gets a tone score from -1 (dark) to +1 (uplifting).
                     This powers sentiment-based matching.
3. parse_query     - a rule-based parser that turns a sentence like
                     "feel-good 90s comedies with Tom Hanks under 2 hours"
                     into structured filters, a mood and keywords.

Why VADER: it's built for short, informal text, needs no training data
and runs fast, which suits short plot summaries.
Why rules for the query parser: they are predictable, explainable (we can
show users exactly what we understood) and need no paid AI service.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from filters import FilterSettings

_analyzer = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# 1. Text "soup" for content-based filtering
# ---------------------------------------------------------------------------
def _token(name):
    """'Tom Hanks' -> 'tom_hanks' so a full name counts as one token and
    'Tom Hanks' isn't confused with 'Tom Cruise'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def build_soup(row):
    """Weighted text document for one movie.

    Weights (by repetition): director x3, genres x2, keywords x2, top 3
    cast x1, plot summary x1. Directors and genres say a lot about what a
    movie feels like, so they get more weight than individual plot words.
    """
    parts = []
    parts += [_token(d) for d in row["director"]] * 3
    parts += [_token(g) for g in row["genres"]] * 2
    parts += [_token(k) for k in row["keywords"]] * 2
    parts += [_token(c) for c in list(row["cast"])[:3]]
    parts.append(str(row["overview"]).lower())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 2. Sentiment analysis
# ---------------------------------------------------------------------------
def sentiment_scores(text):
    """VADER scores for a plot summary: compound (-1..1), pos, neg, neu."""
    s = _analyzer.polarity_scores(str(text or ""))
    return {"compound": s["compound"], "pos": s["pos"], "neg": s["neg"], "neu": s["neu"]}


def mood_label(compound):
    if compound >= 0.3:
        return "Uplifting"
    if compound <= -0.3:
        return "Dark"
    return "Mixed"


# ---------------------------------------------------------------------------
# 3. Natural language query parser
# ---------------------------------------------------------------------------
GENRE_SYNONYMS = {
    r"rom-?coms?|romantic comed(?:y|ies)": ["Romance", "Comedy"],
    r"action": ["Action"],
    r"adventures?": ["Adventure"],
    r"animated|animation|cartoons?|anime": ["Animation"],
    r"comed(?:y|ies)|funny|hilarious": ["Comedy"],
    r"crime|heists?|gangsters?|mafia": ["Crime"],
    r"documentar(?:y|ies)": ["Documentary"],
    r"dramas?": ["Drama"],
    r"fantasy": ["Fantasy"],
    r"historical|history|period pieces?": ["History"],
    r"horror|scary": ["Horror"],
    r"musicals?": ["Music"],
    r"myster(?:y|ies)|whodunn?its?": ["Mystery"],
    r"romance|romantic|love stor(?:y|ies)": ["Romance"],
    r"sci-?fi|science fiction": ["Science Fiction"],
    r"thrillers?|suspense": ["Thriller"],
    r"war": ["War"],
    r"westerns?": ["Western"],
    r"family|kids?": ["Family"],
}

MOOD_WORDS = {
    "positive": r"feel-?good|happy|uplifting|heart-?warming|wholesome|light-?hearted|"
                r"cheerful|fun|inspiring|inspirational|upbeat|cozy|cosy|hopeful",
    "negative": r"dark|sad|gritty|depressing|bleak|intense|disturbing|tragic|"
                r"tear-?jerkers?|haunting|grim",
}

LANGUAGE_WORDS = {
    "french": "fr", "korean": "ko", "japanese": "ja", "spanish": "es", "hindi": "hi",
    "bollywood": "hi", "italian": "it", "german": "de", "chinese": "zh",
    "mandarin": "zh", "cantonese": "cn", "swedish": "sv", "danish": "da",
    "portuguese": "pt", "russian": "ru", "turkish": "tr", "thai": "th",
}

FILLER = {
    "movie", "movies", "film", "films", "show", "want", "watch", "watching",
    "recommend", "recommendation", "recommendations", "suggest", "find", "give",
    "good", "great", "something", "anything", "stuff", "looking", "like", "need",
    "id", "im", "tonight", "today", "pick", "picks", "please", "really", "ones",
    "minutes", "mins", "hours", "hour", "rated", "rating", "stars", "year", "years",
}

NEGATION_BEFORE = re.compile(r"(?:\bno|\bnot|\bwithout|\bexcept|\bnothing|\bzero|\bskip)\s+(?:\w+\s+)?$")

# Names start with a capital letter. Trigger words are case-insensitive.
_NAME = r"([A-Z][\w'.\-]*(?:\s+(?:[A-Z][\w'.\-]*|de|van|von|del|da|di|le|la|du))*)"
PERSON_RE = re.compile(r"(?i:\b(directed by|by director|starring|featuring|with|by))\s+" + _NAME)
PERSON_LOWER_RE = re.compile(r"\b(directed by|starring|featuring)\s+([a-z][\w'\-]+\s+[a-z][\w'\-]+)", re.I)
SEED_RE = re.compile(
    r"(?i:\b(?:movies?|films?|something|anything|stuff)\s+like|similar\s+to|in the style of)\s+"
    r"(.+?)(?=\s+(?:but|with|starring|directed|from|in the|under|over|that|which|rated)\b|[,.!?;]|$)"
)


@dataclass
class ParsedQuery:
    filters: FilterSettings = field(default_factory=FilterSettings)
    mood: str | None = None                 # "positive" / "negative" / None
    seed_title: str | None = None           # "movies like Inception"
    people: list = field(default_factory=list)  # [(name, "actor"/"director")]
    keywords: str = ""                      # leftover words for text matching
    understood: list = field(default_factory=list)  # labels shown to the user


def _minutes(num, unit):
    num = float(num)
    return int(round(num * 60)) if unit.lower().startswith("h") else int(round(num))


def _decade_start(token):
    """'90' -> 1990, '1990' -> 1990, '00' -> 2000, '10' -> 2010."""
    n = int(token)
    if n >= 1000:
        return n
    return 2000 + n if n <= 20 else 1900 + n


def parse_query(text):
    """Turn a free-text request into a ParsedQuery."""
    q = ParsedQuery()
    f = q.filters
    original = (text or "").strip()
    low = original.lower()
    used = []  # character spans already explained, removed from keywords later

    def mark(m, group=0):
        used.append(m.span(group))

    def find(pattern):
        """First match of `pattern` that isn't part of an already-used phrase
        (so 'The Dark Knight' in a title doesn't count as a 'dark' mood)."""
        for m in re.finditer(pattern, low):
            if not any(m.start() < e and s < m.end() for s, e in used):
                return m
        return None

    # --- "movies like X" -------------------------------------------------
    m = SEED_RE.search(original)
    seed_end = 0
    if m:
        seed_end = m.end()
        q.seed_title = m.group(1).strip(" '\"")
        mark(m)
        q.understood.append(f"🎯 similar to “{q.seed_title}”")

    # --- people ------------------------------------------------------------
    for m in PERSON_RE.finditer(original):
        if m.start() < seed_end:
            continue  # part of the movie title, e.g. "movies like Knives Out"
        trigger, name = m.group(1).lower(), m.group(2).strip()
        if name.lower() in LANGUAGE_WORDS or len(name) < 3:
            continue
        role = "director" if "direct" in trigger else "actor"
        q.people.append((name, role))
        mark(m)
    if not q.people:
        for m in PERSON_LOWER_RE.finditer(original):
            role = "director" if "direct" in m.group(1).lower() else "actor"
            q.people.append((m.group(2).title(), role))
            mark(m)
    for name, role in q.people:
        q.understood.append(f"🎬 {role}: {name}")

    # --- years and decades -------------------------------------------------
    year_found = False
    for pattern in (r"\b(?:between|from)\s+((?:19|20)\d{2})\s+(?:and|to|-)\s+((?:19|20)\d{2})\b",
                    r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b"):
        m = find(pattern)
        if m:
            a, b = sorted([int(m.group(1)), int(m.group(2))])
            f.year_min, f.year_max = a, b
            mark(m)
            year_found = True
            break
    if not year_found:
        m = find(r"\b(?:the\s+)?((?:19|20)?\d0)'?s\b")
        if m:
            start = _decade_start(m.group(1))
            f.year_min, f.year_max = start, start + 9
            mark(m)
            year_found = True
    m = find(r"\b(?:after|since|post|newer than|later than|(?:nothing|not|no)\s+older than)\s+((?:19|20)\d{2})\b")
    if m:
        f.year_min = int(m.group(1))
        mark(m)
        year_found = True
    m = find(r"\b(?:before|pre|until|older than|earlier than)\s+((?:19|20)\d{2})\b")
    if m:
        f.year_max = int(m.group(1)) - (0 if m.group(0).startswith("until") else 1)
        mark(m)
        year_found = True
    if not year_found:
        m = find(r"\b(?:in\s+)?((?:19|20)\d{2})\b")
        if m:
            f.year_min = f.year_max = int(m.group(1))
            mark(m)
            year_found = True
    if not year_found:
        m = find(r"\b(recent|new|newer|latest)\b")
        if m:
            f.year_min = date.today().year - 3
            mark(m)
        m = find(r"\b(classic|old|older|vintage)\b")
        if m:
            f.year_max = 1989
            mark(m)
    if f.year_min or f.year_max:
        q.understood.append(f"📅 {f.year_min or '…'}–{f.year_max or 'today'}")

    # --- runtime --------------------------------------------------------------
    unit = r"(hours?|hrs?|h|minutes?|mins?|m)\b"
    m = find(r"\b(?:under|less than|shorter than|below|max(?:imum)?|no more than)\s+(\d+(?:\.\d+)?)\s*" + unit)
    if m:
        f.runtime_max = _minutes(m.group(1), m.group(2))
        mark(m)
    m = find(r"\b(?:over|more than|longer than|at least)\s+(\d+(?:\.\d+)?)\s*" + unit)
    if m:
        f.runtime_min = _minutes(m.group(1), m.group(2))
        mark(m)
    m = find(r"\bnot (?:too |very |that |overly )?long\b")
    if m:
        f.runtime_max = 120
        mark(m)
    if not (f.runtime_min or f.runtime_max):
        m = find(r"\b(short|quick)\b")
        if m:
            f.runtime_max = 100
            mark(m)
        m = find(r"\b(long|epic)\b")
        if m:
            f.runtime_min = 150
            mark(m)
    if f.runtime_min or f.runtime_max:
        q.understood.append(f"⏱️ {f.runtime_min or 0}–{f.runtime_max or '∞'} min")

    # --- rating ---------------------------------------------------------------
    m = find(r"\b(?:rated|rating|score)\s*(?:of\s*)?(?:above|over|at least|>=?)?\s*(\d(?:\.\d)?)\b") \
        or find(r"\b(\d(?:\.\d)?)\s*(?:\+|/10|stars?|or (?:higher|more|better))")
    if m:
        f.min_rating = min(float(m.group(1)), 10.0)
        f.min_votes = max(f.min_votes, 50)
        mark(m)
    m = find(r"\b(hidden gems?|underrated|under the radar)\b")
    if m:
        f.min_rating = max(f.min_rating, 7.0)
        f.min_votes, f.max_votes = max(f.min_votes, 150), 3000
        mark(m)
        q.understood.append("💎 hidden gems (well rated, less known)")
    m = find(r"\b(highly rated|critically acclaimed|acclaimed|top rated|top-rated|best|masterpieces?|must-see)\b")
    if m:
        f.min_rating = max(f.min_rating, 7.5)
        f.min_votes = max(f.min_votes, 500)
        mark(m)
    if f.min_rating:
        q.understood.append(f"⭐ rating {f.min_rating:g}+")

    # --- certification / audience -------------------------------------------
    m = find(r"\b(kid-?friendly|for (?:the )?kids|family-?friendly|for children|all ages)\b")
    if m:
        f.certifications = ["G", "PG"]
        mark(m)
    else:
        m = find(r"\bpg-?13\b")
        if m:
            f.certifications = ["PG-13"]
            mark(m)
        m2 = find(r"\b(?:rated r|r-rated|r rated)\b")
        if m2:
            f.certifications = f.certifications + ["R"]
            mark(m2)
    if f.certifications:
        q.understood.append("🔞 " + "/".join(f.certifications))

    # --- language -------------------------------------------------------------
    for word, code in LANGUAGE_WORDS.items():
        m = find(rf"\b{word}\b")
        if m:
            f.language = code
            mark(m)
            q.understood.append(f"🌐 {word.title()}-language")
            break

    # --- genres (with "no horror" style exclusions) ---------------------------
    for pattern, genres in GENRE_SYNONYMS.items():
        for m in re.finditer(rf"\b(?:{pattern})\b", low):
            if any(s <= m.start() < e for s, e in used):
                continue  # already part of something else, e.g. "rom-com"
            negated = NEGATION_BEFORE.search(low[:m.start()])
            target = f.genres_exclude if negated else f.genres_include
            for g in genres:
                if g not in target:
                    target.append(g)
            mark(m)
            if negated:
                used.append(negated.span())
    if f.genres_include:
        q.understood.append("🎭 " + " + ".join(f.genres_include))
    for g in f.genres_exclude:
        q.understood.append(f"🚫 no {g}")

    # --- mood (sentiment target) ---------------------------------------------
    for mood, pattern in MOOD_WORDS.items():
        m = find(rf"\b(?:{pattern})\b")
        if m:
            q.mood = mood
            mark(m)
            q.understood.append("😊 uplifting tone" if mood == "positive" else "🌑 darker tone")
            break

    # --- leftover keywords ----------------------------------------------------
    # Drop negated words ("less violence", "without gore"): we can't search for absence.
    for m in re.finditer(r"\b(?:less|no|not|without|fewer|nothing)\s+(?:too\s+)?\w+", low):
        mark(m)
    chars = list(original)
    for s, e in used:
        for i in range(s, min(e, len(chars))):
            chars[i] = " "
    leftover = re.findall(r"[a-zA-Z][a-zA-Z\-']+", "".join(chars).lower())
    words = [w for w in leftover
             if w not in ENGLISH_STOP_WORDS and w not in FILLER and len(w) > 2 and "'" not in w]
    q.keywords = " ".join(words)
    if q.keywords:
        q.understood.append(f"🔑 about: {q.keywords}")

    return q
