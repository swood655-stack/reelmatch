"""
filters.py
----------
The filtering system. One settings object (FilterSettings) is used two ways:

1. `to_discover_params` turns it into TMDB /discover parameters, so the
   Discover page can search TMDB's whole catalogue, not just our pool.
2. `apply_filters` applies the same rules to a local DataFrame, so
   recommendations and natural-language results respect the filters too.

Keeping both in one file means the rules stay consistent everywhere.
Genre logic supports AND (must have all), OR (any of) and NOT (exclude).
"""

from dataclasses import dataclass, field

import pandas as pd

CERTIFICATIONS = ["G", "PG", "PG-13", "R", "NC-17"]

LANGUAGES = {
    "Any": None, "English": "en", "French": "fr", "Spanish": "es", "Korean": "ko",
    "Japanese": "ja", "Hindi": "hi", "Italian": "it", "German": "de",
    "Chinese (Mandarin)": "zh", "Cantonese": "cn", "Portuguese": "pt",
    "Swedish": "sv", "Danish": "da", "Russian": "ru", "Turkish": "tr", "Thai": "th",
}


@dataclass
class FilterSettings:
    """Every field's default means 'no filter'."""
    year_min: int | None = None
    year_max: int | None = None
    min_rating: float = 0.0
    min_votes: int = 0
    max_votes: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None
    certifications: list = field(default_factory=list)
    language: str | None = None
    people: list = field(default_factory=list)          # actor/director names
    genres_include: list = field(default_factory=list)
    genre_mode: str = "AND"                             # "AND" or "OR"
    genres_exclude: list = field(default_factory=list)


def validate(f):
    """Return a list of problems with the settings (empty list = all good)."""
    problems = []
    if f.year_min and f.year_max and f.year_min > f.year_max:
        problems.append("The start year is after the end year.")
    if f.runtime_min and f.runtime_max and f.runtime_min > f.runtime_max:
        problems.append("The minimum runtime is longer than the maximum runtime.")
    both = set(f.genres_include) & set(f.genres_exclude)
    if both:
        problems.append(f"{', '.join(sorted(both))} can't be both included and excluded.")
    if not 0 <= f.min_rating <= 10:
        problems.append("Minimum rating must be between 0 and 10.")
    return problems


def describe(f):
    """Short human-readable labels for the active filters (shown as chips)."""
    out = []
    if f.year_min or f.year_max:
        out.append(f"📅 {f.year_min or '…'}–{f.year_max or 'today'}")
    if f.min_rating:
        out.append(f"⭐ {f.min_rating:g}+")
    if f.min_votes:
        out.append(f"🗳️ {f.min_votes:,}+ votes")
    if f.max_votes:
        out.append(f"💎 under {f.max_votes:,} votes")
    if f.runtime_min or f.runtime_max:
        out.append(f"⏱️ {f.runtime_min or 0}–{f.runtime_max or '∞'} min")
    if f.certifications:
        out.append("🔞 " + "/".join(f.certifications))
    if f.language:
        name = next((k for k, v in LANGUAGES.items() if v == f.language), f.language)
        out.append(f"🌐 {name}")
    for p in f.people:
        out.append(f"🎬 {p}")
    if f.genres_include:
        joiner = " AND " if f.genre_mode == "AND" else " OR "
        out.append("🎭 " + joiner.join(f.genres_include))
    for g in f.genres_exclude:
        out.append(f"🚫 {g}")
    return out


def apply_filters(df, f):
    """Apply the filter settings to a movie DataFrame."""
    if df is None or df.empty:
        return df
    mask = pd.Series(True, index=df.index)

    if f.year_min:
        mask &= df["year"].fillna(0) >= f.year_min
    if f.year_max:
        mask &= df["year"].fillna(9999) <= f.year_max
    if f.min_rating:
        mask &= df["vote_average"] >= f.min_rating
    if f.min_votes:
        mask &= df["vote_count"] >= f.min_votes
    if f.max_votes:
        mask &= df["vote_count"] <= f.max_votes
    if f.runtime_min:
        mask &= df["runtime"].fillna(0) >= f.runtime_min
    if f.runtime_max:
        mask &= df["runtime"].fillna(10_000) <= f.runtime_max
    if f.certifications:
        mask &= df["certification"].isin(f.certifications)
    if f.language:
        mask &= df["language"] == f.language

    for person in f.people:
        p = person.lower().strip()
        mask &= df.apply(lambda r: any(p == n.lower() for n in list(r["cast"]) + list(r["director"])), axis=1)

    if f.genres_include:
        wanted = set(f.genres_include)
        if f.genre_mode == "AND":
            mask &= df["genres"].apply(lambda g: wanted.issubset(g))
        else:
            mask &= df["genres"].apply(lambda g: bool(wanted & set(g)))
    if f.genres_exclude:
        banned = set(f.genres_exclude)
        mask &= df["genres"].apply(lambda g: not (banned & set(g)))

    return df[mask]


def to_discover_params(f, genre_ids, person_ids=None):
    """Translate settings into TMDB /discover/movie query parameters.

    genre_ids: {genre_name: genre_id}
    person_ids: list of TMDB person ids (already looked up from names)
    """
    p = {}
    if f.year_min:
        p["primary_release_date.gte"] = f"{f.year_min}-01-01"
    if f.year_max:
        p["primary_release_date.lte"] = f"{f.year_max}-12-31"
    if f.min_rating:
        p["vote_average.gte"] = f.min_rating
    if f.min_votes:
        p["vote_count.gte"] = f.min_votes
    if f.max_votes:
        p["vote_count.lte"] = f.max_votes
    if f.runtime_min:
        p["with_runtime.gte"] = f.runtime_min
    if f.runtime_max:
        p["with_runtime.lte"] = f.runtime_max
    if f.certifications:
        p["certification_country"] = "US"
        p["certification"] = "|".join(f.certifications)
    if f.language:
        p["with_original_language"] = f.language
    if person_ids:
        p["with_people"] = ",".join(str(i) for i in person_ids)  # comma = all of them
    inc = [str(genre_ids[g]) for g in f.genres_include if g in genre_ids]
    if inc:
        # TMDB syntax: comma means AND, pipe means OR
        p["with_genres"] = ("," if f.genre_mode == "AND" else "|").join(inc)
    exc = [str(genre_ids[g]) for g in f.genres_exclude if g in genre_ids]
    if exc:
        p["without_genres"] = ",".join(exc)
    return p
