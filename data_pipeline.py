"""
data_pipeline.py
----------------
Turns raw TMDB JSON into a clean table (pandas DataFrame) and adds the
features the recommender needs.

Key decisions:
- We build a "movie pool" of roughly 600 well-known movies (popular +
  top-rated). Recommendations are ranked within this pool, plus extra
  candidates pulled in for each seed movie, so results stay relevant
  without downloading TMDB's entire catalogue.
- Each movie gets a text "soup": genres, keywords, director, top cast
  and plot words combined into one string. Director and genres are
  repeated so they count for more in the similarity score.
"""

import pandas as pd

import nlp
import tmdb_client as tmdb
from tmdb_client import cache_data

POOL_PAGES = 15  # 15 pages x 20 movies x 2 lists = up to 600 movies


def _us_certification(details):
    """US rating (G, PG, PG-13, R, NC-17) from release_dates; '' if unknown."""
    for country in details.get("release_dates", {}).get("results", []):
        if country.get("iso_3166_1") == "US":
            for rel in country.get("release_dates", []):
                if rel.get("certification"):
                    return rel["certification"].strip()
    return ""


def parse_movie(d):
    """Flatten one TMDB details JSON into a simple dict (one table row)."""
    credits = d.get("credits", {})
    crew = credits.get("crew", [])
    cast = sorted(credits.get("cast", []), key=lambda c: c.get("order", 99))
    release = d.get("release_date") or ""
    year = int(release[:4]) if release[:4].isdigit() else None

    return {
        "id": d.get("id"),
        "title": d.get("title") or d.get("original_title") or "Untitled",
        "year": year,
        "release_date": release,
        "runtime": d.get("runtime") or None,
        "certification": _us_certification(d),
        "language": d.get("original_language") or "",
        "genres": [g["name"] for g in d.get("genres", [])],
        "keywords": [k["name"] for k in d.get("keywords", {}).get("keywords", [])],
        "cast": [c["name"] for c in cast[:6]],
        "director": [c["name"] for c in crew if c.get("job") == "Director"],
        "overview": d.get("overview") or "",
        "tagline": d.get("tagline") or "",
        "vote_average": d.get("vote_average") or 0.0,
        "vote_count": d.get("vote_count") or 0,
        "popularity": d.get("popularity") or 0.0,
        "budget": d.get("budget") or 0,
        "revenue": d.get("revenue") or 0,
        "poster_path": d.get("poster_path"),
    }


def add_features(df):
    """Feature engineering: decade, text soup and sentiment scores."""
    if df.empty:
        return df
    df = df.copy()
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    df["soup"] = df.apply(nlp.build_soup, axis=1)
    sent = df["overview"].apply(nlp.sentiment_scores).apply(pd.Series)
    df = pd.concat([df.drop(columns=[c for c in sent.columns if c in df]), sent], axis=1)
    df["mood"] = df["compound"].apply(nlp.mood_label)
    return df


def movies_to_df(details_list):
    rows = [parse_movie(d) for d in details_list if d and d.get("id")]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("id").reset_index(drop=True)
    return add_features(df)


@cache_data(ttl=24 * 60 * 60, show_spinner=False)
def build_movie_pool(pages=POOL_PAGES):
    """Collect popular and top-rated movies, fetch full details for each,
    and return the feature-ready DataFrame. Cached for a day."""
    ids = []
    for page in range(1, pages + 1):
        pop = tmdb.discover({"sort_by": "popularity.desc", "vote_count.gte": 300}, page)
        top = tmdb.tmdb_get("/movie/top_rated", {"page": page})
        ids += [m["id"] for m in pop.get("results", []) + top.get("results", [])]
    return movies_to_df(tmdb.movie_details_many(ids))


def extend_pool(pool, movie_ids):
    """Add extra movies (by id) to the pool. Used to widen candidates for a
    specific seed movie or for a natural-language query."""
    new_ids = [m for m in movie_ids if m not in set(pool["id"])]
    if not new_ids:
        return pool
    extra = movies_to_df(tmdb.movie_details_many(new_ids))
    if extra.empty:
        return pool
    return pd.concat([pool, extra], ignore_index=True).drop_duplicates("id").reset_index(drop=True)
