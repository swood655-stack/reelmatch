"""
tmdb_client.py
--------------
Everything that talks to the TMDB API lives here.

Key decisions:
- All requests go through one function (`tmdb_get`) so retries, error
  handling and caching are handled in a single place.
- Results are cached with Streamlit's cache for several hours. Movie data
  barely changes day to day, and caching keeps us well under TMDB's rate
  limits (the assignment notes ask us to plan for this).
- The API key is read from Streamlit "secrets" (or an environment
  variable), never written into the code, so it stays private on GitHub.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

try:  # Streamlit is available when the app runs; tests can run without it.
    import streamlit as st

    cache_data = st.cache_data
except ImportError:  # pragma: no cover - only used in offline tests
    st = None

    def cache_data(*dargs, **dkwargs):
        if dargs and callable(dargs[0]):
            return dargs[0]
        return lambda f: f


BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/"
REQUEST_TIMEOUT = 10  # seconds
MAX_RETRIES = 3


class TMDBError(Exception):
    """Raised when TMDB cannot be reached or returns an error."""


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
def get_api_key():
    """Return the TMDB key from Streamlit secrets or the environment."""
    key = None
    if st is not None:
        try:
            key = st.secrets.get("TMDB_API_KEY")
        except Exception:  # no secrets file at all
            key = None
    key = key or os.environ.get("TMDB_API_KEY")
    return key.strip() if key else None


def _auth(key):
    """TMDB accepts either a short v3 key (query param) or a long v4
    'Read Access Token' (Bearer header). Support both so either one works."""
    if key.startswith("eyJ"):  # v4 tokens are JWTs and start with eyJ
        return {"Authorization": f"Bearer {key}"}, {}
    return {}, {"api_key": key}


# ---------------------------------------------------------------------------
# Core request function
# ---------------------------------------------------------------------------
def _request(path, params, key):
    """Send one GET request to TMDB with retries. Not cached (see below)."""
    if not key:
        raise TMDBError("No TMDB API key found. Add TMDB_API_KEY to the app's secrets.")
    headers, auth_params = _auth(key)
    query = {"language": "en-US", **(params or {}), **auth_params}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(f"{BASE_URL}{path}", params=query,
                                headers=headers, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise TMDBError(f"Could not reach TMDB ({exc.__class__.__name__}).") from exc
            time.sleep(1 + attempt)
            continue

        if resp.status_code == 429:  # rate limited: wait and try again
            wait = float(resp.headers.get("Retry-After", 1 + attempt))
            time.sleep(min(wait, 5))
            continue
        if resp.status_code == 401:
            raise TMDBError("TMDB rejected the API key. Double-check TMDB_API_KEY in the app's secrets.")
        if resp.status_code == 404:
            raise TMDBError("TMDB could not find that item.")
        if not resp.ok:
            if attempt == MAX_RETRIES - 1:
                raise TMDBError(f"TMDB returned an error (status {resp.status_code}).")
            time.sleep(1 + attempt)
            continue
        return resp.json()

    raise TMDBError("TMDB is busy right now. Please try again in a moment.")


@cache_data(ttl=6 * 60 * 60, show_spinner=False)
def tmdb_get(path, params=None):
    """Cached GET of a TMDB endpoint; returns the JSON body as a dict."""
    return _request(path, params, get_api_key())


# ---------------------------------------------------------------------------
# Convenience wrappers for the endpoints the app uses
# ---------------------------------------------------------------------------
def poster_url(path, size="w342"):
    return f"{IMAGE_BASE}{size}{path}" if path else None


@cache_data(ttl=24 * 60 * 60, show_spinner=False)
def genre_map():
    """{genre_id: genre_name} for all TMDB movie genres."""
    data = tmdb_get("/genre/movie/list")
    return {g["id"]: g["name"] for g in data.get("genres", [])}


def search_movies(query, page=1):
    query = (query or "").strip()
    if not query:
        return []
    data = tmdb_get("/search/movie", {"query": query, "page": page, "include_adult": "false"})
    return data.get("results", [])


def search_person(name):
    """Return (person_id, matched_name) for the most popular match, or None."""
    name = (name or "").strip()
    if not name:
        return None
    results = tmdb_get("/search/person", {"query": name}).get("results", [])
    if not results:
        return None
    best = max(results, key=lambda p: p.get("popularity", 0))
    return best["id"], best["name"]


# Movie details are fetched in parallel threads. Streamlit's cache is designed
# for the main script thread, so details use a simple thread-safe cache of
# their own that lasts for the life of the app (refreshed after 24 hours).
_DETAILS_CACHE = {}
_DETAILS_LOCK = threading.Lock()
DETAILS_TTL = 24 * 60 * 60


def movie_details(movie_id, key=None):
    """Full details for one movie, including keywords, credits and
    certifications, in a single request (append_to_response)."""
    now = time.time()
    with _DETAILS_LOCK:
        hit = _DETAILS_CACHE.get(movie_id)
    if hit and now - hit[0] < DETAILS_TTL:
        return hit[1]
    data = _request(f"/movie/{movie_id}",
                    {"append_to_response": "keywords,credits,release_dates"},
                    key or get_api_key())
    with _DETAILS_LOCK:
        _DETAILS_CACHE[movie_id] = (now, data)
    return data


def movie_details_many(movie_ids, max_workers=8):
    """Fetch many movies in parallel. Movies that fail are skipped instead
    of crashing the whole page."""
    key = get_api_key()  # read once here, in the main thread

    def safe(mid):
        try:
            return movie_details(mid, key)
        except TMDBError:
            return None

    ids = list(dict.fromkeys(movie_ids))  # remove duplicates, keep order
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(safe, ids))
    return [r for r in results if r]


def discover(params, page=1):
    """TMDB /discover/movie: filter the whole TMDB catalogue."""
    return tmdb_get("/discover/movie", {**params, "page": page, "include_adult": "false"})


def trending(window="week", page=1):
    return tmdb_get(f"/trending/movie/{window}", {"page": page}).get("results", [])


def tmdb_recommendations(movie_id, pages=2):
    """IDs TMDB itself links to a movie. We only use these to widen the
    candidate pool; our own algorithm still does the ranking."""
    ids = []
    for page in range(1, pages + 1):
        try:
            data = tmdb_get(f"/movie/{movie_id}/recommendations", {"page": page})
        except TMDBError:
            break
        ids += [m["id"] for m in data.get("results", [])]
        if page >= data.get("total_pages", 1):
            break
    return ids
