"""
ui.py
-----
Reusable interface pieces: branding, styling, movie cards and the
sidebar filter panel. Keeping these here makes app.py easier to read
and keeps the look consistent on every page.
"""

from datetime import date

import streamlit as st

import tmdb_client as tmdb
from filters import CERTIFICATIONS, LANGUAGES, FilterSettings

APP_NAME = "ReelMatch"
TAGLINE = "Find your next favourite movie in seconds, and see why it fits."
THIS_YEAR = date.today().year
MOOD_BADGE = {"Uplifting": "🌤️ Uplifting", "Mixed": "🌗 Mixed", "Dark": "🌑 Dark"}

CSS = """
<style>
.rm-wordmark {font-size: 2.1rem; font-weight: 800; letter-spacing: -0.5px; line-height: 1.1;}
.rm-wordmark .reel {color: #F5B700;}
.rm-wordmark .match {color: inherit;}
.rm-tag {opacity: 0.75; margin-top: 0.1rem; margin-bottom: 1rem;}
.rm-chip {display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0; border-radius: 999px;
          background: rgba(245, 183, 0, 0.15); border: 1px solid rgba(245, 183, 0, 0.45);
          font-size: 0.82rem;}
.rm-genre {display: inline-block; padding: 1px 8px; margin: 2px 4px 2px 0; border-radius: 6px;
           background: rgba(128, 128, 128, 0.18); font-size: 0.78rem;}
.rm-noposter {height: 210px; border-radius: 8px; background: rgba(128,128,128,0.18);
              display: flex; align-items: center; justify-content: center; font-size: 2.2rem;}
.rm-title {font-size: 1.12rem; font-weight: 700; margin-bottom: 0.1rem;}
.rm-meta {opacity: 0.75; font-size: 0.86rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def wordmark(size="2.1rem", show_tagline=True):
    st.markdown(
        f'<div class="rm-wordmark" style="font-size:{size}">🎞️ <span class="reel">Reel</span>'
        f'<span class="match">Match</span></div>', unsafe_allow_html=True)
    if show_tagline:
        st.markdown(f'<div class="rm-tag">{TAGLINE}</div>', unsafe_allow_html=True)


def chips(labels, empty_text=None):
    if not labels:
        if empty_text:
            st.caption(empty_text)
        return
    st.markdown("".join(f'<span class="rm-chip">{_esc(x)}</span>' for x in labels),
                unsafe_allow_html=True)


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _runtime(mins):
    try:
        mins = int(mins)
    except (TypeError, ValueError):
        return None
    if mins <= 0:
        return None
    return f"{mins // 60}h {mins % 60:02d}m" if mins >= 60 else f"{mins}m"


def _money(n):
    n = float(n or 0)
    if n <= 0:
        return None
    return f"${n / 1e9:.2f}B" if n >= 1e9 else f"${n / 1e6:.0f}M"


def movie_card(row, reasons=None, match=None, scores=None, key=None):
    """One movie, shown as a bordered card with poster and details."""
    with st.container(border=True):
        left, right = st.columns([1, 2.6])
        with left:
            url = tmdb.poster_url(row.get("poster_path"))
            if url:
                st.image(url)
            else:
                st.markdown('<div class="rm-noposter">🎬</div>', unsafe_allow_html=True)
        with right:
            year = f" ({int(row['year'])})" if row.get("year") and row["year"] == row["year"] else ""
            st.markdown(f'<div class="rm-title">{_esc(row["title"])}{year}</div>', unsafe_allow_html=True)

            meta = [row.get("certification") or "Not rated", _runtime(row.get("runtime")),
                    str(row.get("language", "")).upper() or None,
                    MOOD_BADGE.get(row.get("mood"))]
            st.markdown(f'<div class="rm-meta">{" · ".join(m for m in meta if m)}</div>',
                        unsafe_allow_html=True)

            rating = f"⭐ **{row.get('vote_average', 0):.1f}**/10 · {int(row.get('vote_count', 0)):,} votes"
            box = _money(row.get("revenue"))
            st.markdown(rating + (f" · 💰 {box} box office" if box else ""))

            genres = row.get("genres") or []
            if len(genres):
                st.markdown("".join(f'<span class="rm-genre">{_esc(g)}</span>' for g in genres),
                            unsafe_allow_html=True)
            people = []
            if len(row.get("director") or []):
                people.append("**Director:** " + ", ".join(row["director"]))
            if len(row.get("cast") or []):
                people.append("**Cast:** " + ", ".join(list(row["cast"])[:4]))
            if people:
                st.markdown("  \n".join(people))

            if match is not None:
                st.progress(int(match) / 100, text=f"{int(match)}% match")
            if reasons:
                st.markdown("**Why this pick:**  \n" + "  \n".join(f"• {r}" for r in reasons))
            if scores:
                st.caption(" · ".join(f"{k} {v:.2f}" for k, v in scores.items()))

            overview = row.get("overview") or "No plot summary available."
            with st.expander("Plot summary"):
                st.write(overview)


def card_grid(df, columns=2, with_reasons=False, show_scores=False):
    """Lay out movie cards in a grid."""
    rows = list(df.iterrows())
    for i in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, (_, r) in zip(cols, rows[i:i + columns]):
            with col:
                scores = None
                if show_scores and "content_score" in r:
                    scores = {"Story": r["content_score"], "Tone": r["sentiment_score"],
                              "Quality": r["quality_score"]}
                movie_card(r, reasons=r.get("reasons") if with_reasons else None,
                           match=r.get("match") if with_reasons else None, scores=scores)


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
FILTER_KEYS = ["f_years", "f_rating", "f_votes", "f_runtime", "f_cert", "f_lang",
               "f_people", "f_gin", "f_gmode", "f_gex"]
VOTE_STEPS = [0, 50, 100, 250, 500, 1000, 5000, 10000]


def reset_filters():
    for k in FILTER_KEYS:
        st.session_state.pop(k, None)


def sidebar_filters(genre_names, default_votes=100):
    """Render the filter panel in the sidebar and return FilterSettings."""
    sb = st.sidebar
    sb.markdown("### 🎛️ Filters")
    years = sb.slider("Release years", 1920, THIS_YEAR, (1920, THIS_YEAR), key="f_years")
    rating = sb.slider("Minimum rating", 0.0, 10.0, 0.0, 0.5, key="f_rating")
    votes = sb.select_slider("Minimum number of votes", VOTE_STEPS, value=default_votes, key="f_votes",
                             help="Filters out obscure titles rated by only a handful of people.")
    runtime = sb.slider("Runtime (minutes)", 0, 240, (0, 240), 10, key="f_runtime",
                        help="240 means no upper limit.")
    certs = sb.multiselect("Certification (US)", CERTIFICATIONS, key="f_cert")
    lang = sb.selectbox("Original language", list(LANGUAGES), key="f_lang")
    people = sb.text_input("Actor or director", key="f_people",
                           placeholder="e.g. Greta Gerwig, Ryan Gosling",
                           help="Separate several names with commas. Movies must include all of them.")
    sb.markdown("**Genres**")
    g_in = sb.multiselect("Include", genre_names, key="f_gin")
    g_mode = sb.radio("Match", ["AND", "OR"], horizontal=True, key="f_gmode",
                      help="AND: movie must have every genre picked. OR: any of them.",
                      disabled=len(g_in) < 2)
    g_ex = sb.multiselect("Exclude (NOT)", genre_names, key="f_gex")
    sb.button("↺ Reset filters", on_click=reset_filters)

    return FilterSettings(
        year_min=years[0] if years[0] > 1920 else None,
        year_max=years[1] if years[1] < THIS_YEAR else None,
        min_rating=float(rating),
        min_votes=int(votes),
        runtime_min=runtime[0] or None,
        runtime_max=runtime[1] if runtime[1] < 240 else None,
        certifications=list(certs),
        language=LANGUAGES[lang],
        people=[p.strip() for p in people.split(",") if p.strip()],
        genres_include=list(g_in),
        genre_mode=g_mode,
        genres_exclude=list(g_ex),
    )
