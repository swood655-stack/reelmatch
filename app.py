"""
ReelMatch - movie recommendation app (Streamlit)
================================================
Run locally:   streamlit run app.py
Needs:         TMDB_API_KEY in .streamlit/secrets.toml (or Streamlit Cloud secrets)

Pages
-----
Home            - what the app does, with trending posters
Discover        - search TMDB's full catalogue with every filter
Similar Movies  - pick a movie, get ranked matches with reasons
Ask ReelMatch   - type what you want in plain English
Trending        - what's popular today / this week
Insights        - charts on ratings, genres, runtime, mood and box office

Module map
----------
tmdb_client.py   API calls, retries, caching
data_pipeline.py turning TMDB data into a clean table + features
nlp.py           TF-IDF text soup, VADER sentiment, query parser
recommender.py   content, sentiment and hybrid recommendation logic
filters.py       the filter rules (used for TMDB queries and local data)
ui.py            branding, movie cards, sidebar filter panel
"""

import pandas as pd
import plotly.express as px
import streamlit as st

import data_pipeline as dp
import filters as flt
import nlp
import tmdb_client as tmdb
import ui
from recommender import METHODS, Recommender

st.set_page_config(page_title="ReelMatch – Movie Recommendations", page_icon="🎞️", layout="wide")
ui.inject_css()

PAGES = ["🏠 Home", "🔎 Discover", "🎯 Similar Movies", "💬 Ask ReelMatch", "🔥 Trending", "📊 Insights"]
FILTER_PAGES = {"🔎 Discover", "🎯 Similar Movies"}


# ---------------------------------------------------------------------------
# Shared, cached resources
# ---------------------------------------------------------------------------
def get_pool():
    with st.spinner("Loading the movie library… the first load takes about 20 seconds."):
        return dp.build_movie_pool()


@st.cache_resource(show_spinner=False)
def _fit_recommender(_pool, pool_key):
    # `_pool` starts with an underscore so Streamlit doesn't try to hash the
    # whole table; `pool_key` tells the cache when the pool has changed.
    return Recommender(_pool)


def get_recommender(pool):
    return _fit_recommender(pool, (len(pool), int(pool["id"].sum())))


def genre_ids():
    """{genre_name: id}"""
    return {name: gid for gid, name in tmdb.genre_map().items()}


def lookup_people(names):
    """Turn names into TMDB person ids. Returns (ids, matched_names, missing)."""
    ids, matched, missing = [], [], []
    for name in names:
        hit = tmdb.search_person(name)
        if hit:
            ids.append(hit[0])
            matched.append(hit[1])
        else:
            missing.append(name)
    return ids, matched, missing


def go_to(page):
    st.session_state.page = page


def no_results(tips):
    st.info("No movies match all of these settings. Try one of these:\n\n" + "\n".join(f"- {t}" for t in tips))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    ui.wordmark("2.8rem")
    st.write(
        "ReelMatch helps you skip the endless scrolling. Tell it a movie you love, or just describe "
        "your mood, and it finds strong matches from TMDB's database of over a million titles. "
        "Every pick comes with a plain-English reason, so you know why it's there."
    )
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "🎯 Similar Movies", "Loved a movie? Get ranked matches by story, cast, crew and tone."),
        (c2, "💬 Ask ReelMatch", "Type it like you'd say it: “feel-good 90s comedies under 2 hours”."),
        (c3, "🔎 Discover", "Dig through the full catalogue with precise filters."),
        (c4, "📊 Insights", "See trends in ratings, genres, mood and box office."),
    ]
    for col, page, text in cards:
        with col:
            with st.container(border=True):
                st.markdown(f"**{page}**")
                st.caption(text)
                st.button("Open", key=f"open_{page}", on_click=go_to, args=(page,))

    st.subheader("🔥 Trending this week")
    movies = tmdb.trending("week")[:6]
    cols = st.columns(6)
    for col, m in zip(cols, movies):
        with col:
            url = tmdb.poster_url(m.get("poster_path"), "w185")
            if url:
                st.image(url)
            st.caption(f"**{m.get('title', '')}**  \n⭐ {m.get('vote_average', 0):.1f}")


def page_discover(f):
    st.header("🔎 Discover")
    st.caption("Searches TMDB's full catalogue using the filters in the sidebar.")
    ui.chips(flt.describe(f), "No filters yet. Use the sidebar to narrow things down.")

    problems = flt.validate(f)
    if problems:
        for p in problems:
            st.warning(p)
        return

    sort_options = {"Most popular": "popularity.desc", "Highest rated": "vote_average.desc",
                    "Newest": "primary_release_date.desc", "Biggest box office": "revenue.desc"}
    c1, c2 = st.columns([3, 1])
    sort_label = c1.selectbox("Sort by", list(sort_options))
    page = c2.number_input("Page", min_value=1, max_value=500, value=1, step=1)

    person_ids, matched, missing = lookup_people(f.people)
    if missing:
        st.warning(f"Couldn't find anyone named: {', '.join(missing)}. Check the spelling.")
        return
    if matched and [m.lower() for m in matched] != [p.lower() for p in f.people]:
        st.caption("Searching for: " + ", ".join(matched))

    params = flt.to_discover_params(f, genre_ids(), person_ids)
    params["sort_by"] = sort_options[sort_label]
    if sort_label == "Newest":  # skip unreleased titles with no votes yet
        params["primary_release_date.lte"] = params.get("primary_release_date.lte",
                                                        pd.Timestamp.today().strftime("%Y-%m-%d"))
    data = tmdb.discover(params, int(page))
    results = data.get("results", [])
    total = data.get("total_results", 0)

    if not results:
        no_results(["Lower the minimum rating or number of votes",
                    "Widen the year or runtime range",
                    "Switch genres from AND to OR",
                    "Remove the certification or language filter"])
        return

    st.write(f"**{total:,} movies** match. Showing page {int(page)} of {min(data.get('total_pages', 1), 500)}.")
    with st.spinner("Loading movie details…"):
        df = dp.movies_to_df(tmdb.movie_details_many([m["id"] for m in results]))
    order = {m["id"]: i for i, m in enumerate(results)}
    df = df.sort_values("id", key=lambda s: s.map(order))
    ui.card_grid(df, columns=2)


def page_similar(f):
    st.header("🎯 Similar Movies")
    st.caption("Pick a movie you like. ReelMatch ranks matches by story, cast, crew and tone, "
               "then applies your sidebar filters.")

    query = st.text_input("Movie you like", placeholder="e.g. Inception, Mean Girls, Parasite")
    if not query.strip():
        st.info("Start by typing a movie title above.")
        return
    results = [m for m in tmdb.search_movies(query) if m.get("id")][:10]
    if not results:
        st.warning(f"No movies found for “{query}”. Check the spelling or try a shorter title.")
        return

    labels = {f"{m['title']} ({(m.get('release_date') or '????')[:4]})": m["id"] for m in results}
    c1, c2, c3 = st.columns([2.4, 2, 1])
    choice = c1.selectbox("Did you mean…", list(labels))
    method_label = c2.selectbox("Recommendation method", list(METHODS))
    n = c3.number_input("How many", 3, 30, 10)

    problems = flt.validate(f)
    if problems:
        for p in problems:
            st.warning(p)
        return

    seed_df = dp.movies_to_df([tmdb.movie_details(labels[choice])])
    if seed_df.empty:
        st.error("Couldn't load that movie's details. Try another one.")
        return
    seed = seed_df.iloc[0]

    with st.container(border=True):
        st.markdown("**You picked**")
        ui.movie_card(seed)
        st.caption(f"Plot tone score (VADER sentiment): {seed['compound']:+.2f} → {seed['mood']}")

    pool = get_pool()
    with st.spinner("Finding matches…"):
        candidates = dp.extend_pool(pool, tmdb.tmdb_recommendations(seed["id"]))
        filtered = flt.apply_filters(candidates, f)
        if filtered is None or filtered.empty or (filtered["id"] != seed["id"]).sum() == 0:
            no_results(["Loosen the sidebar filters (rating, votes, years)",
                        "Remove the actor/director filter",
                        "Try a different method or movie"])
            return
        recs = get_recommender(pool).recommend(seed, filtered, METHODS[method_label], int(n))

    ui.chips(flt.describe(f))
    st.subheader(f"Top {len(recs)} matches for {seed['title']}")
    ui.card_grid(recs, columns=2, with_reasons=True, show_scores=True)


EXAMPLES = [
    "feel-good 90s comedies under 2 hours",
    "movies like Interstellar but no horror",
    "dark crime thrillers directed by David Fincher",
    "kid-friendly animated adventures rated 7+",
    "hidden gem Korean thrillers",
    "romantic comedies with Julia Roberts",
]


def _set_example(text):
    st.session_state.ask_text = text


def page_ask():
    st.header("💬 Ask ReelMatch")
    st.caption("Describe what you're in the mood for. ReelMatch reads your sentence, turns it into "
               "filters, a mood and keywords, then ranks the results.")

    st.text_input("What do you feel like watching?", key="ask_text",
                  placeholder="e.g. uplifting sports movies from the 2000s with Denzel Washington")
    st.markdown("**Try one:**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 3].button(ex, key=f"ex_{i}", on_click=_set_example, args=(ex,))

    text = st.session_state.get("ask_text", "").strip()
    if not text:
        return
    if len(text) > 300:
        st.warning("That's a bit long. Try describing what you want in one sentence.")
        return

    q = nlp.parse_query(text)
    st.markdown("**Here's what I understood:**")
    ui.chips(q.understood, "I couldn't pick out specific filters, so I'll match on your words.")

    f = q.filters
    problems = flt.validate(f)
    if problems:
        for p in problems:
            st.warning(p)
        return

    # Look up people mentioned in the sentence.
    person_ids, matched, missing = lookup_people([name for name, _ in q.people])
    if missing:
        st.warning(f"Couldn't find {', '.join(missing)} on TMDB, so I ignored that part.")
    f.people = matched

    pool = get_pool()
    rec = get_recommender(pool)

    # Case 1: "movies like X" -> similar-movie flow with the parsed filters.
    if q.seed_title:
        hits = tmdb.search_movies(q.seed_title)
        if hits:
            seed_df = dp.movies_to_df([tmdb.movie_details(hits[0]["id"])])
            seed = seed_df.iloc[0]
            st.caption(f"Using **{seed['title']} ({seed['year']})** as the starting point.")
            with st.spinner("Finding matches…"):
                candidates = dp.extend_pool(pool, tmdb.tmdb_recommendations(seed["id"]))
                filtered = flt.apply_filters(candidates, f)
                if filtered.empty or (filtered["id"] != seed["id"]).sum() == 0:
                    no_results(["Remove one of the conditions", "Try a different movie"])
                    return
                recs = rec.recommend(seed, filtered, "hybrid", 12)
            st.subheader(f"Top {len(recs)} matches")
            ui.card_grid(recs, columns=2, with_reasons=True)
            return
        st.warning(f"Couldn't find a movie called “{q.seed_title}”, so I'll search by description.")

    # Case 2: filters + mood + keywords.
    with st.spinner("Searching…"):
        params = flt.to_discover_params(f, genre_ids(), person_ids)
        params.setdefault("vote_count.gte", 50)
        params["sort_by"] = "popularity.desc"
        ids = []
        for page in (1, 2, 3):
            data = tmdb.discover(params, page)
            ids += [m["id"] for m in data.get("results", [])]
            if page >= data.get("total_pages", 1):
                break
        fresh = dp.movies_to_df(tmdb.movie_details_many(ids)) if ids else pd.DataFrame()
        local = flt.apply_filters(pool, f)
        candidates = pd.concat([fresh, local], ignore_index=True)
        if not candidates.empty:
            candidates = candidates.drop_duplicates("id")

    if candidates.empty:
        no_results(["Drop one condition (for example the year or runtime)",
                    "Check the spelling of any names",
                    "Use broader words like 'comedy' or 'thriller'"])
        return
    recs = rec.rank_for_query(candidates, q.keywords, q.mood, n=12)
    if q.keywords and recs["text_score"].max() == 0:
        st.info(f"No movies matched the words “{q.keywords}”, so these are ranked by your other "
                "conditions and overall rating. Try different or simpler words.")
    st.subheader(f"Top {len(recs)} picks")
    ui.card_grid(recs, columns=2, with_reasons=True)


def page_trending():
    st.header("🔥 Trending")
    window = st.radio("Trending", ["This week", "Today"], horizontal=True, label_visibility="collapsed")
    movies = tmdb.trending("week" if window == "This week" else "day")
    if not movies:
        st.info("Trending data isn't available right now.")
        return
    gmap = tmdb.genre_map()
    cols_per_row = 5
    for i in range(0, len(movies), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, m in zip(cols, movies[i:i + cols_per_row]):
            with col:
                with st.container(border=True):
                    url = tmdb.poster_url(m.get("poster_path"))
                    if url:
                        st.image(url)
                    year = (m.get("release_date") or "")[:4]
                    st.markdown(f"**{m.get('title', 'Untitled')}** {f'({year})' if year else ''}")
                    st.caption(f"⭐ {m.get('vote_average', 0):.1f} · "
                               + ", ".join(gmap.get(g, '') for g in m.get("genre_ids", [])[:2]))
                    with st.expander("Plot"):
                        st.write(m.get("overview") or "No summary yet.")


def page_insights():
    st.header("📊 Insights")
    pool = get_pool()
    st.caption(f"Based on ReelMatch's library of {len(pool):,} popular and top-rated movies from TMDB.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies analysed", f"{len(pool):,}")
    c2.metric("Average rating", f"{pool['vote_average'].mean():.2f}")
    c3.metric("Median runtime", f"{int(pool['runtime'].median())} min")
    c4.metric("Uplifting plots", f"{(pool['mood'] == 'Uplifting').mean():.0%}")

    color_map = {"Uplifting": "#F5B700", "Mixed": "#8E9AAF", "Dark": "#3D405B"}

    left, right = st.columns(2)
    with left:
        by_decade = (pool.dropna(subset=["decade"]).groupby("decade")
                     .agg(avg_rating=("vote_average", "mean"), movies=("id", "count")).reset_index())
        by_decade["decade"] = by_decade["decade"].astype(int).astype(str) + "s"
        fig = px.bar(by_decade, x="decade", y="avg_rating", text="movies",
                     title="Average rating by decade (label = number of movies)",
                     labels={"avg_rating": "Average rating", "decade": ""})
        fig.update_traces(marker_color="#F5B700")
        fig.update_yaxes(range=[max(0, by_decade["avg_rating"].min() - 0.5), 10])
        st.plotly_chart(fig)
    with right:
        genres = pool.explode("genres").dropna(subset=["genres"])
        mood_mix = (genres.groupby(["genres", "mood"]).size().reset_index(name="n"))
        mood_mix["share"] = mood_mix["n"] / mood_mix.groupby("genres")["n"].transform("sum")
        fig = px.bar(mood_mix, y="genres", x="share", color="mood", orientation="h",
                     title="Plot tone by genre (from NLP sentiment analysis)",
                     color_discrete_map=color_map,
                     category_orders={"mood": ["Uplifting", "Mixed", "Dark"]},
                     labels={"share": "Share of movies", "genres": ""})
        fig.update_layout(xaxis_tickformat=".0%", height=520)
        st.plotly_chart(fig)

    left, right = st.columns(2)
    with left:
        rt = pool.dropna(subset=["runtime"])
        rt = rt[(rt["runtime"] > 0) & (rt["runtime"] < 300)]
        fig = px.scatter(rt, x="runtime", y="vote_average", color="mood", hover_name="title",
                         color_discrete_map=color_map, opacity=0.75,
                         title="Do longer movies rate higher?",
                         labels={"runtime": "Runtime (min)", "vote_average": "Rating"})
        st.plotly_chart(fig)
    with right:
        money = pool[(pool["budget"] > 1e6) & (pool["revenue"] > 1e6)].copy()
        if money.empty:
            st.info("Not enough box office data to chart.")
        else:
            money["ROI (x)"] = (money["revenue"] / money["budget"]).round(1)
            fig = px.scatter(money, x="budget", y="revenue", hover_name="title", color="vote_average",
                             log_x=True, log_y=True, color_continuous_scale="YlOrBr",
                             hover_data={"ROI (x)": True},
                             title="Budget vs box office (log scale)",
                             labels={"budget": "Budget ($)", "revenue": "Box office ($)",
                                     "vote_average": "Rating"})
            st.plotly_chart(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if "page" not in st.session_state:
        st.session_state.page = PAGES[0]

    with st.sidebar:
        ui.wordmark("1.6rem", show_tagline=False)
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()

    if not tmdb.get_api_key():
        st.error("ReelMatch needs a TMDB API key to run.")
        st.markdown("Add this line to the app's **Secrets** (Streamlit Cloud → App settings → Secrets):")
        st.code('TMDB_API_KEY = "your-key-here"', language="toml")
        st.stop()

    page = st.session_state.page
    try:
        f = ui.sidebar_filters(list(genre_ids())) if page in FILTER_PAGES else None
        if page == "🏠 Home":
            page_home()
        elif page == "🔎 Discover":
            page_discover(f)
        elif page == "🎯 Similar Movies":
            page_similar(f)
        elif page == "💬 Ask ReelMatch":
            page_ask()
        elif page == "🔥 Trending":
            page_trending()
        elif page == "📊 Insights":
            page_insights()
    except tmdb.TMDBError as e:
        st.error(f"⚠️ {e}")
    except Exception as e:  # last-resort safety net so users never see a raw crash
        st.error("Something went wrong on this page. Please try again or change your settings.")
        with st.expander("Technical details"):
            st.exception(e)

    st.sidebar.divider()
    st.sidebar.caption("Movie data from TMDB. This product uses the TMDB API but is not "
                       "endorsed or certified by TMDB.")


main()
