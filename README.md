# 🎞️ ReelMatch

A movie recommendation app built with Python and Streamlit on top of the TMDB API.
Pick a movie you love or describe your mood in plain English, and ReelMatch finds
strong matches. Every pick comes with a reason.

## Features

- **Two recommendation approaches, plus a hybrid of both**
  - Content-based: TF-IDF on genres, keywords, director, cast and plot, compared with cosine similarity
  - Sentiment-based: VADER sentiment analysis of plot summaries, matching movies by emotional tone
  - Hybrid (default): 65% content + 25% tone + 10% quality (vote-weighted rating)
- **All required filters:** years, minimum rating and votes, runtime, US certification,
  language, actor/director, and genres with AND / OR / NOT
- **Extra features:** natural-language search ("Ask ReelMatch"), similar movies with
  explanations, trending movies, and data visualizations (Insights)

## Project structure

| File | What it does |
|---|---|
| `app.py` | Streamlit app: navigation and the six pages |
| `tmdb_client.py` | TMDB API calls, retries, error handling, caching |
| `data_pipeline.py` | Cleans TMDB data into a table; feature engineering |
| `nlp.py` | Text soup for TF-IDF, VADER sentiment, natural-language query parser |
| `recommender.py` | Content, sentiment and hybrid recommendation logic + explanations |
| `filters.py` | Filter rules, used for both TMDB queries and local data |
| `ui.py` | Branding, movie cards, sidebar filter panel |
| `requirements.txt` | Python libraries needed |
| `.streamlit/config.toml` | Colour theme |

## Running it

**Streamlit Community Cloud (recommended):** deploy this repository with `app.py` as the
main file, then add your key under *App settings → Secrets*:

```toml
TMDB_API_KEY = "your-tmdb-api-key"
```

**Locally:** `pip install -r requirements.txt`, copy `.streamlit/secrets.toml.example`
to `.streamlit/secrets.toml` and add your key, then run `streamlit run app.py`.

## Data credit

This product uses the TMDB API but is not endorsed or certified by TMDB.
