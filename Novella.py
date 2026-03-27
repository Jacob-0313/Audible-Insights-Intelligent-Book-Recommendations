
import streamlit as st
import pickle
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import requests

# -----------------------------
# LOAD DATA
# -----------------------------
df = pickle.load(open("books.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

st.set_page_config(page_title="📚 Audible Insights", layout="wide")

# -----------------------------
# 🎨 PREMIUM CSS
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(to right, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    color: transparent;
}
.card {
    border-radius: 15px;
    padding: 20px;
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 🌟 HEADER
# -----------------------------
st.markdown('<p class="big-title">📚 Audible Insights</p>', unsafe_allow_html=True)
st.markdown("### Discover books tailored to your taste 🚀")

# -----------------------------
# 📑 TABS
# -----------------------------
tab1, tab2 = st.tabs(["📚 Recommender", "📊 Analytics Dashboard"])

# =========================================================
# 📚 TAB 1 → RECOMMENDER SYSTEM
# =========================================================
with tab1:

    st.sidebar.header("🔍 Smart Filters")

    # Extract genres
    genres = set()
    for g in df['Ranks and Genre'].astype(str):
        for item in g.split("|"):
            genres.add(item.strip())

    genre_list = sorted(genres)

    selected_genres = st.sidebar.multiselect("📚 Select Genres", genre_list)
    selected_author = st.sidebar.text_input("👤 Search Author")

    # Apply filters
    filtered_df = df.copy()

    if selected_genres:
        filtered_df = filtered_df[
            filtered_df['Ranks and Genre'].str.contains("|".join(selected_genres), case=False, na=False)
        ]

    if selected_author:
        filtered_df = filtered_df[
            filtered_df['Author'].str.contains(selected_author, case=False, na=False)
        ]

    # Search bar
    search_query = st.text_input("🔎 Search Book")

    if search_query:
        filtered_df = filtered_df[
            filtered_df['Book Name'].str.contains(search_query, case=False, na=False)
        ]

    # Dropdown
    book_list = filtered_df['Book Name'].values

    if len(book_list) == 0:
        st.warning("⚠️ No books found")
        st.stop()

    selected_book = st.selectbox("📖 Select Book", book_list)

    # -----------------------------
    # HYBRID MODEL
    # -----------------------------
    def hybrid_recommend(book_name, top_n=10):
        idx = df[df['Book Name'] == book_name].index[0]

        similarity = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        hybrid_score = 0.7 * similarity + 0.3 * (df['Rating'] / df['Rating'].max())

        df['score'] = hybrid_score

        return df.sort_values(by='score', ascending=False).iloc[1:top_n+1]

    # -----------------------------
    # IMAGE FETCH
    # -----------------------------
    def get_book_image(title):
        try:
            url = f"https://www.googleapis.com/books/v1/volumes?q={title}"
            res = requests.get(url).json()
            return res['items'][0]['volumeInfo']['imageLinks']['thumbnail']
        except:
            return "https://via.placeholder.com/128x180.png?text=No+Image"

    # -----------------------------
    # DISPLAY CARDS
    # -----------------------------
    def display_books(results):
        cols = st.columns(3)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 3]:
                image = get_book_image(row['Book Name'])

                st.markdown(f"""
                <div class="card">
                    <img src="{image}" width="100%">
                    <h4>📘 {row['Book Name']}</h4>
                    <p><b>👤</b> {row['Author']}</p>
                    <p><b>⭐ Rating:</b> {round(row['Rating'],2)}</p>
                    <p><b>🏷</b> {row['Ranks and Genre']}</p>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(row['Rating']/5, 1.0))

    # -----------------------------
    # BUTTON
    # -----------------------------
    if st.button("✨ Get Smart Recommendations"):
        results = hybrid_recommend(selected_book, 10)

        st.subheader("🔥 Top 10 Picks For You")
        display_books(results)

# =========================================================
# 📊 TAB 2 → ANALYTICS DASHBOARD
# =========================================================
with tab2:

    st.header("📊 Data Analytics Dashboard")

    # KPI Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("📚 Total Books", len(df))
    col2.metric("⭐ Avg Rating", round(df['Rating'].mean(), 2))
    col3.metric("👤 Unique Authors", df['Author'].nunique())

    # -----------------------------
    # Rating Distribution
    # -----------------------------
    st.subheader("⭐ Rating Distribution")

    fig1 = px.histogram(df, x="Rating", nbins=20)
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------
    # Top Authors
    # -----------------------------
    st.subheader("👤 Top Authors")

    top_authors = (
        df.groupby("Author")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig2 = px.bar(top_authors, x="Rating", y="Author", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # Genre Popularity
    # -----------------------------
    st.subheader("📚 Genre Popularity")

    genre_series = df['Ranks and Genre'].dropna().str.split('|').explode()
    genre_counts = genre_series.value_counts().head(10)

    fig3 = px.bar(x=genre_counts.values, y=genre_counts.index, orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

    # -----------------------------
    # Reviews vs Rating
    # -----------------------------
    st.subheader("🔥 Reviews vs Rating")

    fig4 = px.scatter(df, x="Number of Reviews", y="Rating", color="Rating")
    st.plotly_chart(fig4, use_container_width=True)

    # -----------------------------
    # Price vs Rating
    # -----------------------------
    st.subheader("💰 Price vs Rating")

    fig5 = px.scatter(df, x="Price", y="Rating", color="Rating")
    st.plotly_chart(fig5, use_container_width=True)
