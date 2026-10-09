import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="TuneLens",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>

    /* =====================================================
       GLOBAL
       ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 20% 0%,
                rgba(124, 58, 237, 0.18),
                transparent 32%
            ),
            radial-gradient(
                circle at 85% 35%,
                rgba(79, 70, 229, 0.10),
                transparent 28%
            ),
            #111114;

        color: #ffffff;
    }

    .main {
        padding-top: 1.5rem;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background: #0d0d11;
        border-right: 1px solid #29262f;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .sidebar-brand {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .sidebar-subtitle {
        color: #96929f;
        font-size: 13px;
        margin-top: -5px;
        margin-bottom: 20px;
    }


    /* =====================================================
       HERO
       ===================================================== */

    .hero {
        position: relative;

        min-height: 300px;

        padding: 42px;

        border-radius: 28px;

        overflow: hidden;

        background:
            radial-gradient(
                circle at 80% 20%,
                rgba(168, 85, 247, 0.55),
                transparent 28%
            ),
            radial-gradient(
                circle at 60% 90%,
                rgba(99, 102, 241, 0.35),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #18151e,
                #21152d 50%,
                #17131d
            );

        border: 1px solid #3b3048;

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.35);
    }

    .hero::after {
        content: "";

        position: absolute;

        width: 280px;
        height: 280px;

        right: -80px;
        top: -80px;

        border-radius: 50%;

        background:
            radial-gradient(
                circle,
                rgba(196, 181, 253, 0.28),
                transparent 65%
            );

        filter: blur(10px);
    }

    .hero-eyebrow {
        color: #c4b5fd;
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;

        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 48px;
        line-height: 1.05;
        font-weight: 850;
        letter-spacing: -2px;

        max-width: 650px;
    }

    .hero-title span {
        background:
            linear-gradient(
                90deg,
                #c4b5fd,
                #a855f7,
                #818cf8
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #bbb5c4;

        font-size: 16px;
        line-height: 1.6;

        max-width: 600px;

        margin-top: 16px;
    }

    .hero-note {
        display: inline-block;

        margin-top: 24px;

        padding: 8px 14px;

        border-radius: 999px;

        background: rgba(255,255,255,0.07);

        border: 1px solid rgba(255,255,255,0.10);

        color: #d7d2df;

        font-size: 13px;
    }


    /* =====================================================
       SECTION HEADINGS
       ===================================================== */

    .section-title {
        font-size: 27px;
        font-weight: 750;

        margin-top: 34px;
        margin-bottom: 18px;

        letter-spacing: -0.5px;
    }

    .section-subtitle {
        color: #9994a3;
        font-size: 14px;
        margin-top: -12px;
        margin-bottom: 20px;
    }


    /* =====================================================
       STATS
       ===================================================== */

    .stat-card {
        background: #19191f;

        border: 1px solid #2c2932;

        border-radius: 18px;

        padding: 22px;

        min-height: 110px;

        transition: all 0.2s ease;
    }

    .stat-card:hover {
        border-color: #6d4aa3;

        transform: translateY(-3px);
    }

    .stat-icon {
        font-size: 22px;
    }

    .stat-number {
        font-size: 26px;
        font-weight: 800;

        margin-top: 8px;
    }

    .stat-label {
        color: #96919e;
        font-size: 13px;
        margin-top: 3px;
    }


    /* =====================================================
       MUSIC CARDS
       ===================================================== */

    .music-card {
        background: #19191f;

        border: 1px solid #2b2831;

        border-radius: 18px;

        padding: 15px;

        transition: all 0.2s ease;
    }

    .music-card:hover {
        transform: translateY(-4px);

        border-color: #7547a8;

        box-shadow:
            0 12px 35px rgba(
                124,
                58,
                237,
                0.13
            );
    }

    .album-art {
        height: 105px;

        border-radius: 13px;

        margin-bottom: 14px;

        background:
            linear-gradient(
                135deg,
                #5b21b6,
                #7c3aed 45%,
                #312e81
            );

        display: flex;

        align-items: center;
        justify-content: center;

        font-size: 36px;

        box-shadow:
            inset 0 0 35px rgba(
                255,
                255,
                255,
                0.08
            );
    }

    .music-title {
        font-size: 15px;
        font-weight: 700;

        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .music-artist {
        color: #9b96a4;

        font-size: 13px;

        margin-top: 5px;

        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .music-meta {
        color: #b8a7d0;

        font-size: 12px;

        margin-top: 10px;
    }


    /* =====================================================
       MOOD CARDS
       ===================================================== */

    .mood-card {
        min-height: 145px;

        border-radius: 20px;

        padding: 22px;

        position: relative;

        overflow: hidden;

        border: 1px solid rgba(
            255,
            255,
            255,
            0.08
        );
    }

    .mood-card::after {
        content: "";

        position: absolute;

        width: 130px;
        height: 130px;

        right: -40px;
        bottom: -50px;

        border-radius: 50%;

        background: rgba(
            255,
            255,
            255,
            0.12
        );

        filter: blur(5px);
    }

    .mood-1 {
        background:
            linear-gradient(
                135deg,
                #6d28d9,
                #a855f7
            );
    }

    .mood-2 {
        background:
            linear-gradient(
                135deg,
                #312e81,
                #7c3aed
            );
    }

    .mood-3 {
        background:
            linear-gradient(
                135deg,
                #4338ca,
                #6366f1
            );
    }

    .mood-4 {
        background:
            linear-gradient(
                135deg,
                #37304a,
                #5b21b6
            );
    }

    .mood-icon {
        font-size: 28px;
    }

    .mood-name {
        font-size: 15px;
        font-weight: 750;

        margin-top: 18px;
    }

    .mood-count {
        color: rgba(255,255,255,0.72);

        font-size: 12px;

        margin-top: 4px;
    }


    /* =====================================================
       BUTTONS
       ===================================================== */

    .stButton > button {
        border-radius: 12px;

        background:
            linear-gradient(
                135deg,
                #7c3aed,
                #9333ea
            );

        border: 1px solid #a855f7;

        color: white;

        font-weight: 700;

        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);

        box-shadow:
            0 8px 25px rgba(
                124,
                58,
                237,
                0.3
            );
    }


    /* =====================================================
       SELECT BOX
       ===================================================== */

    div[data-baseweb="select"] > div {
        background: #19191f;

        border: 1px solid #35303d;

        border-radius: 12px;
    }


    /* =====================================================
       ALERTS
       ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 15px;

        background: #1b1722;

        border: 1px solid #503476;
    }


    /* =====================================================
       SCROLLBAR
       ===================================================== */

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #111114;
    }

    ::-webkit-scrollbar-thumb {
        background: #4c3b5e;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #7c3aed;
    }

</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------
# LOAD AND CLEAN DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/spotify_tracks.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop_duplicates()

    df = df.dropna(
        subset=["artists", "album_name", "track_name"]
    )

    df = df.reset_index(drop=True)

    df["duration_min"] = df["duration_ms"] / 60000

    def classify_mood(row):

        if row["energy"] >= 0.7 and row["valence"] >= 0.6:
            return "Energetic & Happy"

        elif row["energy"] >= 0.7 and row["valence"] < 0.6:
            return "Energetic & Moody"

        elif row["energy"] < 0.7 and row["valence"] >= 0.6:
            return "Calm & Positive"

        else:
            return "Calm & Moody"

    df["mood"] = df.apply(
        classify_mood,
        axis=1
    )

    return df


df = load_data()


# ---------------------------------------------------------
# RECOMMENDATION ENGINE
# ---------------------------------------------------------

recommendation_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

recommendation_data = df[
    recommendation_features
].copy()

scaler = StandardScaler()

feature_matrix = scaler.fit_transform(
    recommendation_data
)


def recommend_songs(track_name, n=5):

    matches = df[
        df["track_name"].str.lower()
        == track_name.lower()
    ]

    if matches.empty:
        return None

    track_index = matches.index[0]

    similarity_scores = cosine_similarity(
        feature_matrix[track_index].reshape(1, -1),
        feature_matrix
    )[0]

    similar_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for index in similar_indices:

        if index != track_index:
            recommendations.append(index)

        if len(recommendations) == n:
            break

    result = df.loc[
        recommendations,
        [
            "track_name",
            "artists",
            "track_genre",
            "popularity",
            "mood"
        ]
    ].copy()

    return result


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown("# 🎵 TuneLens")

st.sidebar.markdown(
    "### Discover your sound."
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Explore",
    [
        "🏠 Home",
        "🎧 Discover",
        "📊 Music Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "TuneLens • Music Analytics & Recommendation Platform"
)


if page == "🏠 Home":

    # =====================================================
    # HERO
    # =====================================================

    st.markdown(
        '<div class="hero">'
        '<div class="hero-eyebrow">MUSIC ANALYTICS • DISCOVERY</div>'
        '<div class="hero-title">'
        'Discover your <span>sound.</span><br>'
        'Understand your music.'
        '</div>'
        '<div class="hero-description">'
        'TuneLens explores music through data — '
        'uncovering patterns across genres, artists, '
        'moods and audio characteristics.'
        '</div>'
        '<div class="hero-note">'
        '🎧 113K+ tracks • 114 genres • data-driven discovery'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    # =====================================================
    # DATA SNAPSHOT
    # =====================================================

    st.markdown(
        '<div class="section-title">Your Music Landscape</div>',
        unsafe_allow_html=True
    )

    total_tracks = len(df)
    total_artists = df["artists"].nunique()
    total_genres = df["track_genre"].nunique()
    avg_popularity = df["popularity"].mean()

    col1, col2, col3, col4 = st.columns(4)

    stats = [
        ("🎵", f"{total_tracks:,}", "Tracks analysed"),
        ("🎤", f"{total_artists:,}", "Artists"),
        ("🎼", f"{total_genres}", "Genres"),
        ("✨", f"{avg_popularity:.1f}", "Avg. popularity")
    ]

    for col, (icon, number, label) in zip(
        [col1, col2, col3, col4],
        stats
    ):

        with col:

            st.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-icon">{icon}</div>'
                f'<div class="stat-number">{number}</div>'
                f'<div class="stat-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


    # =====================================================
    # TRENDING SOUNDS
    # =====================================================

    st.markdown(
        '<div class="section-title">🔥 Trending Sounds</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Tracks with the highest popularity scores in the dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    popular = (
        df[
            [
                "track_name",
                "artists",
                "track_genre",
                "popularity"
            ]
        ]
        .sort_values(
            "popularity",
            ascending=False
        )
        .drop_duplicates(
            subset=["track_name", "artists"]
        )
        .head(6)
    )

    cols = st.columns(3)

    art_styles = [
        "linear-gradient(135deg,#7c3aed,#312e81)",
        "linear-gradient(135deg,#a855f7,#4338ca)",
        "linear-gradient(135deg,#4f46e5,#7c3aed)",
        "linear-gradient(135deg,#6d28d9,#c084fc)",
        "linear-gradient(135deg,#3730a3,#9333ea)",
        "linear-gradient(135deg,#581c87,#6366f1)"
    ]

    for i, (_, row) in enumerate(
        popular.iterrows()
    ):

        with cols[i % 3]:

            st.markdown(
                f'<div class="music-card">'
                f'<div class="album-art" style="background:{art_styles[i]}">♪</div>'
                f'<div class="music-title">{row["track_name"]}</div>'
                f'<div class="music-artist">{row["artists"]}</div>'
                f'<div class="music-meta">'
                f'{row["track_genre"]} &nbsp; • &nbsp; '
                f'Popularity {row["popularity"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


    # =====================================================
    # MOOD LANDSCAPE
    # =====================================================

    st.markdown(
        '<div class="section-title">🎭 Mood Landscape</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Explore how energy and positivity shape the dataset.'
        '</div>',
        unsafe_allow_html=True
    )

    moods = [
        ("⚡", "Energetic & Happy", "mood-1"),
        ("🌙", "Energetic & Moody", "mood-2"),
        ("☀️", "Calm & Positive", "mood-3"),
        ("🌌", "Calm & Moody", "mood-4")
    ]

    mood_cols = st.columns(4)

    for col, (icon, mood, style) in zip(
        mood_cols,
        moods
    ):

        with col:

            count = (
                df["mood"] == mood
            ).sum()

            st.markdown(
                f'<div class="mood-card {style}">'
                f'<div class="mood-icon">{icon}</div>'
                f'<div class="mood-name">{mood}</div>'
                f'<div class="mood-count">{count:,} tracks</div>'
                f'</div>',
                unsafe_allow_html=True
            )

# ---------------------------------------------------------
# DISCOVER PAGE
# ---------------------------------------------------------

elif page == "🎧 Discover":

    st.markdown(
        '<div class="main-title">Discover</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Find tracks that match your sound.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">🎧 Song Recommendations</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select a track and TuneLens will recommend songs "
        "with similar audio characteristics."
    )

    # Create unique track list
    track_options = (
        df[["track_name", "artists"]]
        .drop_duplicates()
        .sort_values("track_name")
    )

    selected_track = st.selectbox(
        "Choose a track",
        track_options["track_name"].tolist()
    )

    if st.button("✨ Find Similar Tracks"):

        recommendations = recommend_songs(
            selected_track,
            n=5
        )

        if isinstance(recommendations, str):

            st.warning(recommendations)

        else:

            st.markdown(
                '<div class="section-title">Recommended for You</div>',
                unsafe_allow_html=True
            )

            for _, song in recommendations.iterrows():

                st.markdown(
                    f'<div class="song-card">'
                    f'<div class="song-name">🎵 {song["track_name"]}</div>'
                    f'<div class="artist-name">{song["artists"]}</div>'
                    f'<div class="genre-tag">'
                    f'{song["track_genre"]} &nbsp; • &nbsp; '
                    f'Popularity {song["popularity"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.caption(
                "Recommendations are generated using similarity "
                "across audio features such as energy, danceability, "
                "valence, acousticness and tempo."
            )



# =========================================================
# MUSIC ANALYTICS
# =========================================================

elif page == "📊 Music Analytics":

    st.markdown(
        '<div class="main-title">📊 Music Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore patterns, characteristics and relationships across music data.'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # KEY INSIGHTS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Your Music Landscape</div>',
        unsafe_allow_html=True
    )

    top_genre = df["track_genre"].value_counts().idxmax()
    top_artist = df["artists"].value_counts().idxmax()

    avg_energy = df["energy"].mean()
    avg_danceability = df["danceability"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">🎼</div>
                <div class="metric-label">Top Genre</div>
                <div>{top_genre}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">🎤</div>
                <div class="metric-label">Most Represented Artist</div>
                <div>{top_artist}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{avg_energy:.2f}</div>
                <div class="metric-label">Average Energy</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">{avg_danceability:.2f}</div>
                <div class="metric-label">Average Danceability</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # TOP GENRES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎼 Genre Landscape</div>',
        unsafe_allow_html=True
    )

    top_genres = (
        df["track_genre"]
        .value_counts()
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        top_genres.index,
        top_genres.values
    )

    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    ax.set_title("Top 10 Genres")

    plt.tight_layout()

    st.pyplot(fig)

    # -----------------------------------------------------
    # AUDIO CHARACTERISTICS
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎚️ How Does the Music Sound?</div>',
        unsafe_allow_html=True
    )

    audio_features = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence"
    ]

    averages = (
        df[audio_features]
        .mean()
        .sort_values(
            ascending=True
        )
    )

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.barh(
        averages.index,
        averages.values
    )

    ax2.set_xlabel("Average Value")
    ax2.set_title("Average Audio Characteristics")

    plt.tight_layout()

    st.pyplot(fig2)

    # -----------------------------------------------------
    # MOOD DISTRIBUTION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">😊 Mood Landscape</div>',
        unsafe_allow_html=True
    )

    mood_counts = (
        df["mood"]
        .value_counts()
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.bar(
        mood_counts.index,
        mood_counts.values
    )

    ax3.set_ylabel("Number of Tracks")
    ax3.set_title("Music Distribution by Mood")

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    st.pyplot(fig3)

    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🔗 What Moves Together?</div>',
        unsafe_allow_html=True
    )

    correlation_features = [
        "popularity",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo"
    ]

    correlation_matrix = (
        df[correlation_features]
        .corr()
    )

    fig4, ax4 = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        ax=ax4
    )

    ax4.set_title(
        "Audio Feature Correlation"
    )

    plt.tight_layout()

    st.pyplot(fig4)

    # -----------------------------------------------------
    # POPULARITY VS ENERGY
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">⚡ Popularity vs Energy</div>',
        unsafe_allow_html=True
    )

    sample = df.sample(
        min(5000, len(df)),
        random_state=42
    )

    fig5, ax5 = plt.subplots(
        figsize=(10, 5)
    )

    ax5.scatter(
        sample["energy"],
        sample["popularity"],
        alpha=0.35
    )

    ax5.set_xlabel("Energy")
    ax5.set_ylabel("Popularity")
    ax5.set_title(
        "Relationship Between Energy and Popularity"
    )

    plt.tight_layout()

    st.pyplot(fig5)

    # -----------------------------------------------------
    # TUNELENS INSIGHT
    # -----------------------------------------------------

    most_common_mood = (
        df["mood"]
        .value_counts()
        .idxmax()
    )

    mood_percentage = (
        df["mood"].value_counts(normalize=True)
        .loc[most_common_mood] * 100
    )

    st.markdown(
        '<div class="section-title">💡 TuneLens Insight</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"{most_common_mood} is the most common mood category "
        f"in the dataset, representing approximately "
        f"{mood_percentage:.1f}% of analysed tracks."
    )