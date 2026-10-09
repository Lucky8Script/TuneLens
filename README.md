# TuneLens — Music Analytics & Recommendation Platform

TuneLens is a Python-based music analytics application that explores music data through interactive visualizations, mood classification, and content-based song recommendations.

## Features

* **Data Cleaning & Analysis:** Processes a dataset of 113K+ tracks, handling missing values and duplicate records.
* **Music Analytics:** Explores genre distribution, artist trends, popularity, and audio characteristics through visualizations.
* **Mood Classification:** Categorizes tracks into four mood groups using energy and valence.
* **Song Recommendations:** Uses standardized audio features and cosine similarity to identify tracks with similar characteristics.
* **Interactive Dashboard:** Provides a Streamlit interface for exploring music data and discovering tracks.

## Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Streamlit

## Dataset

The project uses the Spotify Tracks Dataset, containing music metadata and audio features such as energy, danceability, valence, acousticness, and tempo.

The cleaned dataset contains approximately 113K tracks across 114 genres.

## Run Locally

1. Clone this repository:

   ```bash
   git clone YOUR_REPOSITORY_URL
   cd TuneLens
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place the dataset at `data/spotify_tracks.csv`.

4. Launch the application:

   ```bash
   streamlit run app.py
   ```

## Project Objective

TuneLens combines exploratory data analysis, data visualization, and similarity-based recommendations in an interactive application. It demonstrates how Python can be used to transform raw data into useful insights and an engaging user experience.

## Future Improvements

* Integrate a database for structured music data storage.
* Add more advanced search and filtering options.
* Explore automated insight generation.
