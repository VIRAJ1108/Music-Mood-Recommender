import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    df = pd.read_csv("dataset.csv")  # path to your dataset
    return df

df = load_data()

feature_cols = [
    'danceability', 'energy', 'valence', 'tempo', 'acousticness',
    'instrumentalness', 'speechiness', 'liveness', 'loudness'
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

kmeans = KMeans(n_clusters=4, random_state=42)
valence_energy = df[['valence', 'energy']]
valence_energy_scaled = StandardScaler().fit_transform(valence_energy)
clusters = kmeans.fit_predict(valence_energy_scaled)
df['cluster'] = clusters

def map_clusters_to_moods(df):
    mood_map = {}
    stats = df.groupby('cluster')[['valence', 'energy']].median()
    
    for i, row in stats.iterrows():
        val, ener = row['valence'], row['energy']
        if val >= 0.6 and ener >= 0.6:
            mood_map[i] = "Happy / Energetic"
        elif val >= 0.6 and ener < 0.6:
            mood_map[i] = "Happy / Calm"
        elif val < 0.6 and ener >= 0.6:
            mood_map[i] = "Angry / Energetic"
        else:
            mood_map[i] = "Sad / Calm"
    return mood_map

mood_map = map_clusters_to_moods(df)
df['mood_label'] = df['cluster'].map(mood_map)

def get_context_weights(activity):
    weights = {f: 1.0 for f in feature_cols}
    
    if activity == 'Workout':
        weights['energy'] = 1.3
        weights['tempo'] = 1.2
        weights['danceability'] = 1.1
    elif activity == 'Studying':
        weights['acousticness'] = 1.2
        weights['instrumentalness'] = 1.3
        weights['speechiness'] = 0.7
    elif activity == 'Relaxing':
        weights['acousticness'] = 1.3
        weights['energy'] = 0.8
        
    return weights

def recommend_songs(user_mood, activity, top_k=10):
    # Filter songs with matching mood
    cluster_ids = [k for k, v in mood_map.items() if user_mood in v]
    candidate_songs = df[df['cluster'].isin(cluster_ids)].reset_index(drop=True)

    # Compute prototype
    prototype = candidate_songs[feature_cols].median().copy()

    # Adjust prototype using context weights
    context_weights = get_context_weights(activity)
    for f, w in context_weights.items():
        prototype[f] *= w

    # Compute similarity
    proto_scaled = scaler.transform([prototype.values])
    sims = cosine_similarity(proto_scaled, X_scaled).flatten()

    # Select top-K
    top_indices = sims.argsort()[-top_k:][::-1]
    return df.iloc[top_indices][['track_name', 'artists', 'mood_label', 'energy', 'valence', 'tempo']]



st.set_page_config(page_title="🎧 Music Mood Recommender", layout="centered")

st.title("🎧 Music for Mood")
st.markdown("Select your mood and activity to get personalized song recommendations.")

# User Inputs
mood = st.selectbox(
    "Select your Mood:",
    ["Happy / Energetic", "Happy / Calm", "Angry / Energetic", "Sad / Calm"]
)

activity = st.selectbox(
    "Select your Activity:",
    ["Workout", "Studying", "Relaxing"]
)

top_k = st.slider("Number of Recommendations:", 5, 20, 10)

if st.button("🎵 Get Recommendations"):
    recommendations = recommend_songs(mood, activity, top_k)
    st.success(f"Here are your top {top_k} recommendations:")
    st.dataframe(recommendations, use_container_width=True)
