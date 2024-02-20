from flask import Flask, redirect, jsonify, session, request 
from dotenv import load_dotenv
from flask import render_template
from datetime import datetime
import urllib.parse
import requests
import sqlite3
import json
import os

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import numpy as np
import pandas as pd

# --------------------------- Setup --------------------------- 
app = Flask(__name__)
app.secret_key = 'ELxq2ff1wBqgZlUArbGOrVNzWhRwluIl'

load_dotenv()

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = 'http://localhost:5000/callback'

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1/'

playlist_URI = 'spotify:playlist:1PNVazAHdA7oOnFrmLe3rV'
playlist_id = '1PNVazAHdA7oOnFrmLe3rV'
user_id = ''

# Connect to SQLite database (or create it if it doesn't exist)
db_path = 'spotify_tracks.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the table (do this once)
cursor.execute('''CREATE TABLE IF NOT EXISTS tracks
                  (spotify_id TEXT PRIMARY KEY, name TEXT, audio_features TEXT)''')

# --------------------------- End of Setup --------------------------- 

@app.route('/')
def index():
    return "Welcome <a href='/login'>Login with Spotify</a>"

@app.route('/login')
def login():
    scope = 'user-read-private user-read-email playlist-read-private playlist-modify-private playlist-modify-public'

    params = {
        'client_id': CLIENT_ID, 
        'response_type': 'code',
        'scope': scope,
        'redirect_uri': REDIRECT_URI,
        'show_dialog': True
    }

    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}" 

    return redirect(auth_url)

@app.route('/callback')
def callback():
    if 'error' in request.args:
        return jsonify({"error": request.args['error']})
    
    if 'code' in request.args:
        req_body = {
            'code': request.args['code'],
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID, 
            'client_secret': CLIENT_SECRET
        }
    
    response = requests.post(TOKEN_URL, data=req_body)
    token_info = response.json()

    session['access_token'] = token_info['access_token']
    session['refresh_token'] = token_info['refresh_token']
    session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']

    return render_template('options.html')
    # return redirect('/playlists')


@app.route('/playlists')
def get_playlists():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp()>session['expires_at']:
        redirect('/refresh-token')

    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }

    response = requests.get(API_BASE_URL + 'me/playlists/' , headers=headers)
    playlists = response.json()

    return jsonify(playlists)

@app.route('/action', methods=['POST'])
def action():
    # selected_playlist = request.form['button']
    # print("Selected option: ",selected_playlist)
    # Your Python code based on the selected option
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp()>session['expires_at']:
        redirect('/refresh-token')

    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }
    
    response = requests.get(API_BASE_URL + 'me/playlists/' , headers=headers)
    if response.status_code == 200:
        playlists_data = response.json()
        playlists_dict = {playlist['name']: playlist['id'] for playlist in playlists_data['items']}
        print(playlists_dict)
    else:
        playlists_dict = {'Error': 'Failed to retrieve playlists'}
    
    # selected_playlist_id = playlists_dict[selected_playlist]
    # print("Selected playlist id: ",selected_playlist_id)

    return render_template('playlists.html', playlists=playlists_dict)
    # playlists = response.json()

    # return jsonify(playlists)

# ---------------------- database stuff ----------------------

# Function to insert a track
def insert_track(spotify_id, name, audio_features):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    audio_features_json = json.dumps(audio_features)  # Convert dict to JSON string
    # Use INSERT OR IGNORE to avoid duplicates
    cursor.execute('INSERT OR IGNORE INTO tracks (spotify_id, name, audio_features) VALUES (?, ?, ?)',
                   (spotify_id, name, audio_features_json))
    conn.commit()
    conn.close()

# Function to retrieve a track's name and audio features by Spotify ID
def get_track(spotify_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT name, audio_features FROM tracks WHERE spotify_id = ?', (spotify_id,))
    result = cursor.fetchone()
    if result:
        name, audio_features_json = result
        audio_features = json.loads(audio_features_json)  # Convert JSON string back to dict
        return name, audio_features
    else:
        return None
    conn.close()
    
def display_all_tracks(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # SQL query to select all from a specific table
    query = "SELECT * FROM tracks"
    cursor.execute(query)

    # Fetch and print all rows of the query result
    rows = cursor.fetchall()
    for row in rows:
        spotify_id, name, audio_features_json = row
        audio_features = json.loads(audio_features_json)  # Convert JSON string back to a dictionary
        print(f"Spotify ID: {spotify_id}, Name: {name}, Audio Features: {audio_features}")

    # Close the connection
    conn.close()
    

def retrieve_db_pd(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT spotify_id, audio_features FROM tracks"
    df = pd.read_sql_query(query, conn, index_col='spotify_id')
    df['audio_features'] = df['audio_features'].apply(json.loads)
    conn.close()
    print(df.head())
    return df

# ---------------------- end of database stuff ----------------------

# ---------------------- ML -----------------------------------------
def visualize_clusters(df, centroids, filename='clusters_visualization.png'):
    features_df = pd.json_normalize(df['audio_features'])
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features_df)
    reduced_centroids = pca.transform(centroids)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df['cluster_label'], cmap='viridis', marker='o', edgecolor='k', s=70, alpha=0.5)
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='red', marker='X', s=200)
    plt.title('Clusters Visualization with PCA')
    plt.xlabel('PCA Feature 1')
    plt.ylabel('PCA Feature 2')

    # Save the plot to a file
    plt.savefig('clustering_plot.png')
    plt.close()  # Close the figure to free memory

def find_closest_tracks(df, features, centroids, n_closest=5):
    # Calculate the distance from each track to each centroid
    distances = euclidean_distances(features, centroids)
    
    # For each centroid, find the indices of the n closest tracks
    closest_tracks_indices = np.argsort(distances, axis=0)[:n_closest]
    
    # Create a dictionary to hold the closest tracks for each centroid
    closest_tracks = {}
    for centroid_index in range(centroids.shape[0]):
        closest_tracks_for_centroid = df.iloc[closest_tracks_indices[:, centroid_index]].index.tolist()
        closest_tracks[centroid_index] = closest_tracks_for_centroid
    
    return closest_tracks

def perform_kmeans_clustering(db_path, n_clusters=5):
    df = retrieve_db_pd(db_path)  # Assuming this function returns a DataFrame with Spotify IDs as index and features ready for clustering

    # Assuming 'audio_features' are the columns you want to include in the clustering
    # You may need to adjust this to fit your actual DataFrame structure
    features = pd.json_normalize(df['audio_features'])

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    
    # Assign labels back to the original DataFrame
    df['cluster_label'] = kmeans.labels_

    # Optionally, return centroids for new song suggestions
    centroids = kmeans.cluster_centers_

    visualize_clusters(df, centroids)

    # these 5 closest tracks will be used later to seed the spotify song reccomender
    closest_tracks = find_closest_tracks(df, features, centroids, n_closest=5)

    return df, centroids, closest_tracks

# ---------------------- End of ML ----------------------------------

def generate_recommendation_urls(centroids, seed_tracks, feature_names):
    base_url = "https://api.spotify.com/v1/recommendations?"
    seed_tracks_param = "seed_tracks=" + "+".join([str(track_id) for track_id in seed_tracks])

    urls = []

    for centroid in centroids:
        # Map centroid features (numerical values) to their names
        features_dict = dict(zip(feature_names, centroid))
        # Convert feature values to strings and format the URL
        features_params = "&".join([f"target_{feature}={value}" for feature, value in features_dict.items()])
        url = f"{base_url}{seed_tracks_param}&{features_params}"
        urls.append(url)

    return urls



@app.route('/playlist_selection', methods=['POST'])
def handle_playlist_selection():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp()>session['expires_at']:
        redirect('/refresh-token')

    headers = {
        'Authorization': f"Bearer {session['access_token']}"
    }

    playlist_id = request.form.get('playlist_id')
    playlist_name = request.form.get('playlist_name')
    response = API_BASE_URL + '/playlists/' + playlist_id + '/tracks'
    # print("response: ",response)
    response = requests.get(API_BASE_URL + 'playlists/' + playlist_id + '/tracks', headers=headers)
    playlist = response.json()

    tracks = playlist.get('items', [])
    track_features = {}

    for track in tracks:
        track_data = track['track']
        spotify_id = track_data['id']
        track_name = track_data['name'] 
        # req = 'f"{API_BASE_URL}audio-features/{spotify_id}", headers=headers'
        # print("req: ", req)
        # response = requests.get(f"{API_BASE_URL}audio-features/{spotify_id}", headers=headers)
        response = requests.get(API_BASE_URL + 'audio-features/' + spotify_id, headers=headers)
        
        if response.status_code == 200:
            audio_features = response.json()
            excluded_fields = ['analysis_url', 'track_href', 'type', 'uri', 'id']
            audio_features_filtered = {k: v for k, v in audio_features.items() if k not in excluded_fields}
            track_features[spotify_id] = {'name': track_name, 'audio_features': audio_features_filtered}
            insert_track(spotify_id, track_name, audio_features_filtered)
        else:
            print(f"Failed to fetch audio features for track ID {spotify_id}. ({response.status_code})")

    # return jsonify(playlist)
    n_clusters = 3
    display_all_tracks(db_path)
    df, centroids, seed_tracks = perform_kmeans_clustering(db_path, n_clusters)

    print("seed tracks: ", seed_tracks)

    feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
    urls = generate_recommendation_urls(centroids, seed_tracks, feature_names)
    
    uris_list = []

    for url in urls:
        response = requests.get(url, headers=headers)
        recommendations = response.json()
        return recommendations
        track_ids = [track["id"] for track in recommendations["tracks"]]
        
        # Format and append each track ID to the uris_list
        for tr_id in track_ids:
            uris_list.append(f"spotify:track:{tr_id}")

    new_playlist_id = create_playlist()

    # Create the final JSON object
    tracks_json = {"uris": uris_list}
    req_body = json.dumps({"uris": uris_list})
    print("Request body for adding items to playlist: ", req_body)
    response = requests.post(API_BASE_URL + 'playlists/' + new_playlist_id + '/tracks', json=req_body, headers=headers)

    return tracks_json


@app.route('/create_playlist')
def create_playlist():
    if 'access_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')  # Ensure this redirect occurs

    req_body = {
        "name": "Playlist Based On: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "New playlist description",
        "public": False
    }

    headers = {'Authorization': f"Bearer {session['access_token']}"}

    response = requests.get(API_BASE_URL + 'me/', headers=headers)
    if response.status_code == 200:
        user_id = response.json().get('id')
        response = requests.post(API_BASE_URL + 'users/' + user_id + '/playlists', json=req_body, headers=headers)
        if response.status_code == 201:
            new_playlist_id = response.json().get('id')
            return new_playlist_id  # Return the new playlist ID
        else:
            return "Error creating playlist"
    else:
        return "Error: Failed to retrieve user ID"

# @app.route('/create_playlist')
# def create_playlist():
#     if 'access_token' not in session:
#         return redirect('/login')
    
#     if datetime.now().timestamp()>session['expires_at']:
#         redirect('/refresh-token')

#     req_body = {
#             "name": "Playlist Based On: ",
#             "description": "New playlist description",
#             "public": False
#         }

#     headers = {
#         'Authorization': f"Bearer {session['access_token']}"
#     }

#     response = requests.get(API_BASE_URL + 'me/' , headers=headers)
#     if response.status_code == 200:
#         user_profile = response.json()
#         user_id = user_profile.get('id', 'Unknown')
#     else:
#         user_id = 'Error: Failed to retrieve user ID'

#     # print("user_id:", user_id)
#     # SPOTIFY_CREATE_PLAYLIST_URL = API_BASE_URL + 'users/' + user_id + '/playlists'
#     # print("SPOTIFY_CREATE_PLAYLIST_URL", SPOTIFY_CREATE_PLAYLIST_URL)
#     response = requests.post(API_BASE_URL + 'users/' + user_id + '/playlists', json=req_body, headers=headers)
#     print("Create playlist response code: ",response.status_code)
#     if response.status_code == 201:
#         create_playlists_data = response.json()
#         new_playlist_id = create_playlists_data.get('id', 'Unknown ID')
#         print("New playlist id: ", new_playlist_id)
#     else:
#         playlists_dict = {'Error': 'Failed to retrieve playlists'}
#         print("error creating playlist")

#     return new_playlist_id

# def generate_songs():


@app.route('/refresh-token')
def refresh_token():
    if 'refresh_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp()>session['expires_at']:
        req_body = {
            'grant_type': 'refresh_token',
            'refresh_token': session['refresh_token'],
            'client_id': CLIENT_ID, 
            'client_secret': CLIENT_SECRET
        }

    response = requests.post(TOKEN_URL, data=req_body)
    new_token_info = response.json()

    session['access_token'] = new_token_info['access_token']  
    session['expires_at'] = datetime.now().timestamp() + new_token_info ['expires_in']
    return redirect('/playlists')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

conn.close()