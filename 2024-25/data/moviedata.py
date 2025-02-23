import requests
import time
import json

with open("/Users/Flint/Data/apikeys/keys.json", 'r') as keyfile:
    keys = json.load(keyfile)

API_KEY = keys['TMDB']
BASE_URL = "https://api.themoviedb.org/3"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json;charset=utf-8"
}

def get_tv_series_popular(pages=5):
    series_list = []
    for page in range(1, pages+1):
        url = f"{BASE_URL}/tv/popular?api_key={API_KEY}&language=en-US&page={page}"
        response = requests.get(url, headers=headers)
        data = response.json()
        for tv in data['results']:
            series_list.append({
                "id": tv["id"],
                "title": tv["name"],
                "overview": tv["overview"],
                "rating": tv["vote_average"],
                "first_air_date": tv["first_air_date"],
                "genres": tv["genre_ids"],
                "countries": tv["origin_country"]
            })
        time.sleep(0.5)
    return series_list

def get_tv_series_top_rated(pages=5):
    series_list = []
    for page in range(1, pages+1):
        url = f"{BASE_URL}/tv/top_rated?api_key={API_KEY}&language=en-US&page={page}"
        response = requests.get(url, headers=headers)
        data = response.json()
        for tv in data['results']:
            series_list.append({
                "id": tv["id"],
                "title": tv["name"],
                "overview": tv["overview"],
                "rating": tv["vote_average"],
                "first_air_date": tv["first_air_date"],
                "genres": tv["genre_ids"],
                "countries": tv["origin_country"]
            })
        time.sleep(0.5)
    return series_list

def get_tv_episodes(tv_id, num_seasons=3):
    episodes_list = []
    for season in range(1, num_seasons+1):
        url = f"{BASE_URL}/tv/{tv_id}/season/{season}?api_key={API_KEY}&language=en-US"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            continue  # Salta se la stagione non esiste
        season_data = response.json()
        for episode in season_data["episodes"]:
            episodes_list.append({
                "tv_id": tv_id,
                "season": season,
                "episode": episode["episode_number"],
                "title": episode["name"],
                "overview": episode["overview"],
                "air_date": episode["air_date"],
                "rating": episode["vote_average"]
            })
        time.sleep(0.5)
    return episodes_list

def get_tv_reviews(tv_id):
    url = f"{BASE_URL}/tv/{tv_id}/reviews?api_key={API_KEY}&language=en-US"
    response = requests.get(url, headers=headers)
    data = response.json()
    reviews_list = []
    for review in data.get("results", []):
        reviews_list.append({
            "tv_id": tv_id,
            "author": review["author"],
            "content": review["content"],
            "rating": review.get("author_details", {}).get("rating", None),
            "created_at": review["created_at"]
        })
    return reviews_list

