import requests
import pandas as pd
import os
import time

# Path to the existing CSV file
csv_path = "restaurants_italy.csv"

# Your Google Places API Key
GOOGLE_PLACES_API_KEY = 'AIzaSyBTcDmv_SM5Cf0sQlzCc9gidLhhpqYn0nc'  # Replace with your actual API key

# Function to fetch restaurants in Italy using Google Places API (with pagination)
def fetch_restaurants_from_google(name, page_token=None):
    # Coordinates for the center of Italy (Rome)
    italy_lat = 41.9028   # Latitude for Rome, Italy
    italy_lng = 12.4964   # Longitude for Rome, Italy
    radius = 500000       # 500 km radius

    # Google Places Text Search API URL
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    
    # Parameters for the API request
    params = {
        "query": name + " restaurant",  # Search term: restaurant name + "restaurant"
        "key": GOOGLE_PLACES_API_KEY,   # Your API key
        "location": f"{italy_lat},{italy_lng}",  # Location set to Italy's coordinates
        "radius": radius,               # Large radius to cover all of Italy
        "language": "it"                # Force the response language to Italian
    }
    
    # If there's a next page, use the page_token to get more results
    if page_token:
        params["pagetoken"] = page_token
    
    # Make the request
    response = requests.get(url, params=params)
    
    # Parse the response
    if response.status_code == 200:
        data = response.json()
        return data
    return None

# Function to handle API responses and pagination
def get_all_restaurants(name):
    all_restaurants = []
    page_token = None
    
    while True:
        # Fetch restaurants from Google Places API
        data = fetch_restaurants_from_google(name, page_token)
        
        if data:
            results = data.get("results", [])
            for result in results:
                all_restaurants.append({
                    "Name": result.get("name", "N/A"),
                    "Address": result.get("formatted_address", "N/A"),
                    "Type": "Restaurant",  # Hardcoded as "Restaurant"
                    "Popularity": result.get("user_ratings_total", "N/A")
                })
            
            # Check if there is another page of results
            page_token = data.get("next_page_token")
            if not page_token:
                break
            time.sleep(2)  # Wait for 2 seconds before requesting the next page (Google Places pagination delay)
        else:
            break
    
    return all_restaurants

# Function to append new data to the CSV file
def append_to_csv(new_data):
    # Check if the file exists
    if os.path.exists(csv_path):
        # Read the existing CSV file, but handle empty file case
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            # Handle the case where the CSV file is empty, create an empty DataFrame
            existing_df = pd.DataFrame(columns=["Name", "Address", "Type", "Profit", "Popularity"])
    else:
        # If the file doesn't exist, create a new DataFrame with the necessary columns
        existing_df = pd.DataFrame(columns=["Name", "Address", "Type", "Profit", "Popularity"])

    # Convert new data to a DataFrame
    new_df = pd.DataFrame(new_data)
    
    # Append new data to the existing DataFrame
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates based on 'Name' and 'Address'
    combined_df = combined_df.drop_duplicates(subset=["Name", "Address"])

    # Save the updated DataFrame to the CSV
    combined_df.to_csv(csv_path, index=False)
    print(f"Data appended successfully. The updated file is saved as {csv_path}")

# Main function to process and collect restaurant data
def main():
    # Example restaurant names list (replace this with actual names)
    restaurant_names = ["La Trattoria", "Pizzeria Roma", "Osteria del Mare"]
    
    all_rows = []
    for name in restaurant_names:
        print(f"Fetching data for: {name}")
        
        # Fetch all restaurants for this name
        restaurants = get_all_restaurants(name)
        
        if restaurants:
            all_rows.extend(restaurants)

    # If new data exists, append to CSV
    if all_rows:
        append_to_csv(all_rows)
    else:
        print("No new data to append.")

# Run the script
main()
