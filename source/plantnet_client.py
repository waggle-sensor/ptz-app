import requests
import json

# You will get your own API key from the my.plantnet.org website
PLANTNET_API_KEY = "2b10tSubhbpUaT0XF3sNpl0hYe"
PLANTNET_API_URL = "https://my-api.plantnet.org/v2/identify/all"

def identify_plant(image_path: str) -> dict:
    """
    Identifies a plant species from an image using the PlantNet API.
    """
    with open(image_path, 'rb') as image_file:
        image_data = {'images': ('image.jpg', image_file)}
        #payload = {'organs': ['flower', 'leaf', 'fruit']} # Tell PlantNet what parts to look for
        payload = {'organs': ['auto']}
        
        # Make the API request
        req = requests.post(PLANTNET_API_URL, params={'api-key': PLANTNET_API_KEY},
                            files=image_data, data=payload)
        
        if not req.ok:
            raise Exception(f"PlantNet API request failed with status {req.status_code}: {req.text}")
            
        json_results = req.json()
        
        results = json_results.get('results', [])
        if results: # Check if the results list is not empty
            top_result = results[0]
            return {
                "species": top_result['species']['scientificNameWithoutAuthor'],
                "common_names": top_result['species']['commonNames'],
                "score": round(top_result['score'], 4)
            }
        return {} # Return empty dict if no results were found