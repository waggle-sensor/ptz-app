import json
import os

LISTS_FILE_PATH = "species_lists/species_lists.json"

def _load_species_lists() -> dict:
    """Loads all species lists from a single JSON file."""
    if not os.path.exists(LISTS_FILE_PATH):
        return {}
    with open(LISTS_FILE_PATH, 'r') as f:
        data = json.load(f)
    # Convert lists to sets for faster lookups
    data['invasive_species'] = set(data.get('invasive_species', []))
    data['rare_species'] = set(data.get('rare_species', []))
    return data

def check_for_alert(species_name: str) -> tuple:
    """
    Checks if a species is on a list.
    Returns a tuple of (alert_type, species_name) or (None, None).
    """
    all_lists = _load_species_lists()
    
    if species_name in all_lists.get('invasive_species', set()):
        return ("invasive_species", species_name)
    
    if species_name in all_lists.get('rare_species', set()):
        return ("rare_species", species_name)
        
    return (None, None)