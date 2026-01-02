"""Radio 357 Top Songs YouTube Views Analysis

This script:
1. Loads the Radio 357 top songs list from cache
2. Looks up YouTube view counts for each song using yt-dlp (no API key needed!)
3. Fetches YouTube upload date
4. Loads original release years from release_years.csv (manual lookup)
5. Creates visualizations using pandas and seaborn
6. Saves all output to a timestamped folder for historical tracking
"""

import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configuration
BASE_DIR = Path(__file__).parent
CACHE_FILE = BASE_DIR / "songs_cache.json"
MAX_WORKERS = 5  # Parallel requests (be gentle to YouTube)

# Create timestamped output directory
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = BASE_DIR / f"run_{RUN_TIMESTAMP}"
VIEWS_CACHE_FILE = OUTPUT_DIR / "songs_with_views.json"
RELEASE_YEARS_FILE = BASE_DIR / "release_years.csv"


def load_release_years() -> dict[int, int]:
    """
    Load original release years from CSV file.
    
    The CSV should have columns: rank, artist, title, original_release_year
    Returns a dict mapping rank -> release_year
    """
    release_years = {}
    
    if not RELEASE_YEARS_FILE.exists():
        print(f"Note: {RELEASE_YEARS_FILE} not found. Release years will be empty.")
        return release_years
    
    try:
        import csv
        with open(RELEASE_YEARS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rank = int(row['rank'])
                year_str = row.get('original_release_year', '').strip()
                if year_str:
                    try:
                        release_years[rank] = int(year_str)
                    except ValueError:
                        pass  # Skip invalid years
    except Exception as e:
        print(f"Warning: Could not load release years: {e}")
    
    return release_years


def get_youtube_info_ytdlp(artist: str, title: str, num_results: int = 3) -> dict:
    """
    Get YouTube info (views, URL, upload date) for a song using yt-dlp.
    
    Searches for top N results and returns the one with the highest view count.
    
    Args:
        artist: Artist name
        title: Song title
        num_results: Number of search results to consider (default: 3)
    
    Returns:
        Dict with view_count, video_url, upload_date, video_title
    """
    # Search without "official" to get broader results, then pick highest views
    query = f"ytsearch{num_results}:{artist} {title}"
    result_data = {
        'view_count': None,
        'video_url': None,
        'upload_date': None,
        'video_title': None
    }
    
    try:
        # Get flat playlist to quickly compare view counts
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", "--flat-playlist", query],
            capture_output=True,
            text=True,
            timeout=60,
            encoding='utf-8'
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Parse multiple JSON objects (one per line)
            candidates = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        view_count = data.get('view_count')
                        if view_count is not None:
                            candidates.append(data)
                    except json.JSONDecodeError:
                        continue
            
            if candidates:
                # Pick the video with the highest view count
                best = max(candidates, key=lambda x: x.get('view_count', 0) or 0)
                
                result_data['view_count'] = best.get('view_count')
                result_data['video_url'] = best.get('webpage_url') or best.get('url')
                result_data['video_title'] = best.get('title')
                
                # Get upload date (format: YYYYMMDD)
                upload_date_raw = best.get('upload_date')
                if upload_date_raw and len(upload_date_raw) == 8:
                    result_data['upload_date'] = f"{upload_date_raw[:4]}-{upload_date_raw[4:6]}-{upload_date_raw[6:8]}"
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout for: {artist} - {title}")
    except json.JSONDecodeError:
        print(f"  JSON error for: {artist} - {title}")
    except Exception as e:
        print(f"  Error for {artist} - {title}: {e}")
    
    return result_data



def fetch_all_data(songs: list[dict], max_workers: int = MAX_WORKERS) -> list[dict]:
    """
    Fetch YouTube data for all songs.
    Release years are loaded from release_years.csv (manual lookup).
    """
    total = len(songs)
    
    # Load release years from CSV
    release_years = load_release_years()
    years_loaded = len(release_years)
    
    def fetch_youtube(song):
        yt_info = get_youtube_info_ytdlp(song['artist'], song['title'])
        return {
            **song,
            'youtube_views': yt_info['view_count'],
            'youtube_url': yt_info['video_url'],
            'youtube_upload_date': yt_info['upload_date'],
            'youtube_video_title': yt_info['video_title'],
            'original_release_year': release_years.get(song['rank'])
        }
    
    results = []
    
    print("\nFetching YouTube data (searching top 3 results, picking highest views)...")
    if years_loaded > 0:
        print(f"Release years loaded from CSV: {years_loaded} songs\n")
    else:
        print("Note: Fill in release_years.csv to add original release years.\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_youtube, song): i for i, song in enumerate(songs)}
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['youtube_views']:
                    year_info = f", released: {result['original_release_year']}" if result['original_release_year'] else ""
                    print(f"[{len(results)}/{total}] {result['artist']} - {result['title']}: {result['youtube_views']:,} views{year_info}")
                else:
                    print(f"[{len(results)}/{total}] {result['artist']} - {result['title']}: Not found")
                    
            except Exception as e:
                song = songs[idx]
                print(f"[{len(results)}/{total}] Error for {song['artist']}: {e}")
                results.append({
                    **song, 
                    'youtube_views': None, 
                    'youtube_url': None,
                    'youtube_upload_date': None,
                    'youtube_video_title': None,
                    'original_release_year': release_years.get(song['rank'])
                })
    
    # Sort by rank
    results.sort(key=lambda x: x['rank'])
    
    return results


def main():
    """Main entry point."""
    
    # Create timestamped output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Radio 357 Top Songs - YouTube Data Collection")
    print(f"Run timestamp: {RUN_TIMESTAMP}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("="*60)
    
    # Load base song data
    songs = load_cache(CACHE_FILE)
    
    if not songs:
        print("ERROR: No song cache found. Please run scraping first.")
        return
    
    print(f"\nLoaded {len(songs)} songs from cache.")
    print("Fetching YouTube data + original release years...")
    
    songs_with_data = fetch_all_data(songs)
    
    # Save to views cache (JSON)
    save_cache(songs_with_data, VIEWS_CACHE_FILE)
    print(f"\nData cached to {VIEWS_CACHE_FILE}")
    
    # Convert to DataFrame
    df = pd.DataFrame(songs_with_data)
    
    # Save to CSV
    csv_path = OUTPUT_DIR / 'radio357_songs_with_views.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"CSV saved to {csv_path}")
    
    # Save run metadata
    metadata = {
        'run_timestamp': RUN_TIMESTAMP,
        'total_songs': len(df),
        'songs_with_views': int(df['youtube_views'].notna().sum()),
        'songs_with_release_year': int(df['original_release_year'].notna().sum()),
        'data_collected_at': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Data is ready in: {OUTPUT_DIR}")
    print("\nTo analyze this data and create visualizations, run:")
    print(f"python radio357_analyze.py --run run_{RUN_TIMESTAMP}")


if __name__ == "__main__":
    main()
