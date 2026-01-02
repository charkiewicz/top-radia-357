"""Radio 357 Songs Data Analysis

This script performs data analysis and visualization on already-collected song data.
It reads from an existing data file (songs_with_views.json or CSV) and creates visualizations.

This is decoupled from data collection - run radio357_youtube_analysis.py first to fetch data,
then use this script to iterate on visualizations without re-fetching.

Usage:
    python radio357_analyze.py                          # Uses most recent run
    python radio357_analyze.py --data path/to/data.json # Use specific data file
    python radio357_analyze.py --run run_2026-01-02_10-34-18  # Use specific run folder
"""

import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import numpy as np

# Configuration
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "analysis_output"


def find_latest_run() -> Path | None:
    """Find the most recent run folder."""
    run_dirs = sorted(BASE_DIR.glob("run_*"), reverse=True)
    if run_dirs:
        return run_dirs[0]
    return None


def load_data(data_path: Path) -> pd.DataFrame:
    """Load song data from JSON or CSV file."""
    if data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif data_path.suffix == '.csv':
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")


def load_release_years_from_csv() -> dict[int, int]:
    """Load release years from CSV."""
    csv_file = BASE_DIR / "release_years.csv"
    if not csv_file.exists():
        return {}
    
    try:
        import csv
        years = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('rank') and row.get('original_release_year'):
                    years[int(row['rank'])] = int(row['original_release_year'])
        return years
    except:
        return {}


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare DataFrame for analysis with derived columns."""
    df = df.copy()
    
    # Merge release years from CSV if missing
    csv_years = load_release_years_from_csv()
    if csv_years:
        def get_year(row):
            val = row.get('original_release_year')
            # If it's effectively null (None, NaN, or 0)
            if pd.isna(val) or val is None or val == 0:
                return csv_years.get(row['rank'])
            return val
        df['original_release_year'] = df.apply(get_year, axis=1)

    # Add views in millions
    if 'youtube_views' in df.columns:
        df['youtube_views'] = pd.to_numeric(df['youtube_views'], errors='coerce')
        df['youtube_views_millions'] = df['youtube_views'] / 1_000_000
    
    # Calculate days since upload and views per day
    if 'youtube_upload_date' in df.columns:
        # Convert to datetime, handling Nones
        df['upload_dt'] = pd.to_datetime(df['youtube_upload_date'], errors='coerce')
        today = datetime.now()
        df['days_since_upload'] = (today - df['upload_dt']).dt.days
        # Minimum 1 day to avoid division by zero
        df['days_since_upload'] = df['days_since_upload'].clip(lower=1)
        df['views_per_day'] = df['youtube_views'] / df['days_since_upload']
    
    # Add rank bins
    if 'rank' in df.columns:
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        df['rank_bin'] = pd.cut(
            df['rank'],
            bins=[0, 10, 25, 50, 100, 200, 357],
            labels=['Top 10', '11-25', '26-50', '51-100', '101-200', '201-357']
        )
    
    # Add decade
    if 'original_release_year' in df.columns:
        df['original_release_year'] = pd.to_numeric(df['original_release_year'], errors='coerce')
        df['decade'] = (df['original_release_year'] // 10 * 10).astype('Int64')
    
    # Calculate Taste Gap
    if 'rank' in df.columns and 'youtube_views' in df.columns:
        df_gap = df.dropna(subset=['rank', 'youtube_views']).copy()
        if not df_gap.empty:
            max_rank = df_gap['rank'].max()
            df['norm_rank_score'] = 1 - (df['rank'] - 1) / (max_rank - 1)
            
            import numpy as np
            v = df['youtube_views'].fillna(1).clip(lower=1)
            log_v = np.log10(v)
            df['norm_view_score'] = (log_v - log_v.min()) / (log_v.max() - log_v.min())
            df['taste_gap'] = df['norm_view_score'] - df['norm_rank_score']
    
    return df


def create_basic_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create basic visualizations of the song data."""
    # Set seaborn style
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 10
    
    # Filter out songs without view data
    df_with_views = df[df['youtube_views'].notna()].copy()
    
    if len(df_with_views) == 0:
        print("No songs with YouTube view data to visualize!")
        return
    
    print(f"\nCreating visualizations with {len(df_with_views)} songs...")
    
    # 1. Scatter plot: Rank vs YouTube Views
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(
        data=df_with_views,
        x='rank',
        y='youtube_views_millions',
        alpha=0.6,
        s=100,
        ax=ax
    )
    ax.set_xlabel('Rank in Radio 357 Top (Left is better)', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions)', fontsize=12)
    ax.set_title('Radio 357 Top Songs: Rank vs YouTube Views', fontsize=14, fontweight='bold')
    
    # Add labels for top 10 and most viewed
    top_10 = df_with_views.nsmallest(10, 'rank')
    most_viewed = df_with_views.nlargest(5, 'youtube_views')
    labeled = pd.concat([top_10, most_viewed]).drop_duplicates()
    
    for _, row in labeled.iterrows():
        label = f"{row['artist'][:15]}\n{row['title'][:15]}"
        ax.annotate(
            label,
            xy=(row['rank'], row['youtube_views_millions']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=7,
            alpha=0.8
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_vs_views.png', dpi=150)
    plt.close()
    print(f"  Saved: rank_vs_views.png")
    
    # 1b. Scatter plot: Rank vs YouTube Views (LOG SCALE)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(
        data=df_with_views,
        x='rank',
        y='youtube_views_millions',
        alpha=0.6,
        s=100,
        ax=ax
    )
    ax.set_yscale('log')
    ax.set_xlabel('Rank in Radio 357 Top (Left is better)', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions) - Log Scale', fontsize=12)
    ax.set_title('Radio 357 Top Songs: Rank vs YouTube Views (Log Scale)', fontsize=14, fontweight='bold')
    
    for _, row in labeled.iterrows():
        label = f"{row['artist'][:15]}\n{row['title'][:15]}"
        ax.annotate(
            label,
            xy=(row['rank'], row['youtube_views_millions']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=7,
            alpha=0.8
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rank_vs_views_log.png', dpi=150)
    plt.close()
    print(f"  Saved: rank_vs_views_log.png")
    
    # 2. Bar chart: Top 20 most viewed songs from the list
    fig, ax = plt.subplots(figsize=(14, 10))
    top_viewed = df_with_views.nlargest(20, 'youtube_views').copy()
    top_viewed['label'] = top_viewed['artist'].str[:20] + ' - ' + top_viewed['title'].str[:25]
    
    colors = sns.color_palette('viridis', n_colors=20)
    bars = ax.barh(
        y=range(len(top_viewed)),
        width=top_viewed['youtube_views_millions'].values,
        color=colors
    )
    ax.set_yticks(range(len(top_viewed)))
    ax.set_yticklabels(top_viewed['label'].values)
    ax.invert_yaxis()
    ax.set_xlabel('YouTube Views (Millions)', fontsize=12)
    ax.set_ylabel('')
    ax.set_title('Top 20 Most Viewed Songs on YouTube\n(from Radio 357 Top List)', fontsize=14, fontweight='bold')
    
    for i, (bar, views) in enumerate(zip(bars, top_viewed['youtube_views_millions'].values)):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{views:.0f}M', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_viewed_songs.png', dpi=150)
    plt.close()
    print(f"  Saved: top_viewed_songs.png")
    
    # 3. Distribution of YouTube views
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=df_with_views,
        x='youtube_views_millions',
        bins=30,
        kde=True,
        ax=ax,
        color='steelblue'
    )
    ax.set_xlabel('YouTube Views (Millions)', fontsize=12)
    ax.set_ylabel('Number of Songs', fontsize=12)
    ax.set_title('Distribution of YouTube Views for Radio 357 Top Songs', fontsize=14, fontweight='bold')
    
    median_views = df_with_views['youtube_views_millions'].median()
    ax.axvline(median_views, color='red', linestyle='--', linewidth=2, label=f'Median: {median_views:.0f}M')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'views_distribution.png', dpi=150)
    plt.close()
    print(f"  Saved: views_distribution.png")
    
    # 4. Rank bins vs average views
    fig, ax = plt.subplots(figsize=(10, 6))
    rank_stats = df_with_views.groupby('rank_bin', observed=True)['youtube_views_millions'].agg(['mean', 'median']).reset_index()
    
    x = range(len(rank_stats))
    width = 0.35
    ax.bar([i - width/2 for i in x], rank_stats['mean'], width, label='Mean', color='steelblue')
    ax.bar([i + width/2 for i in x], rank_stats['median'], width, label='Median', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(rank_stats['rank_bin'], rotation=45, ha='right')
    ax.set_xlabel('Rank in Radio 357 Top (Left is better)', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions)', fontsize=12)
    ax.set_title('Average YouTube Views by Rank Group', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'views_by_rank_group.png', dpi=150)
    plt.close()
    print(f"  Saved: views_by_rank_group.png")
    
    # 5. Box plot by rank group
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=df_with_views,
        x='rank_bin',
        y='youtube_views_millions',
        palette='viridis',
        ax=ax
    )
    ax.set_xlabel('Rank in Radio 357 Top (Left is better)', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions)', fontsize=12)
    ax.set_title('Distribution of YouTube Views by Rank Group', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'views_boxplot_by_rank.png', dpi=150)
    plt.close()
    print(f"  Saved: views_boxplot_by_rank.png")
    
    # 6. Top artists by total views
    artist_views = df_with_views.groupby('artist')['youtube_views'].sum().nlargest(15) / 1_000_000
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette('viridis', n_colors=len(artist_views))
    bars = ax.barh(range(len(artist_views)), artist_views.values, color=colors)
    ax.set_yticks(range(len(artist_views)))
    ax.set_yticklabels(artist_views.index)
    ax.invert_yaxis()
    ax.set_xlabel('Total YouTube Views (Millions)', fontsize=12)
    ax.set_title('Top 15 Artists by Total YouTube Views\n(from Radio 357 Top List)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'top_artists_by_views.png', dpi=150)
    plt.close()
    print(f"  Saved: top_artists_by_views.png")
    
    print("\nBasic visualizations saved!")


def create_decade_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create visualizations analyzing songs by decade of release."""
    df_with_years = df[df['original_release_year'].notna() & df['youtube_views'].notna()].copy()
    
    if len(df_with_years) < 10:
        print("Not enough songs with release years for decade analysis.")
        return
    
    print(f"\nCreating decade analysis with {len(df_with_years)} songs...")
    
    # 1. Songs per decade
    fig, ax = plt.subplots(figsize=(12, 6))
    decade_counts = df_with_years.groupby('decade').size()
    
    colors = sns.color_palette('viridis', n_colors=len(decade_counts))
    ax.bar(decade_counts.index.astype(str), decade_counts.values, color=colors)
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Number of Songs', fontsize=12)
    ax.set_title('Radio 357 Top Songs by Decade of Release', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    for i, (decade, count) in enumerate(zip(decade_counts.index, decade_counts.values)):
        ax.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'songs_per_decade.png', dpi=150)
    plt.close()
    print(f"  Saved: songs_per_decade.png")
    
    # 2. Average views per decade
    fig, ax = plt.subplots(figsize=(12, 6))
    decade_views = df_with_years.groupby('decade')['youtube_views_millions'].agg(['mean', 'median']).reset_index()
    
    x = range(len(decade_views))
    width = 0.35
    ax.bar([i - width/2 for i in x], decade_views['mean'], width, label='Mean', color='steelblue')
    ax.bar([i + width/2 for i in x], decade_views['median'], width, label='Median', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels(decade_views['decade'].astype(str), rotation=45, ha='right')
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions)', fontsize=12)
    ax.set_title('Average YouTube Views by Release Decade', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'views_by_decade.png', dpi=150)
    plt.close()
    print(f"  Saved: views_by_decade.png")
    
    # 3. Scatter: Release Year vs Views
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(
        data=df_with_years,
        x='original_release_year',
        y='youtube_views_millions',
        alpha=0.6,
        s=80,
        ax=ax
    )
    ax.set_xlabel('Original Release Year', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions)', fontsize=12)
    ax.set_title('Song Age vs YouTube Popularity', fontsize=14, fontweight='bold')
    
    # Add trend line
    z = pd.DataFrame({'x': df_with_years['original_release_year'], 'y': df_with_years['youtube_views_millions']}).dropna()
    if len(z) > 2:
        import numpy as np
        coeffs = np.polyfit(z['x'], z['y'], 1)
        trend_line = np.poly1d(coeffs)
        x_range = range(int(z['x'].min()), int(z['x'].max()) + 1)
        ax.plot(x_range, trend_line(x_range), 'r--', alpha=0.7, label='Trend')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'release_year_vs_views.png', dpi=150)
    plt.close()
    print(f"  Saved: release_year_vs_views.png")
    
    # 4. Box plot by decade
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_with_years,
        x='decade',
        y='youtube_views_millions',
        palette='viridis',
        ax=ax
    )
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('YouTube Views (Millions)', fontsize=12)
    ax.set_title('Distribution of YouTube Views by Release Decade', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'views_boxplot_by_decade.png', dpi=150)
    plt.close()
    print(f"  Saved: views_boxplot_by_decade.png")
    
    print("Decade visualizations saved!")


def create_advanced_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create advanced visualizations: Taste Gap and Views Velocity."""
    df_clean = df[df['youtube_views'].notna() & df['rank'].notna()].copy()
    
    if len(df_clean) == 0:
        return
    
    print(f"\nCreating advanced visualizations...")

    # 1. Taste Gap: Radio Classics vs YouTube Hidden Gems
    # Top 10 Hidden Gems (Higher YT popularity than Radio rank would suggest)
    hidden_gems = df_clean.nlargest(10, 'taste_gap')
    # Top 10 Radio Classics (Higher Radio popularity than YT would suggest)
    radio_classics = df_clean.nsmallest(10, 'taste_gap')
    
    gap_df = pd.concat([hidden_gems, radio_classics])
    gap_df['type'] = ['Hidden Gem'] * 10 + ['Radio Classic'] * 10
    gap_df['label'] = gap_df['artist'].str[:20] + ' - ' + gap_df['title'].str[:25]

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = ['#2ecc71' if t == 'Hidden Gem' else '#e74c3c' for t in gap_df['type']]
    
    bars = ax.barh(range(len(gap_df)), gap_df['taste_gap'], color=colors)
    ax.set_yticks(range(len(gap_df)))
    ax.set_yticklabels(gap_df['label'])
    ax.invert_yaxis()
    ax.set_xlabel('Taste Gap Score (YouTube Popularity - Radio Ranking)', fontsize=12)
    ax.set_title('Radio Classics vs YouTube Hidden Gems\n(Based on Rank vs View Count)', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='#2ecc71', lw=4, label='YouTube Hidden Gems'),
                      Line2D([0], [0], color='#e74c3c', lw=4, label='Radio 357 Classics')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'taste_gap_analysis.png', dpi=150)
    plt.close()
    print(f"  Saved: taste_gap_analysis.png")

    # 2. Views Velocity (Top 20 trending songs)
    if 'views_per_day' in df_clean.columns:
        velocity_df = df_clean.dropna(subset=['views_per_day']).nlargest(20, 'views_per_day').copy()
        if not velocity_df.empty:
            # Include release year and upload date in the label
            def create_velocity_label(row):
                artist = str(row['artist'])[:20]
                title = str(row['title'])[:25]
                year = f"({int(row['original_release_year'])})" if pd.notna(row['original_release_year']) else ""
                upload = f"[YT: {row['youtube_upload_date']}]" if pd.notna(row['youtube_upload_date']) else ""
                return f"{artist} - {title} {year} {upload}".strip()
            
            velocity_df['label'] = velocity_df.apply(create_velocity_label, axis=1)
            
            fig, ax = plt.subplots(figsize=(14, 10))
            colors = sns.color_palette('rocket', n_colors=len(velocity_df))
            bars = ax.barh(range(len(velocity_df)), velocity_df['views_per_day'], color=colors)
            ax.set_yticks(range(len(velocity_df)))
            ax.set_yticklabels(velocity_df['label'])
            ax.invert_yaxis()
            ax.set_xlabel('Average Views Per Day since Upload', fontsize=12)
            ax.set_title('YouTube View Velocity: Top 20 Trending Classics\n(Average views per day since upload)', fontsize=14, fontweight='bold')
            
            for i, bar in enumerate(bars):
                val = velocity_df.iloc[i]['views_per_day']
                if pd.notna(val):
                    ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2, 
                            f'{int(val):,} vpd', va='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'views_velocity.png', dpi=150)
            plt.close()
            print(f"  Saved: views_velocity.png")

    # 3. Artist Bubble Chart (Artist dominance)
    artist_stats = df_clean.groupby('artist').agg({
        'rank': 'mean',
        'youtube_views_millions': 'sum',
        'title': 'count'
    }).rename(columns={'title': 'song_count'})
    
    # Filter for artists with at least 2 songs
    artist_stats = artist_stats[artist_stats['song_count'] >= 2]
    
    if len(artist_stats) > 5:
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(
            artist_stats['rank'],
            artist_stats['youtube_views_millions'],
            s=artist_stats['song_count'] * 200,
            alpha=0.5,
            edgecolors="w",
            linewidth=2,
            c=artist_stats['rank'],
            cmap='viridis_r'
        )
        
        # Add labels for top artists
        top_artists = pd.concat([
            artist_stats.nsmallest(5, 'rank'),
            artist_stats.nlargest(5, 'youtube_views_millions')
        ]).drop_duplicates()
        
        for idx, row in top_artists.iterrows():
            ax.annotate(idx, (row['rank'], row['youtube_views_millions']), 
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Average Rank in Radio 357 Top (Left is better)', fontsize=12)
        ax.set_ylabel('Total YouTube Views (Millions)', fontsize=12)
        ax.set_title('Artist Dominance Bubble Chart\n(Size = Number of songs in Top, Color = Avg Rank)', fontsize=14, fontweight='bold')
        # Rank 1 is on the left by default, which matches "Left is better"
        
        plt.colorbar(scatter, label='Average Rank')
        plt.tight_layout()
        plt.savefig(output_dir / 'artist_bubble_chart.png', dpi=150)
        plt.close()
        print(f"  Saved: artist_bubble_chart.png")


def print_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    df_with_views = df[df['youtube_views'].notna()]
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total songs: {len(df)}")
    print(f"Songs with view data: {len(df_with_views)}")
    print(f"Songs with release year: {df['original_release_year'].notna().sum()}")
    
    if len(df_with_views) > 0:
        print(f"\nYouTube Views Statistics:")
        print(f"  Mean: {df_with_views['youtube_views'].mean():,.0f}")
        print(f"  Median: {df_with_views['youtube_views'].median():,.0f}")
        print(f"  Max: {df_with_views['youtube_views'].max():,.0f}")
        print(f"  Min: {df_with_views['youtube_views'].min():,.0f}")
        print(f"  Total: {df_with_views['youtube_views'].sum():,.0f}")
        
        # Correlation
        corr = df_with_views['rank'].corr(df_with_views['youtube_views'])
        print(f"\nCorrelation (Rank vs Views): {corr:.3f}")
        if corr < 0:
            print("  -> Negative correlation: songs closer to Rank 1 tend to have more views")
        else:
            print("  -> Positive correlation: songs with worse rank numbers tend to have more views")
        
        # Most viewed songs
        print("\n" + "="*60)
        print("TOP 10 MOST VIEWED SONGS")
        print("="*60)
        for _, row in df_with_views.nlargest(10, 'youtube_views').iterrows():
            year_str = f" ({int(row['original_release_year'])})" if pd.notna(row.get('original_release_year')) else ""
            print(f"#{row['rank']:3d} | {row['youtube_views']:>12,} views | {row['artist']} - {row['title']}{year_str}")
        
        # Taste Gap highlights
        if 'taste_gap' in df.columns:
            print("\n" + "="*60)
            print("TASTE GAP ANALYSIS")
            print("="*60)
            print("YouTube HIDDEN GEMS (Higher popularity on YT than Radio rank suggests):")
            for _, row in df.nlargest(5, 'taste_gap').iterrows():
                print(f"  - {row['artist']} - {row['title']} (Rank #{row['rank']}, Views: {row['youtube_views_millions']:.0f}M)")
            
            print("\nRadio 357 CLASSICS (Higher popularity on Radio than YT views suggest):")
            for _, row in df.nsmallest(5, 'taste_gap').iterrows():
                print(f"  - {row['artist']} - {row['title']} (Rank #{row['rank']}, Views: {row['youtube_views_millions']:.0f}M)")

        # Decade analysis if available
        if 'original_release_year' in df.columns and df['original_release_year'].notna().sum() > 10:
            df_with_years = df_with_views[df_with_views['original_release_year'].notna()]
            print("\n" + "="*60)
            print("SONGS BY DECADE")
            print("="*60)
            decade_counts = df_with_years.groupby('decade').size().sort_index()
            for decade, count in decade_counts.items():
                print(f"  {int(decade)}s: {count} songs")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Analyze Radio 357 song data')
    parser.add_argument('--data', type=str, help='Path to data file (JSON or CSV)')
    parser.add_argument('--run', type=str, help='Run folder name (e.g., run_2026-01-02_10-34-18)')
    parser.add_argument('--output', type=str, help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Determine data source
    data_path = None
    
    if args.data:
        data_path = Path(args.data)
    elif args.run:
        run_dir = BASE_DIR / args.run
        if not run_dir.exists():
            print(f"ERROR: Run folder not found: {run_dir}")
            return
        data_path = run_dir / "songs_with_views.json"
    else:
        # Find latest run
        latest_run = find_latest_run()
        if latest_run:
            data_path = latest_run / "songs_with_views.json"
            print(f"Using latest run: {latest_run.name}")
        else:
            print("ERROR: No run folders found. Run radio357_youtube_analysis.py first.")
            return
    
    if not data_path or not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Radio 357 Songs - Data Analysis")
    print(f"Data source: {data_path}")
    print(f"Output folder: {output_dir}")
    print("="*60)
    
    # Load and prepare data
    df = load_data(data_path)
    df = prepare_dataframe(df)
    
    print(f"\nLoaded {len(df)} songs.")
    
    # Create visualizations
    create_basic_visualizations(df, output_dir)
    create_decade_visualizations(df, output_dir)
    create_advanced_visualizations(df, output_dir)
    
    # Print statistics
    print_statistics(df)
    
    print(f"\n[OK] All output saved to: {output_dir}")


if __name__ == "__main__":
    main()
