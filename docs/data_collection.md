# Data Collection Documentation

## Overview
This document describes the data collection process for the Movie Recommendation System project. The system collects movie data from multiple sources to build a comprehensive dataset for training and evaluation.

## Data Sources

### 1. TMDB API
- **Source**: The Movie Database (TMDB) API
- **URL**: https://www.themoviedb.org/documentation/api
- **Data Collected**:
  - Movie metadata (title, release date, runtime, etc.)
  - Cast and crew information
  - User reviews and ratings
  - Movie posters and images
  - Genres and keywords

### 2. Existing Dataset
- **Source**: TMDB Dataset
- **File**: main_data.csv
- **Data Fields**:
  - director_name: Tên đạo diễn
  - actor_1_name, actor_2_name, actor_3_name: Tên diễn viên chính
  - genres: Thể loại phim
  - movie_title: Tên phim
  - comb: Thông tin kết hợp cho recommendation

## Data Collection Process

### 1. Initial Data Collection
1. Download base dataset from TMDB
2. Process and clean the data
3. Save to main_data.csv

### 2. Ongoing Data Collection
1. Use TMDB API to fetch new movie data
2. Update existing dataset with new information
3. Process and integrate new data
4. Save updated dataset

### 3. Data Processing Steps
1. Clean and normalize movie titles
2. Extract and standardize genres
3. Process cast and crew information
4. Generate combined features for recommendation
5. Update similarity matrices

## Data Update Schedule
- Daily: Update movie ratings and reviews
- Weekly: Add new movies and update metadata
- Monthly: Full dataset refresh

## Data Quality Checks
1. Verify all required fields are present
2. Check for duplicate entries
3. Validate data formats
4. Ensure proper encoding of special characters
5. Verify API response integrity

## Usage
To collect new data:
```bash
python data_collection.py
```

## Dependencies
- requests
- pandas
- numpy
- beautifulsoup4
- logging
- datetime

## Future Improvements
1. Add more data sources
2. Implement data validation
3. Add data cleaning steps
4. Create data versioning system
5. Implement automated quality checks 