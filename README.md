# DS 4320 Project 2: Movie Recommendation System

## Executive Summary

This repository contains a complete data science project focused on building
a movie recommendation system using the MovieLens dataset. The project
demonstrates the full pipeline from raw data acquisition through MongoDB
document storage, exploratory analysis, collaborative filtering modeling,
and visualization. The data consists of 100,836 user ratings across 9,742
movies stored in MongoDB Atlas using the document model. The repository
includes a data loading script, an analysis and visualization pipeline,
a press release, and full metadata documentation.

**Alka Link - eju2pk**

| Links |
|-------|
| [![DOI](https://zenodo.org/badge/1217142206.svg)](https://doi.org/10.5281/zenodo.19684378) |
| [Press Release](https://github.com/alkalink1/ds4320-project2-movielens/blob/main/press%20release/press-release.md) |
| [Movie Lens Dataset](https://github.com/alkalink1/ds4320-project2-movielens/tree/main/data) |
| [Pipeline](https://github.com/alkalink1/ds4320-project2-movielens/blob/main/pipeline/pipeline.ipynb) |
| [MIT License](https://github.com/alkalink1/ds4320-project2-movielens/blob/main/LICENSE) |

---

## Problem Definition

### Initial and Refined Problem Statement

**Initial general problem:**
Recommending content to users based on their preferences.

**Refined specific problem statement:**
Build a movie recommendation system that predicts which movies a user is
likely to enjoy based on historical rating data from the MovieLens dataset,
using user-movie interactions stored in a document database.

### Motivation

With the rapid growth of digital content, users are often overwhelmed by
the number of available choices on platforms like Netflix and YouTube.
Recommendation systems are essential for helping users discover relevant
content efficiently and improving user engagement. This project is motivated
by the need to understand how user behavior data, and specifically how movie ratings
can be used to generate personalized recommendations that reflect individual
preferences.

### Rationale for Refinement

The general problem of recommending content is very broad, so the focus was
narrowed to movie recommendations using structured rating data from the
MovieLens dataset. This refinement makes the problem more tractable because
user preferences can be directly quantified through numerical ratings.
Additionally, the dataset provides consistent identifiers (userId, movieId)
that allow for clear relationships between users and movies, enabling the
use of collaborative filtering techniques to generate recommendations.

### Press Release Headline

**Highly Rated Movies Are Not Just Popular — They Are Consistently Loved**
[Jump to Press Release](https://github.com/alkalink1/ds4320-project2-movielens/blob/main/press%20release/press-release.md)

---

## Domain Exposition

### Terminology

| Term | Definition |
|------|------------|
| Recommendation System | Algorithm that suggests content to users |
| Collaborative Filtering | Method that recommends items based on similar users |
| User-Item Matrix | Table of users and their ratings of movies |
| Rating | Numerical score (0.5–5.0) given by a user to a movie |
| Similarity | Measure of how alike two users are based on rating history |
| Prediction | Estimated rating a user would give to an unseen movie |
| Personalization | Tailoring recommendations to individual users |
| Genre | Category of a movie (e.g. Drama, Comedy, Action) |
| Tag | User-generated label describing a movie |
| Cold Start | Problem of recommending to new users with no rating history |

### Domain

This project operates in the domain of recommendation systems, which are
widely used to personalize content on digital platforms. These systems rely
on user interaction data, such as ratings, to identify patterns in preferences
and predict future behavior. In this case, the dataset consists of user
ratings for movies, allowing preferences to be modeled quantitatively.
Recommendation systems use these patterns to suggest content that users are
more likely to enjoy, improving satisfaction and engagement.

### Background Readings

| Title | Brief Description | Link |
|------|------------|------|
| The MovieLens Datasets: History and Context | Official paper describing the MovieLens dataset and its structure. | [movie_lens.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQCGsCh-rrXKRbCQj6JMeynQAZXeEH2nDvdDi_Ko7C0OHn8?e=BfEUyG) |
| Introduction to Recommender Systems | Foundational overview of recommendation algorithms and personalization. | [intro_recommenders.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQBy7_SdOSzLSIVGtNZ6o66LAe3i3Hnt1mXrqUs8Fq0ufC8?e=lYpgBg) |
| Item-Based Collaborative Filtering Recommendation Algorithms | Classic paper explaining collaborative filtering methods. | [collaborative_filtering.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQBlYip4QgmYRrjRezgjFUTkAZDS4fqPMO3V2b1YdeTv8c0?e=vQGmCN) |
| A Survey on Bias and Fairness in Recommender Systems | Discusses fairness, bias, and ethical concerns in recommendation systems. | [fairness_recommenders.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQB4HE-W7dU5SIKrySaPCGOHAc2UGtolo58AE5j-ZGWvfLo?e=5m7K1E) |
| Netflix Recommendations: Beyond the 5 Stars | Industry example of recommendation systems in practice. | [netflix_recommendations.pdf](https://myuva-my.sharepoint.com/:b:/g/personal/eju2pk_virginia_edu/IQDs3ISMOX3JQIIGEpwbOqo6AXfio190ty-OnZ1ZH3YRLPo?e=Ue5gQn) |

## Data Creation

### Provenance

The data for this project comes from the MovieLens dataset, maintained by
GroupLens Research at the University of Minnesota. It was downloaded directly
from grouplens.org/datasets/movielens as a zip archive containing four CSV
files. The version used is ml-latest-small, which contains 100,836 ratings
and 3,683 tag applications across 9,742 movies from 610 users, collected
between March 29, 1996 and September 24, 2018. Users were selected at random
and had each rated at least 20 movies. No demographic information is included.
The data required no preprocessing to load and is well-documented with a
README describing every file and field.

### Code Table

| File | Description | Link |
|------|-------------|------|
| load_movielens.ipynb | Reads ratings.csv and movies.csv, converts genres to arrays, and inserts all documents into MongoDB Atlas | [load_movielens.ipynb](./pipeline/load_movielens.ipynb) |
| pipeline.ipynb | Queries MongoDB to build a dataframe, filters by minimum rating count, runs collaborative filtering, and produces final visualization | [pipeline.ipynb](./pipeline/pipeline.ipynb) |

### Rationale for Critical Decisions

The ml-latest-small version was chosen over the larger 25M rating dataset
to keep computation manageable in a Colab environment while still providing
enough data to produce meaningful recommendations. The 50-rating minimum
threshold for a movie to be included in recommendations was a deliberate
tradeoff: lowering it would include more movies but introduce noisier
recommendations, while raising it would improve reliability but shrink the
pool of recommendable titles. Movie titles were kept as-is from the dataset
including the year suffix in parentheses, since stripping it would require
additional cleaning with no analytical benefit. Genres were converted from
pipe-separated strings to arrays in MongoDB to better align with the
document model and enable cleaner querying.

### Bias Identification

The MovieLens dataset reflects the preferences of users who voluntarily
chose to use the MovieLens platform, introducing self-selection bias. These
users tend to be more engaged with film than the average person. Additionally,
popular and well-known movies accumulate far more ratings than obscure ones,
creating a popularity bias where mainstream films are overrepresented.

### Bias Mitigation

Popularity bias is partially addressed by requiring a minimum of 50 ratings
before a movie is included in recommendations, ensuring recommendations
reflect consistent patterns rather than a few outlier ratings. Self-selection
bias is acknowledged as a limitation since no demographic information was
collected, making it impossible to compare the rater population against a
general audience baseline.

---

## Metadata

### Implicit Schema Guidelines

Because this project uses MongoDB, there is no enforced schema. The following
guidelines define the expected document structure for each collection.

**ratings collection** — one document per rating:
```json
{
  "userId": integer,
  "movieId": integer,
  "rating": float,
  "timestamp": integer
}
```

**movies collection** — one document per movie:
```json
{
  "movieId": integer,
  "title": string,
  "genres": array of strings
}
```

### Data Summary

| Collection | Documents | Description |
|------------|-----------|-------------|
| ratings | 100,836 | One document per user-movie rating containing userId, movieId, rating, and timestamp |
| movies | 9,742 | One document per movie containing movieId, title, and genres as an array |

The relationship between the ratings and movies collections is established
through the shared movieId field. Ratings documents reference movies by
movieId, which corresponds directly to the movieId field in the movies
collection. This link is implicit, MongoDB does not enforce it, but all
pipeline code relies on it to join the two collections in pandas.

### Data Dictionary

| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| userId | integer | Unique anonymized identifier for each user | 1 |
| movieId | integer | Unique identifier for each movie, consistent across all files | 318 |
| rating | float | User rating on a 0.5 to 5.0 scale in half-star increments | 4.5 |
| timestamp | integer | Seconds since Jan 1 1970 UTC when the rating was submitted | 964982703 |
| title | string | Movie title with release year in parentheses | Shawshank Redemption, The (1994) |
| genres | array | List of genre strings from a fixed set of 19 categories | ["Drama", "Crime"] |

### Data Dictionary: Quantification of Uncertainty

| Feature | Min | Max | Mean | Std Dev | Notes |
|---------|-----|-----|------|---------|-------|
| rating | 0.5 | 5.0 | 3.50 | 1.04 | Half-star increments only |
| timestamp | 828,124,615 | 1,537,799,250 | 1,205,946,000 | 216,261,000 | Covers March 1996 to September 2018 |
| userId | 1 | 610 | n/a | n/a | Arbitrarily assigned, no ordinal meaning |
| movieId | 1 | 193,609 | n/a | n/a | Non-sequential, gaps exist |
