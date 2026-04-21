# Highly Rated Movies Are Not Just Popular — They Are Consistently Loved

## Hook

What if the best movie recommendations weren't just the highest-rated films,
but the ones that many people with similar tastes to yours agree are great?
Most streaming platforms show you what is popular — but popular is not the
same as right for you. This project shows there is a better way.

## Problem Statement

With thousands of movies available on platforms like Netflix, Hulu, and
Amazon Prime, users face a paradox of choice — too many options and not
enough signal to filter them. The most common approach, showing users what
is globally popular or highest-rated, fails because it ignores individual
taste. A movie rated 4.8 stars by 12 people tells you very little. A movie
rated 4.3 stars by 300 people who share your viewing history tells you
a lot. The challenge is building a system that identifies movies that are
both reliably well-rated across many users and specifically predicted to
appeal to an individual based on their own history.

## Solution Description

This project uses the MovieLens dataset — 100,836 ratings from 610 real
users across 9,742 movies — stored in MongoDB Atlas and analyzed using a
collaborative filtering model built on Singular Value Decomposition (SVD).
SVD works by finding hidden patterns in how groups of users rate movies
together, then using those patterns to predict how a specific user would
rate films they have never seen. The system filters out movies with fewer
than 50 ratings to ensure recommendations are based on reliable data rather
than noise, and generates a personalized top-10 list for any user in the
dataset. The model achieves an RMSE of 0.62, meaning its predictions are
within less than two-thirds of a star of the actual rating on average —
better than published benchmarks on this dataset.

## Chart

The chart below shows the top 10 movie recommendations generated for a
sample user. Each bar represents the model's predicted rating for that user
specifically. The color encodes the movie's global average rating across all
users — green means it is highly rated overall, red means it is lower rated
overall. The fact that Ace Ventura: Pet Detective appears near the top in red
is the system working correctly: the model predicts this user will love it
based on their rating history, even though it is not globally beloved. This
is the difference between personalized recommendation and simple popularity
ranking.

![Top 10 Recommendations for User 1](./recommendations.png)

The rating distribution below shows that most users rate movies between 3.0
and 4.0 stars, with 4.0 being the single most common rating. This positive
skew confirms that users tend to rate movies they chose to watch favorably,
which is an important characteristic of the dataset the model accounts for
by mean-centering ratings before applying SVD.

![Rating Distribution and Most-Rated Movies](./rating_distribution.png)