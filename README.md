# recommender-learn

The know-how on how recommendation system works, and how to apply different algorithms for your projects.


## Types of recommendation system

- Collaborative Learning
- Content Based (item based or user based)
- Hybrid recommender

Input
- data in the form of rating, score or events, sentiment

Output
- ranking of items to recommend
- pool of items
- score for the items

## Similarity Algorithms

What are the different type of similarity function and what usecases are they suitable for?

- https://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
- https://pdfs.semanticscholar.org/d490/d24e64022735116b45dd18307d6dba97170c.pdf

## Collaborative Filtering

Using One Slope algorithm for binary ranking (1, 0)
https://en.wikipedia.org/wiki/Slope_One

## Recommendation algorithm
- how to write one from scratch
- how to scale one
- using existing libraries
- deploying it to production
- retraining the recommendation algorithm

https://medium.com/arc-software/recommender-systems-behind-the-scenes-a39c831a0ae2
https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
https://en.m.wikipedia.org/wiki/Recommender_system#Content-based_filtering
https://en.m.wikipedia.org/wiki/Tf–idf
https://en.m.wikipedia.org/wiki/SMART_Information_Retrieval_System
https://en.m.wikipedia.org/wiki/Latent_semantic_analysis
https://en.m.wikipedia.org/wiki/Latent_Dirichlet_allocation
https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial
https://www.kaggle.com/pierremegret/gensim-word2vec-tutorial

## Viral factor calculator

Viral factor calculator - calculates the net worth of a social media celebrity. Also includes recency (newest comments/feeds will have higher score) and the success rate of the merchants they promote.

Recommender system algorithm + hot rank algorithm. Incorporate recency into the recommender engine so that newer items will be ranked higher than older ones. Why is this important? Some products (such as events/promotion etc) has expiry time and is valid only for a certain period. Others, such as movies/songs are seasonal - they follow a certain trend and usually according to the time. So it might be more reasonable to recommend songs now than songs in the 50s even if the similiarity score is higher.

## ElasticSearch/Lucene

## Page Rank

## Data Gathering

## Github Recommender

displays a list of top github users with recommendation. Skills proven: data recommendation for users, data scraping and data visualization.
learn to

## Events

| Events |Event	Meaning	| Event Name|
| - | - | - |
| Scrolling a themed row |	User is interested in the theme, e.g. Dramas	|genreView|
|Placing the mouse over a film to request an overview of content	|User is interested in the movie (a drama) thereby showing interest in this category	|details|
|Clicking the film to request the details of the content	|User is more interested in the movie|	moreDetails|
|Adding the film to my list	|User intends to watch the movie later	|addToList|
|Starting to watch the movie	|User purchases the movie	|playStart|

|Logging|userID	|contentID	|event	date|
|-|-|-|-|
|1234|	1|	addToList|	2017-01-01|
|1234|	2|	genreView|	2017-01-01|


## Client side code to capture events
```js
function onScroll(evt) {
	// Get the genre
	if ($el === '.carousel' && $el.dataset[‘genre’])  logGenre()
}
```
```js
function onClick() {
	logEvent(‘details’, userId)
}
```

## Indicator of Interest
clicks 
- links: user wants to get more information
- see more (expansion clicks): user is interested to know more
- item: user selects the items of preference
- filter: user has a specific search criteria (which can be stored for future use)
share: user shows interest in an item
save for later/bookmark: user shows interest in an item
subscribe/unsubscribe: user wants to be notified on new content
scroll
- user wants to see more item
rate: user wants to rate an item. ratings can be explicit or implicit
	- explicit: added manually by the users
	- implicit: calculated from the evidence you collected
vote: user wants to vote/unvote item he/she likes/agree/disagree/has strong opinion <insert context>. Sites that use voting are called reputation systems.
like: user shows like/dislike on an item
comment: user wants to leave comment (positive/negative/neutral)
review: user wants to leave review on a product (positive/negative/neutral)
search
- keywords: user is looking for a particular keyword
- filters: user has a criteria of search (which may change over time, so it’s better to set an “expiry” time on the preference, or store a history of it)
page_duration: For page duration, it depends on the application too. A video streaming site might have users staying longer (if the user stayed for the whole period of the movie, it means they love it!)
while a ecommerce site might expect users to be there just for a short duration
activity: let’s take a music playing app for example
- start playing: the user is interested, that is already a positive
- stop playing: maybe the user think it’s bad, but this is relative to where the user stop. Stopping at the start could mean that it is bad, stopping at the end could mean something else
- resume playing: forget all the negative implicit ratings that the system has registered, the user is probably distracted by something else (someone calling them etc). If playing is resumed in 5 minutes, the user probably has something else at hand. If the playing is resumed 24 hours later, it could mean something else)
- speeding: If the user skips something in the middle, it’s probably not a good sign. Unless the user is resuming
- playing it to the end - we have a winner. This is a good sign, unless the user is idle and left the playlist playing
- replaying - this could be a positive indicator. But replaying a music and an educational video might have different context - the latter might suggest that the content is too difficult to follow.

Creating profile for a single vs average group
- when building a profile, we always attempt to target a group of users and averaging their behaviours.
- this might or might not work, as different users really have different habits
- but building a targetted personalisation would consume too much resources


| Duration on Page | What it means |
| - | - |
|less than 5 seconds | No interest|
|more than 5 seconds | interested|
|more than 1 minute | very interested|
|more than 5 minutes | probably went to get coffee|
|more than 10 mins | interrupted or went away from the page without following the link|


## Cold Start

Dealing with cold start
- show default data by most popular/latest
- we can also segment them by demographic/age etc
- gray sheeps are users that have individual taste that does not resemble other users, and hence it is hard to recommend products to them.

## Building a recommendation engine

https://www.codeproject.com/Articles/1232150/Building-a-Recommendation-Engine-in-Csharp
https://stackoverflow.com/questions/2323768/how-does-the-amazon-recommendation-feature-work
https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-2-d9b96aa399f5

Case based reasoning CBR: https://en.m.wikipedia.org/wiki/Case-based_reasoning
https://www.quora.com/What-are-examples-of-rule-engines-combined-with-machine-learning


## Matrix Factorization

Link to another repo [here](https://github.com/alextanhongpin/matrix-factorization).

- https://en.m.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
- http://rstudio-pubs-static.s3.amazonaws.com/335300_11d40bf12d8940f78d9661b3c63150dc.html
- https://www.freecodecamp.org/news/singular-value-decomposition-vs-matrix-factoring-in-recommender-systems-b1e99bc73599/


## Hacker news algorithm

Pseudo code:
```
score = (P - 1) / (T + 2)^ G
where,
P = points of an item (-1 is to negate submitters vote)
T = time since submission (in hours)
G = gravity, defaults to 1.8
```

Effects of gravity (G) and time (T)
- the score decreases as T increases, meaning older items will get lower and lower score
- the score decreases much faster for older items if gravity is increased

Python implementation:
```python
def calculate_score(votes, item_hour_age, gravity=1.8):
	return (votes - 1) / pow((item_hour_age + 2), gravity)
```

References:
- https://medium.com/hacking-and-gonzo/how-hacker-news-ranking-algorithm-works-1d9b0cf2c08d



## Recommendation

- https://medium.com/recombee-blog/machine-learning-for-recommender-systems-part-1-algorithms-evaluation-and-cold-start-6f696683d0ed
- https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/
- https://en.wikipedia.org/wiki/Recommender_system
- https://towardsdatascience.com/learning-to-make-recommendations-745d13883951

# Learning to Rank (LtR)

## Ranking and Learning to Rank

- types of Learning to Rank (LTR) algorithms, such as pointwise, pairwise and listwise comparison of ranks
- bayesian personalized ranking (BPR) 

## The three types of LTR algorithm

### pointwise

- produce a score for each item and them ranks them accordingly
- the difference between rating prediction and ranking is that with ranking, you don’t care of an item has a utility score of a million or within a rating scale, as long as the score symbolises a position in the rank

### pairwise

- a type of binary classifier
- a function that takes two items an returns an ordering of the two
- in pairwise ranking, output is optimised to minimise the number of inversions of items compared to the optimal rank of the items
- an inversion is when two items changes place
- e.g. Bayesian Personalized Ranking (BPR)

### listwise

- the king of all LTR subgroups
- looks at the whole ranked list and optimises that
- advantage of listwise ranking is that it intuits that ordering is more important at the top of a ranked list than at the bottom
- pointwise and pairwise algorithms don’t distinguish where on the ranked list you are
- e.g. CoFiRank (collaborative filtering for ranking)


## Implicit ratings
```
Implicit Rating, IR = #event_1 * w_1 + #event_2 * w_2 + #event_3 * w_3…#event_n + w_n

IR: the implicit rating
#event: the number of times the specific event occured
w: The weights that you set based on previous analysis (and probably tweak again when you start creating recommendations)
````


## Recommender algorithms
- memory-based: recommender accesses the logs data in real-time
- model-based: signifies that the algorithm aggregates the data beforehand to make it more responsive


## Similarity and Distance

The similarity of item 1 and item 2 can be represented as the following:

- `sim(i_1, i_2), 0=not similar, 1=exact`
- similarity is the inverse of distance
- as the distance increase, the similarity goes towards zero
- as the distance decrease, the similarity goes towards one

| Data Type | Similarity |
| - | - | 
| unary/binary data | jaccard similarity |
| quantitative data | pearson/cosine similarity |



## Types of similarity algorithm
- jaccard
- l1 norm - manhattan distance
- l2 norm - euclidean distance
- pearson 
- cosine

## Jaccard Index
- Jaccard similarity index/coefficient compares members for two sets to see which members are shared and which are distinct.
- It is a measure of similarity for the two sets of data, with a range from 0 to 1.
- The higher the value, the more similar the two populations
```
J(X, Y) = X intersect Y / X union Y
```
In steps, that is:
- count the number of similar members between both sets (intersection)
- count the total number of unique members in both sets (union)
- divide them to get the jaccard similarity 

Observation:
- the value is 1 when both sets have the same members. (a non-empty set that is computed against itself will return 1 too)
- the value is 0 if both sets have no similar members
- the value is 0.5 if both sets share half the members
- the value is 0 if the set is empty

Jaccard Distance
- as opposed to the jaccard similarity, the jaccard distance measures how dissimilar two sets are. 
- in set notation, subtract 1 from the jaccard distance
```
D(X, Y) = 1 - J(X, Y)
```

## L1 Norm
- aka manhattan distance or taxicab norm
- the idea is that if you want to measure the distance between two street corners in Manhattan, you drive a grid, rather than measure as the crow flies
- the sum of magnitudes of the vectors in space
```
Given X = [3,4] // x and y coordinate
||X||1 = |3| + |4| = 7
```
The similarity is just the inverse of the distance. We add one to the denominator to avoid zero-division.
```
S = 1 / (||X||1 + 1)
```
## L2 Norm
- aka euclidean norm
- the distance between two points not travelled by a taxi in Manhattan, but by the crow, going directly from one point to another.
- pythagorean theorem, `a^2 + b^2  = c^2`
```
Given X = [3,4] // x and y coordinate
||X||2 = sqrt(|3^2| + |4^2|) = 5
```

## Cosine similarity
- the angle between two points in a vector space
```
sim(i, j) = item1 * item2 / sqrt(square(item1) * square(item2))
```


## References

- https://github.com/practical-recommender-systems/moviegeek/blob/master/recs/bpr_recommender.py
- http://www.gabormelli.com/RKB/Bayesian_Personalized_Ranking_(BPR)_Algorithm
- https://towardsdatascience.com/recommender-system-using-bayesian-personalized-ranking-d30e98bba0b9
- https://medium.com/radon-dev/implicit-bayesian-personalized-ranking-in-tensorflow-b4dfa733c478
- https://github.com/benfred/implicit/blob/master/implicit/bpr.pyx
- https://github.com/alfredolainez/bpr-spark/blob/master/bpr.py


- https://medium.com/seek-blog/learning-to-rank-dogs-87a6c68dda43
- http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/
- http://www.alfredo.motta.name/learning-to-rank-with-python-scikit-learn/
- https://thenewstack.io/letor-machine-learning-web-search-technique-thats-turned-key-information-retrieval-tool/
- https://www.infoworld.com/article/3259845/introduction-to-learning-to-rank-ltr-search-analysis.html

