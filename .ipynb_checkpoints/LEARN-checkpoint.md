# Recommendation in E-commerce

- [Personalized recommendation algorithm](https://mdpi-res.com/d_attachment/sustainability/sustainability-13-10786/article_deploy/sustainability-13-10786-v2.pdf)


## TODO
- [Movie recommender system](https://www.kaggle.com/rounakbanik/movie-recommender-systems)
- [Recommender System in Python 101](https://www.kaggle.com/gspmoreira/recommender-systems-in-python-101)
- [Recommender System using Amazon Reviews](https://www.kaggle.com/saurav9786/recommender-system-using-amazon-reviews)
- [Matrix factorization using the surprise library](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)
- [Links to real-time training + prediction models](https://github.com/benfred/implicit/issues/491)

## Datasets
- [Amazon Reviews](https://snap.stanford.edu/data/web-Amazon.html)

## TFIDF


[1] Note that the tf-idf functionality in sklearn.feature_extraction.text can produce normalized vectors, in which case cosine_similarity is equivalent to linear_kernel, only slower.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(X)

cosine_similarities = linear_kernel(tfidf_matrix[target_idx], tfidf_matrix).flatten()
related_docs_indices = cosine_similarities.argsort()[:-5:-1] # Meaning reverse the list, but take the last 5 only.
cosine_similarity[related_docs_indices]
```


- [1](https://scikit-learn.org/stable/modules/metrics.html)
- https://datascience.stackexchange.com/questions/62143/ts-ss-and-cosine-similarity-among-text-documents-using-tf-idf-in-python
- https://stackoverflow.com/questions/12118720/python-tf-idf-cosine-to-find-document-similarity/12128777#12128777