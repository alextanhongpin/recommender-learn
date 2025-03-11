
``` python
import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from implicit.nearest_neighbours import BM25Recommender, TFIDFRecommender
from implicit.cpu.bpr import BayesianPersonalizedRanking
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
```

``` python
corpus = [
    "I have a cat",
    "This cat of mine is cute",
    "The weather is amazing",
    "Playing sports on the weekend...",
    "An Item-Item Recommender on TF-IDF distances between items",
    "An Item-Item Recommender on BM25 distance between items",
]
```

``` python
data = CountVectorizer().fit_transform(corpus).T.tocsr()
data
```

    <Compressed Sparse Row sparse matrix of dtype 'int64'
        with 34 stored elements and shape (24, 6)>

``` python
model = TFIDFRecommender()
model.fit(data)
```

    /Users/alextanhongpin/Documents/go/recommender-learn/.venv/lib/python3.12/site-packages/implicit/utils.py:164: ParameterWarning: Method expects CSR input, and was passed coo_matrix instead. Converting to CSR took 9.202957153320312e-05 seconds
      warnings.warn(

      0%|          | 0/6 [00:00<?, ?it/s]

``` python
ids, scores = model.similar_items(0)

np.array(corpus)[ids], scores
```

    (array(['I have a cat', 'This cat of mine is cute'], dtype='<U58'),
     array([1.        , 0.15372732]))

``` python
ids, scores = model.similar_items(4)

np.array(corpus)[ids], scores
```

    (array(['An Item-Item Recommender on TF-IDF distances between items',
            'An Item-Item Recommender on BM25 distance between items',
            'Playing sports on the weekend...'], dtype='<U58'),
     array([1.        , 0.50495891, 0.03082585]))

``` python
model = BM25Recommender()
model.fit(data)
```

    /Users/alextanhongpin/Documents/go/recommender-learn/.venv/lib/python3.12/site-packages/implicit/utils.py:164: ParameterWarning: Method expects CSR input, and was passed coo_matrix instead. Converting to CSR took 0.00010204315185546875 seconds
      warnings.warn(

      0%|          | 0/6 [00:00<?, ?it/s]

``` python
ids, scores = model.similar_items(0)

np.array(corpus)[ids], scores
```

    (array(['I have a cat', 'This cat of mine is cute'], dtype='<U58'),
     array([3.19024436, 0.66062289]))

``` python
ids, scores = model.similar_items(4)

np.array(corpus)[ids], scores
```

    (array(['An Item-Item Recommender on TF-IDF distances between items',
            'An Item-Item Recommender on BM25 distance between items',
            'Playing sports on the weekend...'], dtype='<U58'),
     array([4.16738964, 2.03144396, 0.13862464]))

``` python
model = AlternatingLeastSquares()
model.fit(data)
```

      0%|          | 0/15 [00:00<?, ?it/s]

``` python
ids, scores = model.similar_items(0, N=len(corpus))

np.array(corpus)[ids], scores
```

    (array(['I have a cat', 'This cat of mine is cute',
            'An Item-Item Recommender on TF-IDF distances between items',
            'An Item-Item Recommender on BM25 distance between items',
            'Playing sports on the weekend...', 'The weather is amazing'],
           dtype='<U58'),
     array([ 1.        ,  0.0580235 , -0.01019325, -0.018768  , -0.0355837 ,
            -0.06902261], dtype=float32))

``` python
ids, scores = model.similar_items(4, N=len(corpus))

np.array(corpus)[ids], scores
```

    (array(['An Item-Item Recommender on TF-IDF distances between items',
            'I have a cat', 'Playing sports on the weekend...',
            'This cat of mine is cute', 'The weather is amazing',
            'An Item-Item Recommender on BM25 distance between items'],
           dtype='<U58'),
     array([ 0.9999999 , -0.01019325, -0.0163378 , -0.03531621, -0.04957363,
            -0.11652993], dtype=float32))

``` python
model = BayesianPersonalizedRanking()
model.fit(data)
```

      0%|          | 0/100 [00:00<?, ?it/s]

``` python
ids, scores = model.similar_items(0, N=len(corpus))

np.array(corpus)[ids], scores
```

    (array(['I have a cat',
            'An Item-Item Recommender on BM25 distance between items',
            'The weather is amazing', 'Playing sports on the weekend...',
            'This cat of mine is cute',
            'An Item-Item Recommender on TF-IDF distances between items'],
           dtype='<U58'),
     array([ 1.        ,  0.46750212,  0.0095222 , -0.08070645, -0.457624  ,
            -0.4939893 ], dtype=float32))

``` python
ids, scores = model.similar_items(4, N=len(corpus))

np.array(corpus)[ids], scores
```

    (array(['An Item-Item Recommender on TF-IDF distances between items',
            'This cat of mine is cute', 'The weather is amazing',
            'Playing sports on the weekend...', 'I have a cat',
            'An Item-Item Recommender on BM25 distance between items'],
           dtype='<U58'),
     array([ 1.0000001 ,  0.5336009 , -0.26059774, -0.2618611 , -0.4939893 ,
            -0.59560174], dtype=float32))
