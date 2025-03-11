# Training a model


- [<span class="toc-section-number">1</span> Making
  Recommendations](#making-recommendations)
- [<span class="toc-section-number">2</span> Recommending Similar
  Items](#recommending-similar-items)
- [<span class="toc-section-number">3</span> Making Batch
  Recommendations](#making-batch-recommendations)

``` python
from implicit.datasets.lastfm import get_lastfm

# Location of the file.
# import os; os.path.join(os.path.expanduser("~"), "implicit_datasets")
artists, users, artist_user_plays = get_lastfm()
```

    0.00B [00:00, ?B/s]

``` python
artists[:10]
```

    array([' 2 ', ' 58725ab=>', ' 80lİ yillarin tÜrkÇe sÖzlÜ aŞk Şarkilari',
           ' amy winehouse', ' cours de la somme', ' fatboy slim',
           ' kanye west', ' mala rodriguez', ' mohamed lamine',
           ' oliver shanti & friends'], dtype=object)

``` python
users[:10]
```

    array(['00000c289a1829a808ac09c00daf10bc3c4e223b',
           '00001411dc427966b17297bf4d69e7e193135d89',
           '00004d2ac9316e22dc007ab2243d6fcb239e707d',
           '000063d3fe1cf2ba248b9e3c3f0334845a27a6bf',
           '00007a47085b9aab8af55f52ec8846ac479ac4fe',
           '0000c176103e538d5c9828e695fed4f7ae42dd01',
           '0000ee7dd906373efa37f4e1185bfe1e3f8695ae',
           '0000ef373bbd0d89ce796abae961f2705e8c1faf',
           '0000f687d4fe9c1ed49620fbc5ed5b0d7798ea20',
           '0001399387da41d557219578fb08b12afa25ab67'], dtype=object)

``` python
X = artist_user_plays.tocoo()
list(zip(X.row, X.col, X.data))[:10]
```

    [(np.int32(0), np.int32(73470), np.float32(32.0)),
     (np.int32(0), np.int32(97856), np.float32(24.0)),
     (np.int32(0), np.int32(235382), np.float32(1339.0)),
     (np.int32(0), np.int32(266072), np.float32(211.0)),
     (np.int32(1), np.int32(171865), np.float32(23.0)),
     (np.int32(2), np.int32(180892), np.float32(70.0)),
     (np.int32(3), np.int32(285031), np.float32(23.0)),
     (np.int32(4), np.int32(15103), np.float32(9.0)),
     (np.int32(5), np.int32(81700), np.float32(16.0)),
     (np.int32(6), np.int32(284057), np.float32(56.0))]

``` python
from implicit.nearest_neighbours import bm25_weight

# weight the matrix, both to reduce impact of users that have played the same artist thousands of times
# and to reduce the weight given to popular items
artist_user_plays = bm25_weight(artist_user_plays, K1=100, B=0.8)

# get the transpose since the most of the functions in implicit expect (user, item) sparse matrices instead of (item, user)
user_plays = artist_user_plays.T.tocsr()
```

``` python
from implicit.als import AlternatingLeastSquares

model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
model.fit(user_plays)
```

      0%|          | 0/15 [00:00<?, ?it/s]

## Making Recommendations

``` python
# Get recommendations for the a single user
userid = 12345
ids, scores = model.recommend(
    userid, user_plays[userid], N=10, filter_already_liked_items=False
)
```

``` python
# Use pandas to display the output in a table, pandas isn't a dependency of implicit otherwise
import numpy as np
import pandas as pd

pd.DataFrame(
    {
        "artist": artists[ids],
        "score": scores,
        "already_liked": np.in1d(ids, user_plays[userid].indices),
    }
)
```

    /var/folders/v5/8v9k6wcn65jbbct8spl3wwsh0000gn/T/ipykernel_63163/2636481777.py:9: DeprecationWarning: `in1d` is deprecated. Use `np.isin` instead.
      "already_liked": np.in1d(ids, user_plays[userid].indices),

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | artist                   | score    | already_liked |
|-----|--------------------------|----------|---------------|
| 0   | devil doll               | 1.071781 | False         |
| 1   | spiritual front          | 1.059486 | False         |
| 2   | mortiis                  | 1.012859 | True          |
| 3   | ordo rosarius equilibrio | 0.998481 | False         |
| 4   | rome                     | 0.998062 | True          |
| 5   | the coffinshakers        | 0.990920 | True          |
| 6   | gåte                     | 0.985904 | False         |
| 7   | arditi                   | 0.981291 | True          |
| 8   | d-a-d                    | 0.979130 | True          |
| 9   | the ark                  | 0.975668 | False         |

</div>

## Recommending Similar Items

``` python
# get related items for the beatles (itemid = 25512)
ids, scores = model.similar_items(252512)

# display the results using pandas for nicer formatting
pd.DataFrame({"artist": artists[ids], "score": scores})
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | artist             | score    |
|-----|--------------------|----------|
| 0   | the beatles        | 1.000000 |
| 1   | the beach boys     | 0.993512 |
| 2   | the rolling stones | 0.993092 |
| 3   | john lennon        | 0.992782 |
| 4   | bob dylan          | 0.992323 |
| 5   | the who            | 0.992133 |
| 6   | simon & garfunkel  | 0.991434 |
| 7   | david bowie        | 0.991320 |
| 8   | led zeppelin       | 0.990993 |
| 9   | the white stripes  | 0.989982 |

</div>

## Making Batch Recommendations

``` python
# Make recommendations for the first 1000 users in the dataset
userids = np.arange(1000)
ids, scores = model.recommend(userids, user_plays[userids])
ids, ids.shape
```

    (array([[161850, 150177, 107119, ..., 111603, 136336, 205631],
            [252956, 262990, 128505, ..., 235136, 255779, 189597],
            [186835, 113686, 142885, ..., 167270, 131061, 120981],
            ...,
            [ 83885, 151783, 265625, ..., 202346,  43598, 140971],
            [109930,   1560,  97970, ...,  33602, 236697, 129399],
            [ 21090, 158209, 276679, ..., 272293, 204087, 171553]],
           shape=(1000, 10), dtype=int32),
     (1000, 10))
