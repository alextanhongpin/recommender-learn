
function calculateSimilarItems(prefs, n=10) {
  let results = {};

  // Invert the matrix to be item-centric
  const itemPrefs = transpose(prefs);
  const movies = Object.keys(itemPrefs);
  movies.map((movie) => {
    const score = topMatches(itemPrefs, movie, n=n, Euclidean.similarity);
    results[movie] = score;
  });
  return results;
}
console.log('Similar items to the movie', calculateSimilarItems(data, data.length));
function getRecommendedMovies(prefs, formulae, user) {
  const userRatings = prefs[user];
  let scores = {}
  let totalSim = {}

  const movies = Object.keys(userRatings);
  const similars = calculateSimilarItems(prefs, prefs.length);
  movies.map((movie) => {

    // skip movies that have been rated
    const unratedMovies = similars[movie].filter((movie) => {
      return movies.indexOf(movie[0]) === -1;
    });
    const rating = userRatings[movie];
    unratedMovies.forEach((movie) => {
      const item = movie[0];
      const score = movie[1];
      if (!scores[item]) scores[item] = 0;
      scores[item] += rating * score;

      if (!totalSim[item]) totalSim[item] = 0;
      totalSim[item] += score;
    });
  });

  const all = Object.keys(scores);

  return all.map((m) => {
    const score = scores[m];
    const total = totalSim[m];
    return [m, score / total];
  }).sort((a, b) => {
    return b[1] - a[1];
  });

}