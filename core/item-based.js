
const Match = require('../core/match.js')
const Model = require('../core/model.js')
const Euclidean = require('../algorithm/euclidean.js')

// Item-based collaborative filtering
function similarity (prefs, limit = 10) {
  // Invert the matrix to be item-centric
  const transposed = Model.transpose(prefs)
  return Object.keys(transposed).reduce((prev, movie) => {
    const results = Match(transposed, movie, limit, Euclidean.similarity)
    prev[movie] = results.map((result) => {
      return {
        // Map user -> item
        item: result.user,
        score: result.score
      }
    })
    return prev
  }, {})
}

function recommendations (prefs, formulae, user) {
  const similars = similarity(prefs, prefs.length)
  const userRatings = prefs[user]
  const movies = Object.keys(userRatings)
  let scores = {}
  let totalSim = {}

  movies.map((movie) => {
    // skip movies that have been rated
    const unratedMovies = similars[movie].filter((movie) => {
      return movies.indexOf(movie[0]) === -1
    })
    const rating = userRatings[movie]
    unratedMovies.forEach((result) => {
      const { item, score } = result
      if (!scores[item]) scores[item] = 0
      scores[item] += rating * score

      if (!totalSim[item]) totalSim[item] = 0
      totalSim[item] += score
    })
  })

  const all = Object.keys(scores)

  return all.map((m) => {
    const score = scores[m]
    const total = totalSim[m]
    return {
      item: m,
      score: score / total
    }
  }).sort((a, b) => {
    return b.score - a.score
  })
}

module.exports = { recommendations }
