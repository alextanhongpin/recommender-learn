// similarity.js
//
// similarity() is a method to compute the similarity score
// based on the given distanceFormula (euclidean, pearson).
// It can be a user based filtering or item based filtering
//
const Sets = require('../algorithm/sets.js')

const similarity = (items, person1, person2, distanceFormula) => {
  // Get a list of movies watched by the users
  const person1Items = Object.keys(items[person1])
  const person2Items = Object.keys(items[person2])

  // Compare only similar items, exclude the rest
  const results = Sets.intersection(person1Items, person2Items)
  if (!results.length) return 0

  // Get the ratings from each user for the movies
  const scores = results.map((item) => {
    const score1 = items[person1][item]
    const score2 = items[person2][item]
    return [ score1, score2 ]
  })

  // Calculate the distance based on the scores
  return distanceFormula(scores)
}

module.exports = similarity
