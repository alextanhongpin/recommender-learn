// euclidean.js
// 
// Usage:
// const Euclidean = require('./euclidean')
// 
// Euclidean.distance([[p1, q1], [p2, q2]])
// -> returns the distance score
//
//
// Euclidean.similarity([[p1, q1], [p2, q2]])
// -> returns the similarity score


const distance = (array) => {
  return array.reduce((sum, item) => {
    return sum + Math.pow(item[0] - item[1], 2)
  }, 0)
}

// The inverse is to change from distance to similarity.
function similarity () {
  return 1 / (1 + distance(...arguments))
}

module.exports = { distance, similarity }