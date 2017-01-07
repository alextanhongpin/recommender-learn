// pearson.js
//
// Usage:
// const Pearson = require('./pearson')
//
//
// Pearson.distance([[p1, q1], [p2, q2]])
// -> returns the distance score
//

const Statistic = require('../algorithm/statistic.js')

const distance = (items) => {
  const item1 = items.map(p => p[0])
  const item2 = items.map(q => q[1])
  const sum1 = Statistic.sum(item1)
  const sum2 = Statistic.sum(item2)
  const sumSquare1 = Statistic.sumOfSquares(item1)
  const sumSquare2 = Statistic.sumOfSquares(item2)
  const sumOfProducts = Statistic.sumOfProducts(items)
  const n = items.length

  // Pearson score formulae
  // The distance covariance = n
  const numerator = sumOfProducts - ((sum1 * sum2) / n)
  // The distance standard deviation
  const denominator = Math.sqrt((sumSquare1 - Math.pow(sum1, 2) / n) * (sumSquare2 - Math.pow(sum2, 2) / n))

  if (denominator === 0) return 0
  // distanceCorrelation
  const r = numerator / denominator
  return r
}

// The inverse is to change from distance to similarity.
function similarity () {
  return distance(...arguments)
  // return 1 / (1 + distance(...arguments))
}
module.exports = { distance, similarity }
