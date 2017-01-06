// Description: euclidean() is a method that calculates the 
// distance between two points in Euclidean space.
// @items is a pair of array distance [[p1, q1], [p2, q2]...]
const euclidean = (array) => {
  return array.reduce((sum, item) => {
    return sum + Math.pow(item[0] - item[1], 2)
  }, 0)
}

// The inverse is to change from distance to similarity.
function similarity () {
  return 1 / (1 + euclidean(...arguments))
}

const sum = (array, operation = x => x) => {
  return array.reduce((total, amount) => {
    return total + operation(amount)
  }, 0)
}


const sumOfSquares = (array) => {
  return sum(array, amount => {
    return Math.pow(amount, 2)
  })
}

const sumOfProducts = (array) => {
  return sum(array, (item) => {
    return item[0] * item[1]
  })
}

const selectFromArray = (array, index) => {
  return array.map(item => item[index] )
}

const pearson = (items) => {
  const item1 = items.map(p => p[0])
  const item2 = items.map(q => q[1])
  const sum1 = sum(item1)
  const sum2 = sum(item2)
  const sumSquare1 = sumOfSquares(item1)
  const sumSquare2 = sumOfSquares(item2)
  const sumOfProducts = sumOfProducts(items)
  const n = items.length

  // Pearson score formulae
  const distanceCovariance = n = sumOfProducts - ((sum1 * sum2) / n)
  const distanceStandardDeviations = d = Math.sqrt((sumSquare1 - Math.pow(sum1, 2) / n) * (sumSquare2 - Math.pow(sum2, 2) / n))

  if (d === 0) return 0;
  const distanceCorrelation = c = n / d
  return c
}

// intersect can be simulated via 
const intersection = (array1, array2) => {
  const set1 = new Set([...array1])
  const set2 = new Set([...array2])
  return [...set1].filter(x => set2.has(x))
}


module.exports = { euclidean, pearson, similarity, sum, sumOfSquares, sumOfProducts, selectFromArray, intersection }