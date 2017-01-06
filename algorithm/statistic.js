// statistic.js
// 
// Usage:
// const Statistic = require('./statistic')
// 
// Statistic.sum([n1, n2, n3,...nn])
// -> returns the sum of the array
//
//
// Statistic.sumOfSquares([n1, n2, n3,...nn])
// -> returns the sum of squares of the array
//
//
// Statistic.sumOfProducts([[p1, p2], [p3, p4], [p5, p6],...nn])
// -> returns the sum of products of the array
//

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

module.exports = { sum, sumOfSquares, sumOfProducts }