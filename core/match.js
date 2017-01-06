
const similarity = require('../core/similarity.js')

const match = (items, person, n=3, distanceFormula) => {

  const users = Object.keys(items)
  const others = users.filter(user => user !== person)
  
  return others.reduce((output, user) => {
    const score = similarity(items, person, user, distanceFormula)
    output.push({ user, score })
    return output
  }, []).sort((a, b) => {
    return b.score[1] - a.score[1]
  }).slice(0, n)
}

module.exports = match
