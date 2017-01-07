// match.js
//
// Description: Get a list of similar users
// in order to
//

const similarity = require('../core/similarity.js')

const match = (items, you, n = 3, distanceFormula) => {
  // List of users to be compared
  const users = Object.keys(items)

  // Exclude yourself
  const others = users.filter(user => user !== you)

  return others.reduce((output, user) => {
    // Compute similarity score
    const score = similarity(items, you, user, distanceFormula)
    output.push({ user, score })
    return output
  }, []).sort((a, b) => {
    // Sort by descending order
    return b.score - a.score
  }).slice(0, n)
}

module.exports = match
