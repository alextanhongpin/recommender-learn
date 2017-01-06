// user-based collaborative filtering

const Match = require('../core/match.js')
const Sets = require('../algorithm/sets.js')

const recommendations = (prefs, person, formula) => {
  const matches = Match(prefs, person, prefs.length, formula)
  const items1 = Object.keys(prefs[person])

  let totals = {}
  let sumSimilarity = {}

  matches.map((data) => {
    const { user, score } = data
    // Ignore scores zero or lower
    if (score <= 0) return
    const items2 = Object.keys(prefs[user])
    // Only scored movies not watched by the person
    const results = Sets.difference(items2, items1)

    totals = results.reduce((prev, item) => {
      if (!prev[item]) prev[item] = 0
      prev[item] += prefs[user][item] * score
      return totals
    }, totals)

    sumSimilarity = results.reduce((prev, item) => {
      if (!prev[item]) prev[item] = 0
      prev[item] +=  score
      return prev
    }, sumSimilarity)
  })



  return Object.keys(totals).map((item) => {
    return [item, totals[item] / sumSimilarity[item]]
  }).sort((a, b) => {
    return b[0] - a[0]
  })
}

module.exports = recommendations