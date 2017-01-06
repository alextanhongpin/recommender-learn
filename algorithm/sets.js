// intersect can be simulated via 
const intersection = (array1, array2) => {
  const set2 = new Set([...array2])
  return array1.filter(x => set2.has(x))
}

const difference = (array1, array2) => {
  const set2 = new Set([...array2])
  return array1.filter(x => !set2.has(x))
}

module.exports = { intersection, difference }