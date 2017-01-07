// model.js is a utility to carry operations on a list of objects

function transpose (data) {
  let schema = {}
  Object.keys(data).map((key) => {
    const value = data[key]
    const valueKeys = Object.keys(value)
    const invertedObject = valueKeys.forEach((valueKey) => {
      if (!schema[valueKey]) schema[valueKey] = {}
      schema[valueKey][key] = data[key][valueKey]
    })
  })
  return schema
}

module.exports = { transpose }
