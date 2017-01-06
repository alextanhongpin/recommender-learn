// model.js is a utility to carry operations on a list of objects

const keys = (obj) => {
  return Object.keys(obj)
}


function transpose(data) {

  let schema = {}
  Object.keys(data).map((key) => {
    const value = data[key]
    const itemKeys = Object.keys(value)
    const invertedObject = itemKeys.forEach((value) => {
      if (!schema[value]) schema[value] = {}
      schema[value][key] = data[key][value]
    })
  })
  return schema
}