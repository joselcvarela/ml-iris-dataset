const kNN = require('ml-knn')
const csv2json = require('csvtojson')
const prompt = require('prompt')

const csvPath = __dirname + "/iris.csv"

let knn
let separationSize
let data = []
let trainingSetX = []
let trainingSetY = []
let testX = []
let testY = []

const shuffleArray = array => {
  const max = ~~(array.length / 2)
  for (let i = max; i >= 0; i--) {
    const j = ~~(Math.random() * array.length - max) + max;
    var temp = array[i];
    array[i] = array[j];
    array[j] = temp;
  }
  return array;
}

const predict = () => {
  let temp = []

  prompt.start()
  prompt.get(['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'], (err, res) => {
    if (err) return
    temp = Object.values(res).map(val => parseFloat(val))
    console.log(`Prediction: ${knn.predict(temp)}`)
  })
}

const error = (predicted, expected) => predicted.reduce((acc, curr, index) => {
  return curr !== expected[index] ? acc + 1 : acc
}, 0)

const test = () => {
  const result = knn.predict(testSetX)
  const predictionError = error(result, testSetY)
  console.log(`Test size: ${testSetX.length}`)
  console.log(`Misclassification: ${predictionError}`)
  predict()
}

const train = () => {
  knn = new kNN(trainingSetX, trainingSetY, { k: 7 })
  test()
}

const dressData = () => {
  let types = new Set();
  let X = []
  let y = []

  data.forEach((row) => {
    types.add(row['species'])
  })

  const typesArray = Array.from(types);

  data.forEach((row) => {
    X.push((Object.values(row).map(val => parseFloat(val))).slice(0, 4))
    y.push(typesArray.indexOf(row['species']))
  })

  trainingSetX = X.slice(0, separationSize);
  trainingSetY = y.slice(0, separationSize);
  testSetX = X.slice(separationSize);
  testSetY = y.slice(separationSize);

  train()
}

csv2json().fromFile(csvPath)
  .on('json', jsonObj => { data.push(jsonObj) })
  .on('done', error => {
    separationSize = ~~(.7 * data.length);
    data = shuffleArray(data)
    dressData()
  })