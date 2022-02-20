const data = require("./data/data.js");

// Algorithms
const Euclidean = require("./algorithm/euclidean.js");
const Pearson = require("./algorithm/pearson.js");

const Match = require("./core/match.js");
const UserBased = require("./core/user-based.js");
const ItemBased = require("./core/item-based.js");
const Similarity = require("./core/similarity.js");

const person1 = "Toby";
const person2 = "William";
const resultsWithEuclidean = Similarity(
  data,
  person1,
  person2,
  Euclidean.similarity
);
const resultsWithPearson = Similarity(
  data,
  person1,
  person2,
  Pearson.similarity
);

console.log(
  `Similarity between ${person1} and ${person2} with Euclidean: ${resultsWithEuclidean.toFixed(
    5
  )}`
);
console.log("\n");
console.log(
  `Similarity between ${person1} and ${person2} with Pearson: ${resultsWithPearson.toFixed(
    5
  )}`
);
console.log("\n");

const matchesWithEuclidean = Match(data, person1, 3, Euclidean.similarity);
console.log(`Top matches for ${person1} with Euclidean: `);
matchesWithEuclidean.forEach((p) => console.log(p.user, p.score));
console.log("\n");

const matchesWithPearson = Match(data, person1, 3, Pearson.similarity);
console.log(`Top matches for ${person1} with Pearson: `);
matchesWithPearson.forEach((p) => console.log(p.user, p.score));
console.log("\n");

// user based collaborative filtering
const recommendation = UserBased(data, person1, Euclidean.similarity);
console.log(`Recommendations for ${person1}`);
recommendation.forEach((r) => console.log(r.user, r.score));
console.log("\n");

console.log(ItemBased.recommendations(data, Euclidean.similarity, "Toby"));
