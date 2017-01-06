const data = require('./data.js')
const Euclidean = require('./algorithm/euclidean.js')
const Sets = require('./algorithm/sets.js')

const Similarity = require('./core/similarity.js')
const Match = require('./core/match.js')
const UserBased = require('./core/user-based.js')

const person1 = 'Sam'
const person2 = 'Toby'
const outcome = Similarity(data, person1, person2, Euclidean.similarity)

console.log(`Similarity between ${person1} and ${person2}: ${outcome.toFixed(5)}`)

const matches = Match(data, person1, 3, Euclidean.similarity)
console.log(`Top matches for ${person1}: `, matches)


// user based collaborative filtering
const recommendation = UserBased(data, person1, Euclidean.similarity)
console.log(`Recommendations for ${person1}`, recommendation)

function transpose(data) {
	const users = Object.keys(data);
	let results = {}
	users.map((user) => {
		const movies = Object.keys(data[user]);
		movies.forEach((movie) => {
			if (!results[movie]) results[movie] = {}

			results[movie][user] = data[user][movie]; 
		});
	});
	return results;
}

// item based collaborative filtering
console.log(transpose(data))


function calculateSimilarItems(prefs, n=10) {
	let results = {};

	// Invert the matrix to be item-centric
	const itemPrefs = transpose(prefs);
	const movies = Object.keys(itemPrefs);
	movies.map((movie) => {
		const score = Match(itemPrefs, movie, n=n, Euclidean.similarity);
		results[movie] = score;
	});
	return results;
}
console.log('Similar items to the movie', calculateSimilarItems(data, data.length));
function getRecommendedMovies(prefs, formulae, user) {
	const userRatings = prefs[user];
	let scores = {}
	let totalSim = {}

	const movies = Object.keys(userRatings);
	const similars = calculateSimilarItems(prefs, prefs.length);
	movies.map((movie) => {

		// skip movies that have been rated
		const unratedMovies = similars[movie].filter((movie) => {
			return movies.indexOf(movie[0]) === -1;
		});
		const rating = userRatings[movie];
		unratedMovies.forEach((movie) => {
			const item = movie[0];
			const score = movie[1];
			if (!scores[item]) scores[item] = 0;
			scores[item] += rating * score;

			if (!totalSim[item]) totalSim[item] = 0;
			totalSim[item] += score;
		});
	});

	const all = Object.keys(scores);

	return all.map((m) => {
		const score = scores[m];
		const total = totalSim[m];
		return [m, score / total];
	}).sort((a, b) => {
		return b[1] - a[1];
	});

}
console.log(getRecommendedMovies(data, Euclidean.similarity, 'Toby'));