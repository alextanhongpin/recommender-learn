const data = require('./data.js');
// An array of items [[4.5, 4], [...]]
function euclidean(items) {
	return items.reduce((sum, item) => {
		return sum + Math.pow(item[0] - item[1], 2);
	}, 0);
}
// User based collaborative filtering
function similarity() {
	return 1 / (1 + euclidean(...arguments));
};


// A list of users [Jill, Toby]


function calculateSimilarity(items, person1, person2, formula) {
	const users = Object.keys(items).map((user) => {
		return user;
	}).filter((target) => {
		// Exclude yourself
		return target !== person1;
	});
	console.log(person1, person2)
	const movies1 = Object.keys(items[person1]);
	const movies2 = Object.keys(items[person2]);

	

	// Compare only similar movies
	const similarMovies = movies1.filter((movie) => {
		return movies2.indexOf(movie) !== -1
	});

	// No similar movies
	if (!similarMovies.length) return;

	// Get the score
	const movieRatings = similarMovies.map((movie) => {
		return [items[person2][movie], items[person1][movie]];
	});
	return formula(movieRatings);
}

const outcome = calculateSimilarity(data, 'Sam', 'Toby', similarity);
console.log('Similarity between Sam and Toby', outcome)

function topMatches(prefs, person1, n, formulae) {
	const users = Object.keys(prefs);
	const others = users.filter((user) => {
		return user !== person1;
	})
	// a list of similarity scores
	
	return others.reduce((output, user) => {
		const score = calculateSimilarity(prefs, person1, user, formulae);
		output.push([user, score]);
		return output;
	}, []).sort((a, b) => {
		return b[1] - a[1];
	}).slice(0, n);
}

const topOutcome = topMatches(data, 'Toby', 3, similarity);
console.log('Top matches for Toby', topOutcome)
// // The Euclidean distance score
// let usersRank = [];
// users.forEach((user1) => {
// 	// map the object to an array of movies

// 	const movies1 = Object.keys(data[user1]);

// 	let similarityScore = [];
// 	users.forEach((user2) => {
// 		if (user1 !== user2) {
// 			const movies2 = Object.keys(data[user2]);
// 			const similarMovies = [];

// 			movies1.forEach((movie1) => {
// 				if (movies2.indexOf(movie1) !== -1) {
// 					similarMovies.push(movie1);
// 				}
// 			});
// 			// no similar movies
// 			if (similarMovies.length === 0) {
// 				return 0;
// 			}
// 			// for each similar movie, compute the score
// 			//let sumOfSquares = 0;
// 			const items = similarMovies.map((movie) => {
// 				const score1 = data[user1][movie];
// 				const score2 = data[user2][movie];
				
// 				//sumOfSquares += Math.pow(score1 - score2, 2);
// 				return [score1, score2];
// 			});
// 			const score = similarity(items)
// 			//similarity = 1 / (1 + sumOfSquares);
// 			console.log(`Similarity between ${ user1 } and ${ user2 }:`, score)
// 			similarityScore.push([score, user2]);
// 		}
// 	});
// 	const top = similarityScore.sort((a, b) => {
// 		return b[0] - a[0];
// 	});
// 	usersRank.push({
// 		user: user1,
// 		users: top,
// 		top1: top[0],
// 		top2: top[1],
// 		top3: top[2]
// 	});

// });

// console.log(usersRank)

// // GEt recommendation for toby
// // get recommendations for toby
// const toby = usersRank.filter((user) => {
// 	return user.user === 'Toby';
// })[0];

// console.log(toby)

// // User based collaborative filtering
function getRecommendations(prefs, person, formulae) {


	const matches = topMatches(prefs, person, prefs.length, formulae);
	const movies1 = Object.keys(prefs[person]);
	let totals = {}
	let sumSimilarity = {}
	matches.map((match) => {
		const user = match[0];
		const score = match[1];

		// Ignore scores zero or lower
		if (score <= 0) return;

		const movies2 = Object.keys(prefs[user]);

		// Only scored movies not watched by the person
		const moviesNotWatched = movies2.filter((movie) => {
			return movies1.indexOf(movie) === -1;
		});

		totals = moviesNotWatched.reduce((totals, movie) => {
			if (!totals[movie]) totals[movie] = 0;
			totals[movie] += prefs[user][movie] * score;
			return totals;
		}, totals);

		sumSimilarity = moviesNotWatched.reduce((similarity, movie) => {
			if (!similarity[movie]) similarity[movie] = 0;
			similarity[movie] +=  score;
			return similarity;
		}, sumSimilarity);
	});

	return Object.keys(totals).map((item) => {
		return [item, totals[item] / sumSimilarity[item]];
	}).sort((a, b) => {
		return b[0] - a[0]; 
	});
}
// user based collaborative filtering
const recommendation = getRecommendations(data, 'Toby', similarity);
console.log('Recommendations for Toby', recommendation)

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
		const score = topMatches(itemPrefs, movie, n=n, similarity);
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
console.log(getRecommendedMovies(data, similarity, 'Toby'));