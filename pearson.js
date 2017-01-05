const data = require('./data.js');

// An array of items [[4.5, 4], [...]]
function sum(items) {
	return items.reduce((score, item) => {
		return score + item;
	}, 0);
}

function sumSquare(items) {
	return items.reduce((score, item) => {
		return score + Math.pow(item, 2);
	}, 0);
}

function sumProducts(items) {
	return items.reduce((score, item) => {
		return score + item[0] * item[1];
	}, 0);
}

function pearson(items) {

	const item1 = items.map((item) => {
		return item[0];
	});
	const item2 = items.map((item) => {
		return item[1];
	});

	const sum1 = sum(item1);
	const sum2 = sum(item2);
	const sumSquare1 = sumSquare(item1);
	const sumSquare2 = sumSquare(item2);
	const sumOfProducts = sumProducts(items);
	const n = items.length;

	// Pearson score formulae
	const numerator = sumOfProducts - ((sum1 * sum2) / n);
	const denominator = Math.sqrt((sumSquare1 - Math.pow(sum1, 2) / n) * (sumSquare2 - Math.pow(sum2, 2) / n));

	if (denominator === 0) return 0;
	const r = numerator / denominator;
	return r;
}


const users = Object.keys(data).map((user) => {
	return user;
});

// The Euclidean distance score
let usersRank = [];
// The Euclidean distance score
users.forEach((user1) => {
	// map the object to an array of movies

	const movies1 = Object.keys(data[user1]);

	let similarityScore = []
	users.forEach((user2) => {
		if (user1 !== user2) {
			const movies2 = Object.keys(data[user2]);
			const similarMovies = [];

			movies1.forEach((movie1) => {
				if (movies2.indexOf(movie1) !== -1) {
					similarMovies.push(movie1);
				}
			});
			// no similar movies
			if (similarMovies.length === 0) {
				return 0;
			}
			// for each similar movie, compute the score
			//let sumOfSquares = 0;
			const items = similarMovies.map((movie) => {
				const score1 = data[user1][movie];
				const score2 = data[user2][movie];
				
				//sumOfSquares += Math.pow(score1 - score2, 2);
				return [score1, score2];
			});
			const score = pearson(items)
			//similarity = 1 / (1 + sumOfSquares);
			console.log(`Pearson score between ${ user1 } and ${ user2 }:`, score)
			similarityScore.push([score, user2]);
		}
	});
	const top = similarityScore.sort((a, b) => {
		return b[0] - a[0];
	});
	usersRank.push({
		user: user1,
		users: top,
		top1: top[0],
		top2: top[1],
		top3: top[2]
	});


});

//console.log(usersRank)

// get recommendations for toby
const toby = usersRank.filter((user) => {
	return user.user === 'Toby';
})[0];

function getRecommendations() {
	let totals = {}
	let similaritySums = {}
	// List of movies toby watches
	const movies1 = Object.keys(data['Toby']);
	toby.users.forEach((item) => {
		const score = item[0];
		const user = item[1];

		// ignore scores of zero or lower
		if (score <= 0) return;
		const movies2 = Object.keys(data[user]);

		// only score movies that toby have not watched
		const notWatchedMovies = movies2.filter((movie) => {
			return movies1.indexOf(movie) === -1;
		});

		notWatchedMovies.forEach((movie) => {
			const movies2Score = data[user];
			if (!totals[movie]) totals[movie] = 0;
			totals[movie] += movies2Score[movie] * score;
			if (!similaritySums[movie]) similaritySums[movie] = 0;
			similaritySums[movie] += score;
		});
		

	});

	const items = Object.keys(totals);
	const rankings = items.map((item) => {
		return [totals[item] / similaritySums[item], item];
	}).sort((a, b) => {
		return b[0] - a[0]
	});

	console.log(rankings)
}

getRecommendations()