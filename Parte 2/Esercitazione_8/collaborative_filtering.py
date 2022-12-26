
#!/usr/bin/env python
# Implementation of collaborative filtering recommendation engine


from recommendation_data import dataset
from math import sqrt

def similarity_score(person1,person2):
	
	# Returns ratio Euclidean distance score of person1 and person2 

	both_viewed = {}		# To get both rated items by person1 and person2

	for item in dataset[person1]:
		if item in dataset[person2]:
			both_viewed[item] = 1

		# Conditions to check they both have an common rating items	
		if len(both_viewed) == 0:
			return 0

		# Finding Euclidean distance 
		sum_of_eclidean_distance = []	

		for item in dataset[person1]:
			if item in dataset[person2]:
				sum_of_eclidean_distance.append(pow(dataset[person1][item] - dataset[person2][item],2))
		sum_of_eclidean_distance = sum(sum_of_eclidean_distance)

		return 1/(1+sqrt(sum_of_eclidean_distance))

def pearson_correlation(person1,person2):

	# To get both rated items
	both_rated = {}
	for item in dataset[person1]:
		if item in dataset[person2]:
			both_rated[item] = 1

	number_of_ratings = len(both_rated)		
	
	# Checking for number of ratings in common
	if number_of_ratings == 0:
		return 0

	# Add up all the preferences of each user
	person1_preferences_sum = sum([dataset[person1][item] for item in both_rated])
	person2_preferences_sum = sum([dataset[person2][item] for item in both_rated])

	# Sum up the squares of preferences of each user
	person1_square_preferences_sum = sum([pow(dataset[person1][item],2) for item in both_rated])
	person2_square_preferences_sum = sum([pow(dataset[person2][item],2) for item in both_rated])

	# Sum up the product value of both preferences for each item
	product_sum_of_both_users = sum([dataset[person1][item] * dataset[person2][item] for item in both_rated])

	# Calculate the pearson score
	numerator_value = product_sum_of_both_users - (person1_preferences_sum*person2_preferences_sum/number_of_ratings)
	denominator_value = sqrt((person1_square_preferences_sum - pow(person1_preferences_sum,2)/number_of_ratings) * (person2_square_preferences_sum -pow(person2_preferences_sum,2)/number_of_ratings))
	if denominator_value == 0:
		return 0
	else:
		r = numerator_value/denominator_value
		return r 

def most_similar_users(person,number_of_users):
	# returns the number_of_users (similar persons) for a given specific person.
	scores = [(pearson_correlation(person,other_person),other_person) for other_person in dataset if  other_person != person ]
	
	# Sort the similar persons so that highest scores person will appear at the first
	scores.sort()
	scores.reverse()
	return scores[0:number_of_users]

def user_recommendations(my_person, choice: str):

	# Gets recommendations for a person by using a weighted average of every other user's rankings
	totals = {}
	simSums = {}
	rankings_list = []

	for other in dataset:
		# don't compare me to myself
		if other == my_person:
			continue

		# scelta della funzione di similarity
		if choice == "pearson":
			sim = pearson_correlation(my_person, other)
		elif choice == "euclidea":
			sim = similarity_score(my_person, other)

		# ignore scores of zero or lower
		if sim <= 0:		# scartiamo l'utente 'other'
			continue

		for film in dataset[other]:

			# only score movies i haven't seen yet
			if film not in dataset[my_person] or dataset[my_person][film] == 0:

				# Similarity * score
				totals.setdefault(film,0)
				totals[film] += dataset[other][film]* sim

				# sum of similarities
				simSums.setdefault(film,0)
				simSums[film] += sim

		# STAMPA DELLE INFORMAZIONI
		print(f"\n{other}, SIMILARITY: {round(sim,2)}")
		for film in dataset[other]:		#calcoliamo la similarità solo per i film non visti da Toby
			if film not in dataset[my_person]:
				print(f"Film: {film} "
					  f"- Voto: {round(dataset[other][film],2)} "
					  f"- Film_Similarity: {round(dataset[other][film] * sim, 2)}")	#in questa riga vi è il prodotto tra l'indice di similarità ed il film

	# Create the normalized list
	rankings = [(total/simSums[item],item) for item,total in totals.items()]
	rankings.sort()
	rankings.reverse()

	print("\nRankings")
	for rank in rankings:
		print(f"{rank}")

	# returns the recommended items
	recommendataions_list = [recommend_item for score,recommend_item in rankings]
	return recommendataions_list
		
def calculateSimilarItems(prefs,n=10):
	# Create a dictionary of items showing which other items they
	# are most similar to.
	result={}
	# Invert the preference matrix to be item-centric
	itemPrefs=transformPrefs(prefs)
	c=0
	for item in itemPrefs:
		# Status updates for large datasets
		c+=1
		if c%100==0: print("%d / %d" % (c,len(itemPrefs)))
		# Find the most similar items to this one
		scores=topMatches(itemPrefs,item,n=n,similarity=sim_distance)
		result[item]=scores
	return result

def recommendations_approximation(user, product):

	# Stimiamo la valutazione di 'user' per 'product'
	average_vote_user = 0.0
	sum_sims = 0.0
	sum_sims_weighted = 0.0

	# Calcolo la media delle valutazioni di 'user'
	for film in dataset[user]:
		average_vote_user += dataset[user][film]
	average_vote_user /= len(dataset[user])

	# Per ogni altro utente diverso e che ha valutato 'product'
	for other_user in dataset:
		average_local = 0.0

		if other_user == user:
			continue

		if product in dataset[other_user]:
			# ne calcolo la similarità secondo la formula di Pearson (come richiesto dalla traccia)
			sim = pearson_correlation(user, other_user)

			# calcolo il voto medio di tutti i film che ha vist 'other_user'
			for other_item in dataset[other_user]:
				average_local += dataset[other_user][other_item]
			average_local /= len(dataset[other_user])

			# aggiorno le somme delle simillarità
			voto_film_other_user = dataset[other_user][product]
			sum_sims_weighted += sim * (voto_film_other_user - average_local)
			sum_sims += sim

	approx_vote = average_vote_user + (sum_sims_weighted / sum_sims)

	return {product: approx_vote}

if __name__ == "__main__" :
	print("\n\033[1m Misura di Similarità: Correlazione di Pearson \033[0m")
	print(f"\nFilm consigliati per Toby: {user_recommendations('Toby','pearson')}")
	print("\n---------------------------------------")

	print("\n\033[1m Misura di Similarità: Distanza Euclidea \033[0m")
	print(f"\nFilm consigliati per Toby: {user_recommendations('Toby','euclidea')}")
	print("\n---------------------------------------")

	print("\n\033[1m Stime di Valutazioni: \033[0m")
	for item in ['The Night Listener', 'Lady in the Water', 'Just My Luck']: #vogliamo vedere quanto sono raccomandabili questi film a Toby
		print(f"{recommendations_approximation('Toby', item)}")

	print("Il film più raccomandato è 'Lady in the Water' e a seguire abbiamo 'The Night Listener' e 'Just My Luck'")

	print("\n---------------------------------------")