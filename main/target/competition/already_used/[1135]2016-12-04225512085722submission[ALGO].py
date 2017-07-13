import numpy as np
import pandas as pd
import argparse
import datetime
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import itertools as it

parser = argparse.ArgumentParser()
parser.add_argument('--rating_file', type=str, default="data/competition/interactions.csv")
parser.add_argument('--target_users', type=str, default="data/competition/target_users.csv")
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--user_item_sep', type=str, default=',')
parser.add_argument('--item_item_sep', type=str, default=' ')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--k', type=int, default=120)
parser.add_argument('--to_keep', type=int, default=110)
parser.add_argument('--shrink_factor', type=int, default=0)
parser.add_argument('--normalize', type=int, default=1)
parser.add_argument('--prediction_file', type=str, default="target/competition/"
                                                           + str(datetime.datetime.now()).replace(" ", "").
                                                           replace(".", "").replace(":", "") + "submission.csv")
parser.add_argument('--user_bias', type=int, default=0)
parser.add_argument('--item_bias', type=int, default=0)
parser.add_argument('--rec_length', type=int, default=5)
parser.add_argument('--verbosity_level', type=str, default="Info")
parser.add_argument('--number_of_cpu', type=int, default=1)
parser.add_argument('--urm_file', type=str, default=None)
parser.add_argument('--similarity_matrix_file', type=str, default=None)
parser.add_argument('--estimations_file', type=str, default=None)
parser.add_argument('--holdout_perc', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--weights_policy', type=str, default="man")
parser.add_argument('--recommend_on_pearson', type=int, default=1)
parser.add_argument('--rating_policy', type=str, default="boolean")
parser.add_argument('--help_me', type=int, default=19835)
parser.add_argument('--log_scale', type=int, default=0)
parser.add_argument('--timestamp_key', type=str, default="created_at")
args = parser.parse_args()


def estimate_quality(reco_dictionary, targeto_users, rec_length, test_set):
    dirty_map = 0.0
    number = 0
    for usero in targeto_users:
        dirty_map += map_computer(reco_dictionary[usero], pos_items=test_set[usero], at=rec_length)
        number += 1
    clean_map = dirty_map / number

    return clean_map


def map_computer(ranked_list, pos_items, at=None):
    ranked_list = ranked_list[:at]
    is_relevant = np.in1d(ranked_list, pos_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([len(pos_items), len(ranked_list)])
    assert 0 <= map_score <= 1
    return map_score


def check_expiration():
    # Returns a list in which all the items that are expired are stored
    # Import the item description file in a Pandas data frame
    filename = "data/competition/item_profile.csv"

    # This list contains the id's of all the expired items
    expired_ids = []
    with open(filename, 'r') as f:
        item_profiles_reader = pd.read_csv(filename, delimiter='\t')

    expired_items = item_profiles_reader[item_profiles_reader.active_during_test == 0]

    for i, row in expired_items.iterrows():
        expired_ids.append(row["id"])

    f.close()

    # return the list of invalid item_id
    return expired_ids


def write_recommendations(recommendations_dic, target_users, user_caption="user_id",
                          rec_item_caption="recommended_items",
                          user_items_sep=',', item_item_sep=' ',
                          prediction_file="predictions.csv"):
    # This function takes as input a dictionary composed by lists of tuples (item,estimated_rating). one for each
    # target user. User caption and rec_item_caption are the name to be printed in the header of the file,
    # user_items_sep and item_item_sep are the two separators respectively to divide the user_id from the
    # recommendations and each single recommended item_id. Prediction file is the name of the output file

    # Opening the output file
    output_file = open(prediction_file, "w+")
    dictionary_path = prediction_file.replace(".csv", "[DICTIONARY].txt")
    dictionary_file = open(dictionary_path, "w+")

    # Writing the header
    output_file.write(user_caption+user_items_sep+rec_item_caption+"\n")
    dictionary_file.write("dic = {")
    # Writing the lines per each target user
    for user in target_users:
        # Each line starts with the id of the target user and a user_item_sep
        if user in recommendations_dic.keys():
            best_choices = recommendations_dic[user]
        else:
            best_choices = []
            print("ERROR")
        line = str(user) + user_items_sep
        dic_line = str(user) + ":["
        # Then we iterate on all the recommendations for that user adding them to the string
        for recommendation in best_choices:
            line += str(recommendation[0]) + item_item_sep
            dic_line += "("+str(recommendation[0])+","+str(recommendation[1])+"),"

        dic_line += "]"
        # Removes the comma after the last tuple
        dic_line = dic_line.replace("),]", ")],")

        # At the end we write the line corresponding to that target user on the file
        output_file.write(line+"\n")
        dictionary_file.write(dic_line+"\n")

    # Last token at the end of the dictionary file
    dictionary_file.write("}\n\ndef getDictionary():\n\treturn dic")
    # Closing the resources
    output_file.close()
    dictionary_file.close()

    print("Output operations concluded")


# TODO think about how to set the weights for the top pop suggestions (are they better or worse of an item similar
# but with a rating worse than your average rating?
def non_personalized_recommendation(rec_length, non_recommendable_set):
    # For reasons of time this function is absolutely data-set dependent.
    top_100_pops = [1053452, 2778525, 1244196, 1386412, 657183, 2791339, 536047, 2002097, 1092821, 784737, 1053542,
                    278589, 79531, 1928254, 1133414, 1162250, 1984327, 343377, 1742926, 1233470, 1140869, 830073,
                    460717, 1576126, 2532610, 1443706, 1201171, 2593483, 1056667, 1754395, 1237071, 1117449, 734196,
                    437245, 266412, 2371338, 823512, 2106311, 1953846, 2413494, 2796479, 1776330, 365608, 1165605,
                    2031981, 2402625, 1679143, 2487208, 315676, 1069281, 818215, 419011, 931519, 470426, 1695664,
                    2795800, 2313894, 1119495, 2091019, 2086041, 84304, 72465, 499178, 2156629, 906846, 468120,
                    1427250, 117018, 471520, 2466095, 1920047, 1830993, 2198329, 335428, 2512859, 1500071, 2037855,
                    434392, 951143, 972388, 1047625, 2350341, 2712481, 542469, 1123592, 152021, 1244787, 1899627,
                    625711, 1330328, 2462072, 1419444, 2590849, 1486097, 1788671, 2175889, 110711,
                    16356, 291669, 313851]

    to_recommend = []
    weight = -1
    counter = 0

    for item in top_100_pops:
        if item not in non_recommendable_set:
            to_recommend.append((item, weight))
            weight -= 1
            counter += 1
            if counter == rec_length:
                break

    for i in range(0, rec_length-counter):
        to_recommend.append((0, weight))
        weight -= 1

    return to_recommend


def apply_shrinkage(dist, urm):
    x_ind = urm.copy()
    x_ind.data = np.ones_like(x_ind.data)
    co_counts = x_ind.dot(x_ind.T)
    # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
    # then multiply dist with it
    do = np.ones_like(co_counts.data) * args.shrink_factor
    do = co_counts.data / do
    dist_data = dist.data * co_counts.data / (co_counts.data + args.shrink_factor)
    dist.data = dist_data
    return dist

o_file = open(args.prediction_file.replace(".csv", "[INFO].txt"), "w+")
if args.holdout_perc != 0:
    print("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %")
o_file.write("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %\n")
print("k : " + str(args.k))
o_file.write("k : " + str(args.k) + "\n")
print("Technique : User Based Collaborative Filtering")
o_file.write("Technique : User Based Collaborative Filtering")
print("recommendations per user : " + str(args.rec_length))
o_file.write("recommendations per user : " + str(args.rec_length) + "\n")
print("Weights in the estimated ratings follow the policy : " + args.weights_policy)
o_file.write("Weights in the estimated ratings follow the policy : " + args.weights_policy + "\n")
print("Shrink factor : " + str(args.shrink_factor))
o_file.write("Shrink factor : " + str(args.shrink_factor))
print("Ratings are considered with the following policy : " + args.rating_policy)
o_file.write("Ratings are considered with the following policy : " + args.rating_policy + "\n")
if args.log_scale > 0:
    print("Logaritmic rating' shrinkage factor : " + args.rating_policy)
    o_file.write("Logaritmic rating' shringage factor : " + args.rating_policy + "\n")
if args.normalize == 1:
    print("Normalization active")
o_file.write("Normalization active" + "\n")
if args.user_bias == 1:
    print("User bias subtraction active")
o_file.write("User bias subtraction active" + "\n")
if args.item_bias == 1:
    print("Item bias subtraction active")
o_file.write("Item bias subtraction active" + "\n")
if args.recommend_on_pearson == 0:
    print("Estimated ratings are given using the biased urm")
o_file.write("Estimated ratings are given using the biased urm" + "\n")
if args.number_of_cpu > 1:
    print("Parallelization is performed using " + str(args.number_of_cpu) + " processors")
o_file.write("Parallelization is performed using " + str(args.number_of_cpu) + " processors" + "\n")
o_file.close()

with open(args.rating_file, 'r') as f:
    interactions_reader = pd.read_csv(args.rating_file, delimiter='\t')

with open(args.target_users, 'r') as j:
    target_reader = pd.read_csv(args.target_users, delimiter='\t')

users = interactions_reader[args.user_key].as_matrix()
target_users = target_reader[args.user_key].as_matrix()
items = list(set(interactions_reader[args.item_key].as_matrix()))
expired_items = check_expiration()
user_rated_items = {}
position_uid_dic = {}
uid_position_dic = {}
position_iid_dic = {}
iid_position_dic = {}
data = []
rows = []
columns = []
temp_dic = {}
item_counter = 0
user_counter = 0
rec_dictionary = {}
for i, row in interactions_reader.iterrows():
    item = row[args.item_key]
    user = row[args.user_key]
    rating = row[args.rating_key]
    if item not in iid_position_dic:
        iid_position_dic[item] = item_counter
        position_iid_dic[item_counter] = item
        item_counter += 1
    if user not in uid_position_dic:
        uid_position_dic[user] = user_counter
        position_uid_dic[user_counter] = user
        user_counter += 1
        user_rated_items[user] = []
    user_position = uid_position_dic[user]
    item_position = iid_position_dic[item]
    user_rated_items[user].append(item)
    if (user_position, item_position) not in temp_dic.keys():
        temp_dic[(user_position, item_position)] = rating
        data.append(1)
        rows.append(uid_position_dic[user])
        columns.append(iid_position_dic[item])
    else:
        temp_dic[(user_position, item_position)] += rating

removed_ratings_per_user = {}
if args.holdout_perc != 0:
    print(len(target_users))
    # set the random seed
    r = []
    c = []
    d = []
    rng = np.random.RandomState(1234)
    #  shuffle data
    keys = list(temp_dic.keys())
    number_of_ratings = len(keys)
    acc_order = keys.copy()
    shuffled_access_order = rng.permutation(acc_order)
    train_size = int(number_of_ratings * args.holdout_perc / 100)
    non_recommendable = set(check_expiration())
    train_dic = dict()
    test_target_users = []
    for key in shuffled_access_order[:train_size]:
        key_tuple = key[0], key[1]
        train_dic[key_tuple] = temp_dic[key_tuple]
        d.append(1)
        r.append(key[0])
        c.append(key[1])
    for key in shuffled_access_order[train_size:]:
        user_id = position_uid_dic[key[0]]
        item_id = position_iid_dic[key[1]]
        if user_id not in removed_ratings_per_user.keys():
            removed_ratings_per_user[user_id] = []
        if item_id not in non_recommendable:
            removed_ratings_per_user[user_id].append(item_id)
            test_target_users.append(user_id)

    new_user_rated_items = {}
    for i in range(0, len(d)):
        user = position_uid_dic[r[i]]
        if user not in new_user_rated_items.keys():
            new_user_rated_items[user] = []
        new_user_rated_items[user].append(position_iid_dic[c[i]])
    user_rated_items = new_user_rated_items

    data = d
    rows = r
    columns = c
    target_users = list(set(test_target_users))[:10000]

user_rating_matrix_non_normalized = sps.csc_matrix((data, (rows, columns)), shape=(user_counter, item_counter))
user_rating_matrix = normalize(user_rating_matrix_non_normalized, 'l2', 1, copy=True)
rating_user_matrix = user_rating_matrix.copy().transpose(copy=False)
user_user_similarity_matrix = user_rating_matrix.dot(rating_user_matrix)
if args.shrink_factor > 0:
    user_user_similarity_matrix = apply_shrinkage(user_user_similarity_matrix, user_rating_matrix)

counter = 0
non_rec_counter = 0
for target_user in target_users:
    counter += 1
    print("\n"+str(counter)),
    if target_user in user_rated_items.keys():
        position = uid_position_dic[target_user]
        filters = []
        for filler in range(0, args.k):
            filters.append(1)
        r0 = user_user_similarity_matrix.getrow(position)
        r1 = r0.todense().tolist()[0]
        # get the indices of the most similar
        b_indeces = np.argpartition(r1, -args.k)[-args.k:]
        # In order to obtain a vector with zeros in all positions but the ones of the top k similar users
        # I build a diagonal matrix with a one on every position of the diagonal with index (i, i) where i is the index
        # of one of the most similar items in the vectors and then I multiply the similarity vector for that matrix
        # obtaining a vector with all 0s except for the positions of the top k similar items
        size = user_counter
        similarity_filter = sps.csc_matrix((filters, (b_indeces, b_indeces)), shape=(size, size))
        filtered_similarity = r0.dot(similarity_filter)
        estimated_ratings = filtered_similarity.dot(user_rating_matrix_non_normalized).todense().tolist()[0]
        # I get the columns indices of the top x rated elements in order to be able to find 5 that are
        # not expired already
        k_best_matches = np.argpartition(estimated_ratings, -args.to_keep)[-args.to_keep:]
        non_zeros = list(set(np.nonzero(estimated_ratings)[0]))
        non_recommendable_set = set(it.chain(expired_items, user_rated_items[target_user]))
        # The function returns the 5 top pops more suitable for the user (not expired and not already seen)
        possible_matches = non_personalized_recommendation(args.rec_length, non_recommendable_set)
        for i in k_best_matches:
            if i in non_zeros:
                if position_iid_dic[i] not in non_recommendable_set:
                    possible_matches.append((position_iid_dic[i], estimated_ratings[i]))
        # Sorting the matches
        possible_matches.sort(key=lambda tup: tup[1], reverse=True)
        # Suggesting the best rec_lengths recommendations
        rec_dictionary[target_user] = possible_matches[:args.rec_length]
    else:
        # If the user is non profiled then just recommend the top pops.
        non_rec_counter += 1
        print("-->"+str(non_rec_counter)),
        rec_dictionary[target_user] = non_personalized_recommendation(args.rec_length, expired_items)

if args.holdout_perc != 0:
    mappo = estimate_quality(rec_dictionary, target_users, args.rec_length, removed_ratings_per_user)
    print("\nEstimated map@5: ",
          mappo)
    o_file = open(args.prediction_file.replace(".csv", "[INFO].txt"), "a")
    o_file.write("Estimated map@5: " + str(mappo))

write_recommendations(rec_dictionary, target_users, user_caption="user_id",
                      rec_item_caption="recommended_items",
                      prediction_file=args.prediction_file)


