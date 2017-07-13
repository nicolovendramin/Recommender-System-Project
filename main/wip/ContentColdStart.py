import math as m
import numpy as np
import pandas as pd
import argparse
import datetime
import scipy.sparse as sps
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize
import multiprocessing
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
parser.add_argument('--to_keep', type=int, default=150)
parser.add_argument('--shrink_factor', type=int, default=8)
parser.add_argument('--normalize', type=int, default=1)
parser.add_argument('--prediction_file', type=str, default="target/competition/"
                                                           + str(datetime.datetime.now()).replace(" ", "").
                                                           replace(".", "").replace(":", "") + "submission.csv")
parser.add_argument('--user_bias', type=int, default=0)
parser.add_argument('--item_bias', type=int, default=0)
parser.add_argument('--rec_length', type=int, default=5)
parser.add_argument('--show_progress', type=str, default=1)
parser.add_argument('--number_of_cpu', type=int, default=1)
parser.add_argument('--holdout_perc', type=int, default=0)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--weights_policy', type=str, default="man")
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
    dist_data = dist.data * co_counts.data / (co_counts.data + args.shrink_factor)
    dist.data = dist_data
    return dist


def print_run_details():
    o_file = open(args.prediction_file.replace(".csv", "[INFO].txt"), "w+")
    if args.holdout_perc != 0:
        print("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %")
    o_file.write("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %\n")
    print("k : " + str(args.k))
    o_file.write("k : " + str(args.k) + "\n")
    print("Technique : User Based Collaborative Filtering with Content Based Cold Start")
    o_file.write("Technique : User Based Collaborative Filtering with Content Based Cold Start\n")
    print("recommendations per user : " + str(args.rec_length))
    o_file.write("recommendations per user : " + str(args.rec_length) + "\n")
    print("Weights in the estimated ratings follow the policy : " + args.weights_policy)
    o_file.write("Weights in the estimated ratings follow the policy : " + args.weights_policy + "\n")
    print("Shrink factor : " + str(args.shrink_factor))
    o_file.write("Shrink factor : " + str(args.shrink_factor)+"\n")
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
    o_file.write("Estimated ratings are given using the biased urm" + "\n")
    if args.number_of_cpu > 1:
        print("Parallelization is performed using " + str(args.number_of_cpu) + " processors")
    o_file.write("Parallelization is performed using " + str(args.number_of_cpu) + " processors" + "\n")
    o_file.close()


def print_help():
    print('   --rating_file : to indicate the path of the file with interactions  ' +
          '\n   --target_users : to indicate the path of the file with target_users  ' +
          '\n   --rating_key, item_key, user_key : the keys with which ratings, items and users are '
          'labeled in the data set  ' +
          '\n   --user_item_sep, item_item_sep : the separator to be use in the rec. '
          'file to separate user_id from item_id and each item  ' +
          '\n   --k : the number of neighbours to be considered  ' +
          '\n   --normalize : 0 to avoid normalization 1 to do it  ' +
          '\n   --prediction_file : the path of the file where to write recommendations  ' +
          '\n   --user_bias, item_bias : 0 not to consider the specified bias, 1 otherwise  ' +
          '\n   --rec_lenght : number of items to recommend to each user  ' +
          '\n   --number_of_cpu : parameter to decide on how many processor to split the computational load  ' +
          '\n   --holdout_perc : the percentage of interactions to include in the train set  ' +
          '\n   --seed : the seed of the random function used to shuffle the interactions to randomly select '
          'the train set  ' +
          '\n   --weights_policy : is in {man, mar, constant} man uses all the similarities, mar only the ones'
          ' of the rankers '
          'and constant doesn\'t divide  ' +
          '\n   --rating_policy : is in {sum, max, boolean, decremental_a, decremental_d}'
          ' sum sums interactions, max picks the maximum '
          ',boolean consider 1 if there is at least one otherwise 0 and decremental_a (decremental_d)'
          ' shrinks new (past) interactions by the '
          'value of the past (new) ones'
          '\n   --shrink_factor : is the shrink factor to be considered'
          '\n   --log_scale : is the base of the logarithm used to scale data [if used use also recommend_on_pearson]'
          ' and remember that the higher is the base the major is the shrinkage [eg. 2 maps 1:100 in 0:6, 10 maps it in'
          '0:2')


def main():
    if args.help_me != 19835:
        print_help()
        return

    print_run_details()

    expired_items = check_expiration()

    user_user_similarity_matrix, user_rated_items, target_users, user_rating_matrix_non_normalized, \
        uid_position_dic, user_counter, position_iid_dic, removed_ratings_per_user, user_user_content_similarity \
        = similarity_computer()
    rec_dictionary, non_recommended_users_number = recommend_all(target_users, uid_position_dic,
                                                                 user_user_similarity_matrix,
                                                                 user_rating_matrix_non_normalized, expired_items,
                                                                 user_counter, user_rated_items, position_iid_dic,
                                                                 user_user_content_similarity)

    print("the total number of top-pops recommendations is "+str(non_recommended_users_number))

    if args.holdout_perc != 0:
        calculated_map = estimate_quality(rec_dictionary, target_users, args.rec_length, removed_ratings_per_user)
        print("\nEstimated map@5: ",
              calculated_map)
        o_file = open(args.prediction_file.replace(".csv", "[INFO].txt"), "a")
        o_file.write("\nEstimated map@5: " + str(calculated_map))

    write_recommendations(rec_dictionary, target_users, user_caption="user_id",
                          rec_item_caption="recommended_items", item_item_sep=args.item_item_sep,
                          prediction_file=args.prediction_file, user_items_sep=args.user_item_sep)


def similarity_computer():
    with open(args.rating_file, 'r') as f:
        interactions_reader = pd.read_csv(args.rating_file, delimiter='\t')

    with open(args.target_users, 'r') as j:
        target_reader = pd.read_csv(args.target_users, delimiter='\t')

    users = interactions_reader[args.user_key].as_matrix()
    target_users = target_reader[args.user_key].as_matrix()
    items = list(set(interactions_reader[args.item_key].as_matrix()))
    user_rated_items = {}
    position_uid_dic = {}
    uid_position_dic = {}
    position_iid_dic = {}
    iid_position_dic = {}
    d = []
    r = []
    c = []
    interactions_per_user = {}
    item_counter = 0
    user_counter = 0
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
        user_rated_items[user].append(item)
        d.append(rating)
        r.append(uid_position_dic[user])
        c.append(iid_position_dic[item])
        if user not in interactions_per_user.keys():
            interactions_per_user[user] = 1
        else:
            interactions_per_user[user] += 1

    for usr in target_users:
        if usr not in users:
            uid_position_dic[usr] = user_counter
            position_uid_dic[user_counter] = usr
            user_counter += 1

    f.close()
    j.close()

    temp_dic = rating_processing(d, r, c)

    removed_ratings_per_user = {}
    if args.holdout_perc != 0:
        # set the random seed
        rng = np.random.RandomState(args.seed)
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
        for key in shuffled_access_order[train_size:]:
            user_id = position_uid_dic[key[0]]
            item_id = position_iid_dic[key[1]]
            if user_id not in removed_ratings_per_user.keys():
                removed_ratings_per_user[user_id] = []
            if item_id not in non_recommendable:
                removed_ratings_per_user[user_id].append(item_id)
                test_target_users.append(user_id)

        new_user_rated_items = {}
        for i in train_dic.keys():
            user = position_uid_dic[i[0]]
            item = position_iid_dic[i[1]]
            if user not in new_user_rated_items.keys():
                new_user_rated_items[user] = []
            new_user_rated_items[user].append(item)
        user_rated_items = new_user_rated_items

        temp_dic = train_dic
        target_users = list(set(test_target_users))[:len(target_users)]

    data, rows, columns = sparse_arrays_extractor(temp_dic)

    user_rating_matrix_non_normalized = sps.csc_matrix((data, (rows, columns)), shape=(user_counter, item_counter))
    user_rating_matrix = normalize(user_rating_matrix_non_normalized, 'l2', 1, copy=True)
    rating_user_matrix = user_rating_matrix.copy().transpose(copy=False)
    user_user_similarity_matrix = user_rating_matrix.dot(rating_user_matrix)
    if args.shrink_factor > 0:
        user_user_similarity_matrix = apply_shrinkage(user_user_similarity_matrix, user_rating_matrix)
    user_user_content_similarity = content_similarity(users, target_users, uid_position_dic, item_counter, user_counter)

    return user_user_similarity_matrix, user_rated_items, target_users, user_rating_matrix_non_normalized, \
        uid_position_dic, user_counter, position_iid_dic, removed_ratings_per_user, user_user_content_similarity


def recommend(target_users, uid_position_dic, user_user_similarity_matrix, user_rating_matrix_non_normalized,
              expired_items, user_counter, user_rated_items, position_iid_dic, user_user_content_similarity):
    rec_dictionary = {}
    counter = 0
    non_rec_counter = 0
    for target_user in target_users:
        counter += 1
        print(str(counter)),
        position = uid_position_dic[target_user]
        filters = []
        for filler in range(0, args.k):
            filters.append(1)
        # In case the user is active I use CF similarity
        if target_user in user_rated_items.keys():
            r0 = user_user_similarity_matrix.getrow(position)
        # Otherwise I use CB similarity
        else:
            r0 = user_user_content_similarity.getrow(position)
            user_rated_items[target_user] = []
        r1 = r0.todense().tolist()[0]
        # get the indices of the most similar
        b_indexes = np.argpartition(r1, -args.k)[-args.k:]
        # In order to obtain a vector with zeros in all positions but the ones of the top k similar users
        # I build a diagonal matrix with a one on every position of the diagonal with index (i, i) where i is the
        # index of one of the most similar items in the vectors and then I multiply the similarity vector for that
        # matrix obtaining a vector with all 0s except for the positions of the top k similar items
        size = user_counter
        similarity_filter = sps.csc_matrix((filters, (b_indexes, b_indexes)), shape=(size, size))
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
                    item_id = position_iid_dic[i]
                    possible_matches.append((item_id, estimated_ratings[i]))
        # Sorting the matches
        possible_matches.sort(key=lambda tup: tup[1], reverse=True)
        # Suggesting the best rec_lengths recommendations
        rec_dictionary[target_user] = possible_matches[:args.rec_length]

    return rec_dictionary, non_rec_counter


def recommend_all(target_users, uid_position_dic, user_user_similarity_matrix, user_rating_matrix_non_normalized,
                  expired_items, user_counter, user_rated_items, position_iid_dic, user_user_content_similarity):
    # Splits the work between 4 worker in order to exploit all the 4 CPUs of my machine
    pool = multiprocessing.Pool()
    tot = len(target_users)
    step = m.floor(tot / args.number_of_cpu)

    blocks = []
    for i in range(0, args.number_of_cpu):
        start = i * step
        end = (i + 1) * step
        if i == args.number_of_cpu - 1:
            end = tot
        blocks.append(pool.apply_async(recommend, (target_users[start:end], uid_position_dic,
                                                   user_user_similarity_matrix,
                                                   user_rating_matrix_non_normalized, expired_items, user_counter,
                                                   user_rated_items, position_iid_dic, user_user_content_similarity),))

    results = []
    # Fetching of the results from the workers who run the tasks
    for j in range(0, args.number_of_cpu):
        results.append(blocks[j].get())

    rec_dictionary = dict(results[0][0])
    non_profiled_users_number = results[0][1]
    for i in range(1, args.number_of_cpu):
        non_profiled_users_number += results[i][1]
        rec_dictionary.update(results[i][0])

    return rec_dictionary, non_profiled_users_number


def rating_processing(d, r, c):
    temp_dic = {}
    tot = len(d)
    if "boolean" in args.rating_policy:
        for i in range(0, tot):
            temp_dic[(r[i], c[i])] = 1
    if "sum" in args.rating_policy:
        keys = set()
        for i in range(0, tot):
            key = r[i], c[i]
            if key in keys:
                temp_dic[key] += d[i]
            else:
                keys.add(key)
                temp_dic[key] = d[i]
    if "num" in args.rating_policy:
        keys = set()
        for i in range(0, tot):
            key = r[i], c[i]
            if key in keys:
                temp_dic[key] += 1
            else:
                keys.add(key)
                temp_dic[key] = 1
    if "max" in args.rating_policy:
        keys = set()
        for i in range(0, tot):
            key = r[i], c[i]
            if key in keys:
                temp_dic[key] = max(temp_dic[key], d[i])
            else:
                keys.add(key)
                temp_dic[key] = d[i]
    if "log" in args.rating_policy:
        for key in temp_dic.keys():
            logaritmic = m.log2(temp_dic[key])
            temp_dic[key] = 1 + logaritmic
    if "offset" in args.rating_policy:
        if "_m_" in args.rating_policy:
            keys = set()
            maximums = {}
            for i in range(0, tot):
                key = r[i], c[i]
                if key in keys:
                    maximums[key] = max(maximums[key], d[i])
                else:
                    keys.add(key)
                    maximums[key] = d[i]
            for key in temp_dic.keys():
                temp_dic[key] += maximums[key] - 1
        if "_n_" in args.rating_policy:
            keys = set()
            nums = {}
            for i in range(0, tot):
                key = r[i], c[i]
                if key in keys:
                    nums[key] += 1
                else:
                    keys.add(key)
                    nums[key] = 1
            for key in temp_dic.keys():
                temp_dic[key] += nums[key] - 1

    return temp_dic


def sparse_arrays_extractor(temp_dic):
    data, rows, columns = [], [], []
    user_totals, user_number, item_totals, item_number = {}, {}, {}, {}
    max_user_average, max_item_average = 0, 0

    for key in temp_dic.keys():
        item = key[1]
        user = key[0]
        rating = temp_dic[key]
        if user not in user_totals.keys():
            user_totals[user] = 0
            user_number[user] = 0
        if item not in item_totals.keys():
            item_totals[item] = 0
            item_number[item] = 0
        user_totals[user] = (user_totals[user] * user_number[user] + rating) / (user_number[user] + 1)
        item_totals[item] = (item_totals[item] * item_number[item] + rating) / (item_number[item] + 1)
        user_number[user] += 1
        item_number[item] += 1
        if max_user_average < user_totals[user]:
            max_user_average = user_totals[user]
        if max_item_average < item_totals[item]:
            max_item_average = item_totals[item]

    for key in temp_dic.keys():
        data.append(temp_dic[key]
                    + max_user_average*args.user_bias + max_item_average*args.item_bias + 0.5
                    - user_totals[key[0]]*args.user_bias - item_totals[key[1]]*args.item_bias)
        rows.append(key[0])
        columns.append(key[1])
    return data, rows, columns


# Function to map the age to the probability to recommend it
def item_time_weights():
    # Returns a list in which all the items that are expired are stored
    # Import the item description file in a Pandas data frame
    filename = "data/competition/item_profile.csv"

    # This list contains the id's of all the expired items
    with open(filename, 'r') as f:
        item_profiles_reader = pd.read_csv(filename, delimiter='\t')
    item_time_weight_dic = {}
    maximum_ts = int(item_profiles_reader["created_at"].max())
    maximum_date = datetime.datetime.fromtimestamp(maximum_ts).strftime('%Y-%m-%d %H:%M:%S')
    minimum_ts = int(item_profiles_reader["created_at"].min())
    minimum_date = datetime.datetime.fromtimestamp(minimum_ts).strftime('%Y-%m-%d %H:%M:%S')
    one_third = minimum_ts + (maximum_ts - minimum_ts) / 3
    two_third = minimum_ts + 2 * (maximum_ts - minimum_ts) / 3
    middle = minimum_ts + (maximum_ts - minimum_ts) / 2
    interpolation_function = interp1d([minimum_ts, one_third, middle, two_third, maximum_ts], [0.2, 0.75, 1, 0.75, 0.5])
    for i, row in item_profiles_reader.iterrows():
        timestamp = row.created_at
        if timestamp is None:
            timestamp = middle
        item_time_weight_dic[row.id] = interpolation_function(timestamp) + 0.5

    print("The data are relative to the period between "+str(minimum_date)+" and "+str(maximum_date))
    f.close()

    # return the list of invalid item_id
    return item_time_weight_dic


def content_similarity(users, target_users, uid_position_dic, item_counter, user_counter):
    # This list contains the id's of all the expired items
    user_file = "data/competition/user_profile.csv"
    with open(user_file, 'r') as f:
        user_profiles_reader = pd.read_csv(user_file, delimiter='\t')
    data = []
    rows = []
    columns = []
    columns_counter = {}
    counter = 0
    c, jro, cl, di, ii, ene, eye, ed, efos = [], [], [], [], [], [], [], [], []
    country_mapper, discipline_id_mapper, industry_id_mapper, experience_years_experience_mapper, \
        experience_n_entries_mapper, edu_degrees_mapper, edu_fieldofstudies_mapper, job_roles_mapper, \
        career_level_mapper = {}, {}, {}, {}, {}, {}, {}, {}, {}
    offsets = [0, 4455, 4462, 4486, 4510, 4516, 4520, 4528, 4532]
    all_usrs = list(set(np.append(users, target_users)))
    for i, row in user_profiles_reader.iterrows():
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        if row.user_id in all_usrs:
            data_temp = []
            ones = 0
            row_ind = uid_position_dic[row.user_id]
            country = row.country
            discipline_id = float(row.discipline_id)
            industry_id = float(row.industry_id)
            experience_years_experience = float(row.experience_years_experience)
            experience_n_entries = float(row.experience_n_entries_class)
            edu_degrees = float(row.edu_degree)
            edu_fieldofstudies = str(row.edu_fieldofstudies).split(",")
            job_roles = str(row.jobroles).split(",")
            carrer_level = float(row.career_level)
            if country not in country_mapper.keys():
                c.append(country)
                country_mapper[country] = offsets[4]
                columns_counter[offsets[4]] = 0
                offsets[4] += 1
            col = country_mapper[country]
            columns.append(col)
            columns_counter[col] += 1
            rows.append(row_ind)
            data_temp.append(1)
            ones += 1
            if not m.isnan(discipline_id):
                if discipline_id not in discipline_id_mapper.keys():
                    di.append(discipline_id)
                    discipline_id_mapper[discipline_id] = offsets[2]
                    columns_counter[offsets[2]] = 0
                    offsets[2] += 1
                col = discipline_id_mapper[discipline_id]
                columns.append(col)
                columns_counter[col] += 1
                rows.append(row_ind)
                data_temp.append(1)
                ones += 1
            if not m.isnan(industry_id):
                if industry_id not in industry_id_mapper.keys():
                    ii.append(industry_id)
                    industry_id_mapper[industry_id] = offsets[3]
                    columns_counter[offsets[3]] = 0
                    offsets[3] += 1
                col = industry_id_mapper[industry_id]
                columns.append(col)
                columns_counter[col] += 1
                rows.append(row_ind)
                data_temp.append(1)
                ones += 1
            if not m.isnan(experience_years_experience):
                if experience_years_experience not in experience_years_experience_mapper.keys():
                    eye.append(experience_years_experience)
                    experience_years_experience_mapper[experience_years_experience] = offsets[6]
                    columns_counter[offsets[6]] = 0
                    offsets[6] += 1
                col = experience_years_experience_mapper[experience_years_experience]
                columns.append(col)
                columns_counter[col] += 1
                rows.append(row_ind)
                data_temp.append(1)
                ones += 1
            if not m.isnan(experience_n_entries):
                if experience_n_entries not in experience_n_entries_mapper.keys():
                    ene.append(experience_n_entries)
                    experience_n_entries_mapper[experience_n_entries] = offsets[5]
                    columns_counter[offsets[5]] = 0
                    offsets[5] += 1
                col = experience_n_entries_mapper[experience_n_entries]
                columns.append(col)
                columns_counter[col] += 1
                rows.append(row_ind)
                data_temp.append(1)
                ones += 1
            if not m.isnan(edu_degrees):
                if edu_degrees not in edu_degrees_mapper.keys():
                    ed.append(edu_degrees)
                    edu_degrees_mapper[edu_degrees] = offsets[7]
                    columns_counter[offsets[7]] = 0
                    offsets[7] += 1
                col = edu_degrees_mapper[edu_degrees]
                columns.append(col)
                columns_counter[col] += 1
                rows.append(row_ind)
                data_temp.append(1)
                ones += 1
            if not m.isnan(carrer_level):
                if carrer_level not in career_level_mapper.keys():
                    cl.append(carrer_level)
                    career_level_mapper[carrer_level] = offsets[1]
                    columns_counter[offsets[1]] = 0
                    offsets[1] += 1
                col = career_level_mapper[carrer_level]
                columns.append(col)
                columns_counter[col] += 1
                rows.append(row_ind)
                data_temp.append(1)
                ones += 1
            for jr in job_roles:
                if not m.isnan(float(jr)):
                    if jr not in job_roles_mapper.keys():
                        jro.append(jr)
                        job_roles_mapper[jr] = offsets[0]
                        columns_counter[offsets[0]] = 0
                        offsets[0] += 1
                    col = job_roles_mapper[jr]
                    columns.append(col)
                    columns_counter[col] += 1
                    rows.append(row_ind)
                    data_temp.append(1)
                    ones += 1
            for jr in edu_fieldofstudies:
                if not m.isnan(float(jr)):
                    if jr not in edu_fieldofstudies_mapper.keys():
                        efos.append(jr)
                        edu_fieldofstudies_mapper[jr] = offsets[8]
                        columns_counter[offsets[8]] = 0
                        offsets[8] += 1
                    col = edu_fieldofstudies_mapper[jr]
                    columns.append(col)
                    columns_counter[col] += 1
                    rows.append(row_ind)
                    data_temp.append(1)
                    ones += 1

            # Normalizing the row.
            # data_temp = np.asarray(data_temp) / m.sqrt(ones)
            # Adding it to the matrix
            data.extend(data_temp)

    for i in range(0, len(data)):
        tf = columns_counter[columns[i]] / item_counter
        data[i] /= tf

    user_content_non_normalized = sps.csc_matrix((data, (rows, columns)), shape=(user_counter, np.amax(offsets)))
    user_content_matrix = normalize(user_content_non_normalized, 'l2', 1, copy=True)
    content_user_matrix = user_content_matrix.copy().transpose(copy=False)

    users = list(set(np.append(users, target_users)))
    tot_rows = len(users)
    data_sim, row_indices, col_indices = \
        row_dealer(target_users, user_content_matrix, content_user_matrix, uid_position_dic)

    user_user_similarity_matrix = sps.csc_matrix((data_sim, (row_indices, col_indices)),
                                                 shape=(user_counter, user_counter))
    return user_user_similarity_matrix


def row_dealer(target_users, user_content_matrix, content_user_matrix, uid_position_dic, k=args.k):
    # Instantiate all the variables needed to collect the results
    data = []
    row_ind = []
    col_ind = []
    counter = 0
    # The index will be the new row index in the similarity matrix in which all the
    # rows corresponding to user that are not among the targets are removed
    non_profiled_number = 0

    # For each one of the targets to be analyzed by the worker
    # the dot product is performed with the transposed of the user rating matrix
    for target in target_users:
        row_number = uid_position_dic[target]
        sparse_product_line = \
            user_content_matrix.getrow(row_number).dot(content_user_matrix)
        dense_product_line = np.asarray(sparse_product_line.todense()).ravel()
        dense_product_line[row_number] = 0
        # Only the k best indices are kept
        if len(dense_product_line.copy().nonzero()[0]) == 0:
            non_profiled_number += 1
        k_nearest_indices = np.argpartition(dense_product_line, -k)[-k:]
        for i in range(0, k):
            data.append(dense_product_line[k_nearest_indices[i]])
            col_ind.append(k_nearest_indices[i])
            row_ind.append(row_number)
        if counter % 100 == 0:
            print(counter)
        counter += 1
    print(non_profiled_number)
    return data, row_ind, col_ind

main()
