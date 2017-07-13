import scipy.sparse as sps
import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import normalize
import itertools as it
import math as m
import argparse
import datetime

print("Applying IALS")
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
args = parser.parse_args("")


def rec(argos):
    global args
    args = argos
    st_time = time.time()
    time_span = st_time
    print("Checking the expired items")
    expired_items = check_expiration()
    print(int(time.time() - st_time), "-", int(time.time() - time_span))
    time_span = time.time()
    print("Building the interactions' matrix")
    with open(args.rating_file, 'r') as f:
        interactions_reader = pd.read_csv(args.rating_file, delimiter='\t')

    with open(args.target_users, 'r') as j:
        target_reader = pd.read_csv(args.target_users, delimiter='\t')

    users = interactions_reader[args.user_key].as_matrix()
    target_users = target_reader[args.user_key].as_matrix()
    items = list(set(interactions_reader[args.item_key].as_matrix()))
    expired = set(check_expiration())
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
        if item not in expired:
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
    # TODO check not bad influence on scores
    for usr in target_users:
        if usr not in users:
            uid_position_dic[usr] = user_counter
            position_uid_dic[user_counter] = usr
            user_counter += 1

    f.close()
    j.close()
    print(int(time.time() - st_time), "-", int(time.time() - time_span))
    time_span = time.time()
    print("Preprocessing the ratings")
    temp_dic = rating_processing(d, r, c)
    print(int(time.time() - st_time), "-", int(time.time() - time_span))
    time_span = time.time()
    removed_ratings_per_user = {}
    if args.holdout_perc != 0:
        print("Performing the holdout split")
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
        print(int(time.time() - st_time), "-", int(time.time() - time_span))

    data, rows, columns = sparse_arrays_extractor(temp_dic)

    user_rating_matrix_non_normalized = sps.csc_matrix((data, (rows, columns)), shape=(user_counter, item_counter))
    user_rating_matrix = normalize(user_rating_matrix_non_normalized, 'l2', 1, copy=True)
    rating_user_matrix = user_rating_matrix.copy().transpose(copy=False)
    csr_user_rating_matrix = user_rating_matrix.tocsr().astype(np.float32)
    print("Training the model")
    time_span = time.time()
    U, V = fit(csr_user_rating_matrix, 20, 43, 0.0, 0.1, 300, 5, 10)
    print("Model trained in ", int(time.time() - time_span), " seconds")
    print("Providing recommendations")
    time_span = time.time()
    rec_dictionary = {}
    counter = 0
    total_number = len(target_users)
    non_rec_counter = 0
    for target_user in target_users:
        counter += 1
        if target_user in user_rated_items.keys():
            position = uid_position_dic[target_user]
            estimated_ratings = np.dot(U[position], V.T)
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
        else:
            # If the user is non profiled then just recommend the top pops.
            non_rec_counter += 1
            rec_dictionary[target_user] = non_personalized_recommendation(args.rec_length, expired_items)
        if counter % 500 == 0:
            print("Recommended the ",str(int(counter * 100 / total_number)) + "% of the user "
                                                                              "( including a " +
                  str(int(non_rec_counter * 100 / counter)) + "% of non personalized recommendations)"),

    print(int(time.time() - st_time), "-", int(time.time() - time_span))
    print("Recommendation process is over")
    map_at = 0
    time_span = time.time()
    if args.holdout_perc != 0:
        print("Estimating the map")
        map_at = \
            estimate_quality(rec_dictionary, target_users, rec_length=args.rec_length,
                             test_set=removed_ratings_per_user)
        print("The calculated map was : ", map_at)
        print(int(time.time() - st_time), "-", int(time.time() - time_span))
        time_span = time.time()

    write_recommendations(rec_dictionary, target_users, user_caption="user_id",
                          rec_item_caption="recommended_items", item_item_sep=args.item_item_sep,
                          prediction_file=args.prediction_file, user_items_sep=args.user_item_sep)

    print(int(time.time() - st_time), "-", int(time.time() - time_span))
    return rec_dictionary, target_users, removed_ratings_per_user, map_at, "IALS"


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

    for i in range(0, rec_length - counter):
        to_recommend.append((0, weight))
        weight -= 1

    return to_recommend


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
                    + max_user_average * args.user_bias + max_item_average * args.item_bias + 0.5
                    - user_totals[key[0]] * args.user_bias - item_totals[key[1]] * args.item_bias)
        rows.append(key[0])
        columns.append(key[1])
    return data, rows, columns


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


def print_run_details():
    o_file = open(args.prediction_file.replace(".csv", "[INFO].txt"), "w+")
    if args.holdout_perc != 0:
        print("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %")
    o_file.write("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %\n")
    print("k : " + str(args.k))
    o_file.write("k : " + str(args.k) + "\n")
    print("Technique : User Based Collaborative Filtering")
    o_file.write("Technique : User Based Collaborative Filtering\n")
    print("recommendations per user : " + str(args.rec_length))
    o_file.write("recommendations per user : " + str(args.rec_length) + "\n")
    print("Weights in the estimated ratings follow the policy : " + args.weights_policy)
    o_file.write("Weights in the estimated ratings follow the policy : " + args.weights_policy + "\n")
    print("Shrink factor : " + str(args.shrink_factor))
    o_file.write("Shrink factor : " + str(args.shrink_factor) + "\n")
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
    output_file.write(user_caption + user_items_sep + rec_item_caption + "\n")
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
            dic_line += "(" + str(recommendation[0]) + "," + str(recommendation[1]) + "),"

        dic_line += "]"
        # Removes the comma after the last tuple
        dic_line = dic_line.replace("),]", ")],")

        # At the end we write the line corresponding to that target user on the file
        output_file.write(line + "\n")
        dictionary_file.write(dic_line + "\n")

    # Last token at the end of the dictionary file
    dictionary_file.write("}\n\ndef getDictionary():\n\treturn dic")
    # Closing the resources
    output_file.close()
    dictionary_file.close()

    print("Output operations concluded")


def fit(R, alpha, rnd_seed, init_mean, init_std, num_factors, iters, reg):
    # compute the confidence matrix
    C = R.copy().tocsr()
    # use linear scaling here
    # TODO: add log-scaling
    C.data = 1 + alpha * C.data
    Ct = C.T.tocsr()
    M, N = R.shape

    # set the seed
    np.random.seed(rnd_seed)

    # initialize the latent factors
    X = np.random.normal(init_mean, init_std, size=(M, num_factors))
    Y = np.random.normal(init_mean, init_std, size=(N, num_factors))

    for it in range(iters):
        print("Iteration number: ", it, ".")
        X = _lsq_solver_fast(C, X, Y, reg)
        Y = _lsq_solver_fast(Ct, Y, X, reg)

    return X, Y


def _lsq_solver(C, X, Y, reg):
    # precompute YtY
    rows, factors = X.shape
    YtY = np.dot(Y.T, Y)

    for i in range(rows):
        # accumulate YtCiY + reg*I in A
        A = YtY + reg * np.eye(factors)

        # accumulate Yt*Ci*p(i) in b
        b = np.zeros(factors)

        for j, cij in _nonzeros(C, i):
            vj = Y[j]
            A += (cij - 1.0) * np.outer(vj, vj)
            b += cij * vj

        X[i] = np.linalg.solve(A, b)
    return X


def _lsq_solver_fast(C, X, Y, reg):
    # precompute YtY
    rows, factors = X.shape
    YtY = np.dot(Y.T, Y)

    for i in range(rows):
        # accumulate YtCiY + reg*I in A
        A = YtY + reg * np.eye(factors)

        start, end = C.indptr[i], C.indptr[i + 1]
        j = C.indices[start:end]  # indices of the non-zeros in Ci
        ci = C.data[start:end]  # non-zeros in Ci

        Yj = Y[j]  # only the factors with non-zero confidence
        # compute Yt(Ci-I)Y
        aux = np.dot(Yj.T, np.diag(ci - 1.0))
        A += np.dot(aux, Yj)
        # compute YtCi
        b = np.dot(Yj.T, ci)

        X[i] = np.linalg.solve(A, b)
    return X


def _nonzeros(R, row):
    for i in range(R.indptr[row], R.indptr[row + 1]):
        yield (R.indices[i], R.data[i])
