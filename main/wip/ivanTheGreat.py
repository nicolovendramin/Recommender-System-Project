import datetime
import multiprocessing
import threading
import time
import math as math
import numpy as np
import pandas as pd
import scipy.sparse as sps
from colorama import Fore, Style
from sklearn.preprocessing import normalize
import itertools as it
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rating_file', type=str, default="data/competition/interactions.csv")
parser.add_argument('--target_users', type=str, default="data/competition/target_users.csv")
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--user_item_sep', type=str, default=',')
parser.add_argument('--item_item_sep', type=str, default=' ')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--k', type=int, default=50)
parser.add_argument('--to_keep', type=int, default=50)
parser.add_argument('--shrink_factor', type=int, default=0)
parser.add_argument('--normalize', type=int, default=1)
parser.add_argument('--prediction_file', type=str, default="target/competition/"
                                                           + str(datetime.datetime.now()).replace(" ", "").
                                                           replace(".", "").replace(":", "") + "submission.csv")
parser.add_argument('--user_bias', type=int, default=1)
parser.add_argument('--item_bias', type=int, default=1)
parser.add_argument('--rec_length', type=int, default=5)
parser.add_argument('--verbosity_level', type=str, default="Info")
parser.add_argument('--number_of_cpu', type=int, default=4)
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


def hold_out_split(hold_out_percentage, interactions_file, user_key="user_id", item_key="item_id", rating_key="rating",
                   seed=1234):

    temp_dic, row_number, col_number, target_users, users, position_uid_dic, \
        uid_position_dic, position_iid_dic, iid_position_dic = data_importation(interactions_file, user_key=user_key,
                                                                                item_key=item_key,
                                                                                rating_key=rating_key,
                                                                                target_users_file=args.target_users)

    print(len(target_users))
    # set the random seed
    rng = np.random.RandomState(seed)
    #  shuffle data
    keys = list(temp_dic.keys())
    number_of_ratings = len(keys)
    acc_order = keys.copy()
    shuffled_access_order = rng.permutation(acc_order)
    train_size = int(number_of_ratings * hold_out_percentage/100)
    non_recommendable = set(check_expiration())
    train_dic = dict()
    removed_ratings_per_user = {}
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

    test_target_users = list(set(test_target_users))
    print(len(test_target_users))
    test_target_users = test_target_users[:len(target_users)]
    return train_dic, row_number, col_number, test_target_users, users, position_uid_dic, uid_position_dic, \
        position_iid_dic, iid_position_dic, removed_ratings_per_user


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
    weight = -2
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


def estimate_quality(rec_dictionary, target_users, rec_length, test_set):
    dirty_map = 0.0
    number = 0
    for user in target_users:
        dirty_map += map_computer(rec_dictionary[user], pos_items=test_set[user], at=rec_length)
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


def mapper(array, subject="the given input"):
    # returns a dictionary where each value is associated to position
    to_return = {}
    return_to = {}
    for i in range(len(array)):
        to_return[array[i]] = i
    print("Completed position-id association for ", subject)
    for key in to_return.keys():
        return_to[to_return[key]] = key
    print("Completed id-position association for ", subject)
    return to_return, return_to


def data_importation(interactions, target_users_file=None, user_key="user_id", item_key="item_id", rating_key="rating"):
    # Import the interaction file in a Pandas data frame
    with open(interactions, 'r') as f:
        interactions_reader = pd.read_csv(interactions, delimiter='\t')

    # If the policy is decremental we need to sort the interactions by timestamp in order to be sure to give
    # more value to the first interactions, and less to the further
    # NOTE here is given more importance to the oldest interactions... to change it set ascending to False in the sort
    # If the policy is decremental we need to sort the interactions by timestamp in order to be sure to give
    # more value to the first interactions, and less to the further
    if args.rating_policy == "decremental_a":
        interactions_reader.sort_values(args.timestamp_key, inplace=True)
    if args.rating_policy == "decremental_d":
        interactions_reader.sort_values(args.timestamp_key, inplace=True, ascending=False)

    # Here the program fills the array with all the users to be recommended
    # after this if/else target_users contains this information
    print("Listing the target users")
    interacting_users = interactions_reader[user_key].as_matrix()
    if target_users_file is None:
        target_users = interacting_users
        users = interacting_users
    else:
        with open(target_users_file, 'r') as t:
            target_users_reader = pd.read_csv(target_users_file, delimiter='\t')
        target_users = target_users_reader[user_key].as_matrix()
        users = list(set(np.hstack((target_users, interacting_users))))
        t.close()

    # Computes the size of the user rating matrix using a simple parallel splitting to increase the speed
    print("Defining the dimension of the URM and mapping")
    items = list(set(interactions_reader[item_key].as_matrix()))
    pool = multiprocessing.Pool()
    position_iid_dic_process = pool.apply_async(mapper, (items, "Items",))
    iid_position_dic, position_iid_dic = position_iid_dic_process.get()
    position_uid_dic_process = pool.apply_async(mapper, (users, "Users",))
    uid_position_dic, position_uid_dic = position_uid_dic_process.get()

    pool.close()
    pool.terminate()

    row_number = len(users)
    col_number = len(items)
    temp_dic = {}

    # Building a dictionary indexed by tuples (u_id,i_id) each one associated to the number of interactions
    # of the user u_id with the item i_id. The rating assigned to each user to an item is equal to the number
    # of interactions that he had with the item
    print("Building the temporary dictionary of interactions")
    for i, row in interactions_reader.iterrows():
        key = (uid_position_dic[row[user_key]], iid_position_dic[row[item_key]])
        if key not in temp_dic.keys():
            temp_dic[key] = 0
        if args.rating_policy == "sum":
            temp_dic[key] += row[rating_key]
        elif args.rating_policy == "max":
            temp_dic[key] = max(row[rating_key], temp_dic[key])
        elif args.rating_policy == "boolean":
            temp_dic[key] = 1
        elif args.rating_policy == "decremental_a" or args.rating_policy == "decremental_d":
            if temp_dic[key] == 0:
                temp_dic[key] = row[rating_key]
            else:
                temp_dic[key] += row[rating_key] * (1 / temp_dic[key])

    f.close()

    return temp_dic, row_number, col_number, target_users, users, position_uid_dic, \
        uid_position_dic, position_iid_dic, iid_position_dic


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
    dictionary_file.write("{")
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
        dic_line.replace("),]", ")],")

        # At the end we write the line corresponding to that target user on the file
        output_file.write(line+"\n")
        dictionary_file.write(dic_line+"\n")

    # Last token at the end of the dictionary file
    dictionary_file.write("}")
    # Closing the resources
    output_file.close()
    dictionary_file.close()

    print("Output operations concluded")


def urm_computer(interactions, target_users_file=None, user_key="user_id", item_key="item_id", rating_key="rating",
                 _user_bias_=1, _item_bias_=1):
    # This functions returns the couple, users to recommend and user rating matrix

    if args.holdout_perc != 0:
        temp_dic, row_number, col_number, target_users, users, position_uid_dic, \
            uid_position_dic, position_iid_dic, iid_position_dic, \
            removed_positions = hold_out_split(hold_out_percentage=args.holdout_perc,
                                               interactions_file=interactions, user_key=user_key, item_key=item_key,
                                               rating_key=rating_key, seed=args.seed)
    else:
        (temp_dic, row_number, col_number, target_users, users,
         position_uid_dic, uid_position_dic, position_iid_dic, iid_position_dic) = \
            data_importation(interactions, target_users_file, user_key, item_key, rating_key)
        removed_positions = dict()

    # Converting dictionary values into integers and subtracting the user bias
    print("Computing the user bias for each user")

    # For each line summing all the votes, and counting the number
    totals = {}
    row_elements = {}

    for key in temp_dic.keys():
        value = int(temp_dic[key])
        if key[0] in totals.keys():
            totals[key[0]] += value
            row_elements[key[0]] += 1
        else:
            totals[key[0]] = value
            row_elements[key[0]] = 1

    # Computing the bias of each user
    user_average = {}
    for key in temp_dic.keys():
        user_average[key[0]] = totals[key[0]]/row_elements[key[0]]

    # For each column summing all the votes and counting the number
    print("Computing the item bias for each item")
    totals.clear()
    col_elements = {}

    for key in temp_dic.keys():
        value = int(temp_dic[key])
        if _user_bias_ == 1:
            value -= user_average[key[0]]
        if key[1] in totals.keys():
            totals[key[1]] += value
            col_elements[key[1]] += 1
        else:
            totals[key[1]] = value
            col_elements[key[1]] = 1

    # Computing the bias of each item
    item_average = {}
    for key in temp_dic.keys():
        item_average[key[1]] = totals[key[1]] / col_elements[key[1]]

    # Conversion and subtraction of the bias if requested
    data = []
    data_pearson = []
    row_ind = []
    col_ind = []

    # user rated items is a dictionary in which to every user_id is associated the list of the items (indicated by
    # column number) that he rated
    user_rated_items = {}
    item_rating_users = {}
    print("Converting values to the right format and subtracting the user bias...")
    for key in temp_dic.keys():
        value = int(temp_dic[key])
        data.append(value)
        if _user_bias_ == 1:
            value = value - user_average[key[0]]
        if _item_bias_ == 1:
            value = value - item_average[key[1]]
        if args.log_scale > 0:
            value = math.log(value, args.log_scale)
        if value == 0:
            value = 0.1
        data_pearson.append(value)
        row_ind.append(int(key[0]))
        col_ind.append(int(key[1]))
        key_user = position_uid_dic[key[0]]
        if key_user not in user_rated_items.keys():
            user_rated_items[key_user] = []
        if key[1] not in item_rating_users.keys():
            item_rating_users[key[1]] = []
        item_rating_users[key[1]].append((int(key[0])))
        user_rated_items[key_user].append((int(key[1])))

    tot = 0
    c = 0
    zeros = 0
    interactive_targets = 0
    for key in user_rated_items.keys():
        tot += len(user_rated_items[key])
        c += 1
        if key in target_users:
            interactive_targets += 1
        if len(user_rated_items[key]) == 0 and key in target_users:
            zeros += 1

    zeros += len(target_users)-interactive_targets
    print("in average every user evaluated ", str(tot/c)+" ", " items and ", str(zeros) +
          " of the target users evaluated no items.")

    # Create a sparse matrix from the interactions
    print("Creating the sparse matrix with the data")
    user_rating_matrix = sps.csc_matrix((data, (row_ind, col_ind)), shape=(row_number, col_number))
    pearson_urm = sps.csc_matrix((data_pearson, (row_ind, col_ind)), shape=(row_number, col_number))
    data.clear()
    row_ind.clear()
    col_ind.clear()

    if args.recommend_on_pearson == 1:
        user_rating_matrix = pearson_urm

    # returns the non-normalized User Rating Matrix
    return user_rating_matrix, target_users, users, position_uid_dic, uid_position_dic, position_iid_dic,\
        iid_position_dic, row_number, col_number, user_rated_items, item_rating_users, pearson_urm, removed_positions


def main():
    user_rating_matrix, target_users, users, position_uid_dic, uid_position_dic, position_iid_dic, \
        iid_position_dic, row_number, col_number, user_rated_items, item_rating_users, pearson_urm, removed_positions = \
        urm_computer(args.rating_file, args.target_users, user_key="user_id", item_key="item_id", rating_key="rating",
                     _user_bias_=0, _item_bias_=0)
    prediction_file = args.prediction_file
    rec_dictionary = {}
    expired_items = check_expiration()
    counter = 0
    summer = []
    non_rec_users = 0
    uri_keys = set(user_rated_items.keys())
    if args.normalize:
        # Normalization of the matrix. Row wise
        print("Starting the row-wise normalization")
        user_rating_matrix = normalize(pearson_urm, 'l2', 1, copy=False)
        print("Normalization successfully completed")
    transposed_urm = user_rating_matrix.transpose(copy=True)
    print(user_rating_matrix.shape)

    similarity_matrix = user_rating_matrix.dot(transposed_urm)
    """"
    pool = multiprocessing.Pool()
    tot = len(target_users)
    print(len(target_users))
    step = math.floor(tot / args.number_of_cpu)

    blocks = []
    for i in range(0, args.number_of_cpu):
        start = i * step
        end = (i + 1) * step
        if i == args.number_of_cpu-1:
            end = tot
        blocks.append(pool.apply_async(recommend, (similarity_matrix, user_rating_matrix, target_users[start:end],
                                                   position_iid_dic, uid_position_dic, user_rated_items,
                                                   expired_items, str(i), uri_keys,)))

    results = []
    # Fetching of the results from the workers who run the tasks
    for j in range(0, args.number_of_cpu):
        results.append(blocks[j].get())

    maxind = 0
    extra = 0
    for k in results:
        non_rec_users += k[1]
        extra += k[2]
        rec_dictionary.update(k[0])
        maxind = max(maxind, k[3])
"""""
    iter_counter = 0
    counter = 0
    extra = 0
    summer = []
    maxind = 0
    name = "ciccio"
    rec_dictionary = dict()
    for target in target_users:
        iter_counter += 1
        print(name + " : " + str(iter_counter))
        counter2 = 0
        if target not in uri_keys:
            user_rated_items[target] = []
            counter += 1
        non_recommendable_set = set(it.chain(expired_items, user_rated_items[target]))
        row_ind = uid_position_dic[target]
        row = similarity_matrix.getrow(row_ind).copy()
        size = row.shape[1]
        k_best_indices = np.argpartition(row.todense().tolist()[0], -args.k)[-args.k:]
        rows = k_best_indices.copy()
        data = []
        for i in k_best_indices:
            data.append(1)
        sparse_filter = sps.csc_matrix((data, (rows, k_best_indices)), shape=(size, size))
        filtered_row = row.dot(sparse_filter)
        line = filtered_row.dot(user_rating_matrix)
        line = line.todense().tolist()[0]
        k_best_matches = np.argpartition(line, -args.to_keep)[-args.to_keep:]
        possible_matches = non_personalized_recommendation(args.rec_length, non_recommendable_set)
        for i in k_best_matches:
            if position_iid_dic[i] not in non_recommendable_set:
                possible_matches.append((position_iid_dic[i], line[i]))
                counter2 += 1
                if i > maxind:
                    maxind = i
        if counter2 != 0:
            summer.append(counter2)
        if counter2 < 5:
            extra += 1
        possible_matches.sort(key=lambda tup: tup[1], reverse=True)
        rec_dictionary[target] = possible_matches[:args.rec_length]

    print(non_rec_users)
    print(extra)
    print(maxind)

    if args.holdout_perc != 0:
        real_map = estimate_quality(rec_dictionary, target_users,
                                    rec_length=args.rec_length, test_set=removed_positions)
        print("The estimated map should be: "+str(real_map))
        prediction_file = prediction_file.replace(".csv", "[TEST].csv")
        print(prediction_file)

    write_recommendations(rec_dictionary, target_users, prediction_file=prediction_file)


def recommend(similarity_matrix, user_rating_matrix, target_users, position_iid_dic, uid_position_dic, user_rated_items,
              expired_items, name, uri_keys):
    iter_counter = 0
    counter = 0
    extra = 0
    summer = []
    maxind = 0
    rec_dictionary = dict()
    for target in target_users:
        iter_counter += 1
        print(name + " : " + str(iter_counter))
        counter2 = 0
        if target not in uri_keys:
            user_rated_items[target] = []
            counter += 1
        non_recommendable_set = set(it.chain(expired_items, user_rated_items[target]))
        row_ind = uid_position_dic[target]
        row = similarity_matrix.getrow(row_ind).copy()
        size = row.shape[1]
        k_best_indices = np.argpartition(row.todense().tolist()[0], -args.k)[-args.k:]
        rows = k_best_indices.copy()
        data = []
        for i in k_best_indices:
            data.append(1)
        sparse_filter = sps.csc_matrix((data, (rows, k_best_indices)), shape=(size, size))
        filtered_row = row.dot(sparse_filter)
        line = filtered_row.dot(user_rating_matrix)
        line = line.todense().tolist()[0]
        k_best_matches = np.argpartition(line, -args.to_keep)[-args.to_keep:]
        possible_matches = non_personalized_recommendation(args.rec_length, non_recommendable_set)
        for i in k_best_matches:
            if position_iid_dic[i] not in non_recommendable_set:
                possible_matches.append((position_iid_dic[i], line[i]))
                counter2 += 1
                if i > maxind:
                    maxind = i
        if counter2 != 0:
            summer.append(counter2)
        if counter2 < 5:
            extra += 1
        possible_matches.sort(key=lambda tup: tup[1], reverse=True)
        rec_dictionary[target] = possible_matches[:args.rec_length]

    return rec_dictionary, counter, extra, maxind


main()














