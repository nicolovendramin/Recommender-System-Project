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

lock = threading.Lock()
last_time = 0
time_offset = 0

parser = argparse.ArgumentParser()
parser.add_argument('--rating_file', type=str, default="data/competition/interactions.csv")
parser.add_argument('--target_users', type=str, default="data/competition/target_users.csv")
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--user_item_sep', type=str, default=',')
parser.add_argument('--item_item_sep', type=str, default='\t')
parser.add_argument('--user_key', type=str, default='user_id')
parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--k', type=int, default=50)
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
parser.add_argument_group('--timestamp_key', type=str, default="created_at")
args = parser.parse_args()

""""
Some useful definitions to understand the code base
Users -> array of all the users: interacting + target
Target_users -> array of the users that are the target of our recommendations
Interacting_users -> array of the users that have at least one interaction

Function Interfaces
def time_print(string1, string2="", string3="", string4="", string5="", string6="", style="Log")
def user_knn(interactions, target_users_file=None, k=50, user_key="user_id", item_key="item_id")
def row_dealer(user_rating_matrix, target_users, uid_position_dic, task_name, offset=0, k=50)
def mapper(array, subject="the given input")
def data_importation(interactions, target_users_file=None, user_key="user_id", item_key="item_id")
def urm_computer(interactions, target_users_file=None, user_key="user_id", item_key="item_id")
def recommend(similarity_matrix, user_rating_matrix, target_users,)
def main(interactions, target_users_file=None, k=50, hold_out_percentage=0.8, prediction_file=None,
             ask_to_go=False, interaction_logic=0, user_key="user_id", item_key="item_id")
"""""


def time_print(string1, string2="", string3="", string4="", string5="", string6="", style="Log"):
    # This function is needed to print also the ex time of the task (approx)
    verbosity_levels = ["None", "Log", "Info", "Tips"]
    lock.acquire()
    global last_time
    millis = int(round(time.time() * 1000))
    diff = millis-last_time
    last_time = millis
    if style == "Log" and \
                verbosity_levels.index(args.verbosity_level) >= verbosity_levels.index("Log"):
        print(Style.BRIGHT + Fore.RED + "[", str((last_time-time_offset)/1000)+"]s ( +[", str(diff)
              + "]ms ) Log @ IB_CF : " + Style.RESET_ALL + Style.BRIGHT + Fore.MAGENTA +
              str(string1)+str(string2)+str(string3)+str(string4)+str(string5)+str(string6)+Style.RESET_ALL)
    elif style == "Info" and \
                  verbosity_levels.index(args.verbosity_level) >= verbosity_levels.index("Info"):
        print(Style.BRIGHT + Fore.BLUE + "[", str((last_time-time_offset)/1000)+"]s ( +[", str(diff)
              + "]ms ) Info @ IB_CF : " + Style.RESET_ALL + Style.BRIGHT + Fore.CYAN +
              str(string1)+str(string2)+str(string3)+str(string4)+str(string5)+str(string6)+Style.RESET_ALL)
    lock.release()


def item_knn(interactions, target_users_file=None, k=50, user_key="user_id", item_key="item_id", rating_key="rating",
             _item_bias_=1, _user_bias_=1):
    # This function is called in order to test or recommend using an user based collaborative filtering approach
    time_print("Building the URM...")

    # User rating matrix : sparse matrix with row_number rows and col_number columns
    # rows are users, columns are items
    # target users are the users for which we want to provide recommendations
    # position uid dic : dictionary associating to a row the corresponding user id
    # uid position : the symmetric
    # position iid dic : dictionary associating to a row the corresponding item id
    # iid position : the symmetric
    # user_rated_items is the dictionary in which for each user are listed all the ids of the items that he rated
    user_rating_matrix, target_users, users, items, position_uid_dic, uid_position_dic, position_iid_dic, \
        iid_position_dic, row_number, col_number, user_rated_items, item_rating_user, pearson_urm, removed_positions\
        = urm_computer(interactions, target_users_file, user_key, item_key, _item_bias_=_item_bias_,
                       rating_key=rating_key, _user_bias_=_user_bias_)
    time_print("URM Successfully built")

    if args.normalize:
        # Normalization of the matrix. column wise
        time_print("Starting the column-wise normalization", style="Info")
        user_rating_matrix = normalize(pearson_urm, 'l2', 0, copy=False)
        time_print("Normalization successfully completed")

    # Now it starts the computation of the similarity matrix
    # In order to improve performances 4 workers will be used to perform the task
    tot = len(items)
    pool = multiprocessing.Pool()
    step = math.floor(tot / args.number_of_cpu)

    # The matrix is supposed to be squared (all users * all users) but in order to
    # improve performances we eliminate all the unused rows, keeping only the rows
    # corresponding to the target users.

    # The task of computing the matrix is split between the specified number of child processes
    blocks = []
    for i in range(0, args.number_of_cpu):
        start = i * step
        end = (i + 1) * step
        if i == args.number_of_cpu-1:
            end = tot
        blocks.append(pool.apply_async(row_dealer, (pearson_urm, items[start:end],
                                                    iid_position_dic, str(i), step * i, k,)))

    results = []
    # Fetching of the results from the workers who run the tasks
    for j in range(0, args.number_of_cpu):
        results.append(blocks[j].get())

    # Merging the results obtained by the workers
    time_print("Merging the results obtained by the workers...", style="Info")

    similar_items_id = dict(results[0][4])
    new_iid_position = dict(results[0][3])
    final_data = results[0][0]
    final_row_ind = results[0][1]
    final_col_ind = results[0][2]
    for i in range(1, args.number_of_cpu):
        final_data = np.hstack((final_data, results[i][0]))
        final_row_ind = np.hstack((final_row_ind, results[i][1]))
        final_col_ind = np.hstack((final_col_ind, results[i][2]))
        similar_items_id.update(results[i][4])
        new_iid_position.update(results[i][3])

    time_print("Merging process completed! The matrix is built for ", str(len(new_iid_position)), " items.")
    time_print("Transforming the matrix into a sparse matrix", style="Info")

    # The similarity matrix is the sparse matrix target_users * users in which are stored the similarities
    similarity_matrix = sps.csc_matrix((final_data, (final_row_ind, final_col_ind)), shape=(tot, col_number))

    pool.close()
    pool.terminate()

    return target_users, similarity_matrix, new_iid_position, user_rating_matrix, position_iid_dic,\
        position_uid_dic, iid_position_dic, uid_position_dic, row_number, col_number, similar_items_id,\
        user_rated_items, item_rating_user, removed_positions


def row_dealer(user_rating_matrix, items, iid_position_dic, task_name, offset=0, k=50):
    # This function retrieves a set of rows with the similarity between a bunch of users and all the others
    nonzero = 0
    data = []
    row_ind = []
    col_ind = []
    counter = 0
    new_iid_position_partial = {}
    transposed_urm = user_rating_matrix.transpose(copy=True)
    tot = len(items)
    similar_items_id = {}
    non_recommendables = check_expiration()
    # For each item computes the similarity with all the other by multiplying it to the transpose of the URM
    for item in items:
        if item not in non_recommendables:
            # This is needed to have a dictionary with the new association row <-> id
            new_iid_position_partial[item] = offset + counter
            counter += 1
            # Check the row of the table associated to the non expired item
            index = iid_position_dic[item]
            row = transposed_urm.getrow(index)
            # Performs the dot product and stores it into an array
            similarity_product_array = np.asarray(transposed_urm.dot(row.T).T.toarray()[0])
            # We set to zero the similarity between the user and himself
            similarity_product_array[index] = 0
            # Counts the nonzero for analytics purposes
            nonzero_turn = len(similarity_product_array.nonzero()[0])
            nonzero += nonzero_turn
            adjusted_k = min(k, nonzero_turn)
            # Prints to show progress every 500 targets users processed
            if counter % 500 == 0:
                time_print("[", task_name, "]:", str(counter/tot*100), " % completed.", style="Info")
            # Here we select the k-nearest-neighbours to store
            k_nearest_indices = np.argpartition(similarity_product_array, -adjusted_k)[-adjusted_k:]

            # We store everything in 3 different arrays that will be used later to initialize a sparse matrix
            similar_items_id[item] = k_nearest_indices.copy()
            for iteration in range(0, adjusted_k):
                row_ind.append(new_iid_position_partial[item])
                col_ind.append(k_nearest_indices[iteration])
                data.append(similarity_product_array[k_nearest_indices[iteration]])

    # We print a message to show the completion of the task
    time_print("[", task_name, "]: 100 % completed.")
    # We add info related to the number of average similar items encountered
    time_print("[", task_name, "]: in average were found ", str(nonzero/tot), "elements per row", style="Info")

    return data, row_ind, col_ind, new_iid_position_partial, similar_items_id


def mapper(array, subject="the given input"):
    # returns a dictionary where each value is associated to position
    to_return = {}
    return_to = {}
    for i in range(len(array)):
        to_return[array[i]] = i
    time_print("Completed position-id association for ", subject)
    for key in to_return.keys():
        return_to[to_return[key]] = key
    time_print("Completed id-position association for ", subject)
    return to_return, return_to


def data_importation(interactions, target_users_file=None, user_key="user_id", item_key="item_id", rating_key="rating"):
    # Import the interaction file in a Pandas data frame
    with open(interactions, 'r') as f:
        interactions_reader = pd.read_csv(interactions, delimiter='\t')

    # If the policy is decremental we need to sort the interactions by timestamp in order to be sure to give
    # more value to the first interactions, and less to the further
    if args.rating_policy == "decremental_a":
        interactions_reader.sort_values(args.timestamp_key, inplace=True)
    if args.rating_policy == "decremental_d":
        interactions_reader.sort_values(args.timestamp_key, inplace=True, ascending=False)

    # Here the program fills the array with all the users to be recommended
    # after this if/else target_users contains this information
    time_print("Listing the target users", style="Info")
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
    time_print("Defining the dimension of the URM and mapping", style="Info")
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
    time_print("Building the temporary dictionary of interactions", style="Info")
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

    return temp_dic, row_number, col_number, target_users, users, items, position_uid_dic, \
        uid_position_dic, position_iid_dic, iid_position_dic


def urm_computer(interactions, target_users_file=None, user_key="user_id", item_key="item_id", rating_key="rating",
                 _user_bias_=1, _item_bias_=1):
    # This functions returns the couple, users to recommend and user rating matrix

    if args.holdout_perc != 0:
        temp_dic, row_number, col_number, target_users, users, items, position_uid_dic, \
            uid_position_dic, position_iid_dic, iid_position_dic, \
            removed_positions = hold_out_split(hold_out_percentage=args.holdout_perc,
                                               interactions_file=interactions, user_key=user_key, item_key=item_key,
                                               rating_key=rating_key, seed=args.seed)
    else:
        (temp_dic, row_number, col_number, target_users, users, items,
         position_uid_dic, uid_position_dic, position_iid_dic, iid_position_dic) = \
            data_importation(interactions, target_users_file, user_key, item_key, rating_key)
        removed_positions = dict()

    # Converting dictionary values into integers and subtracting the user bias
    time_print("Computing the user bias for each user", style="Info")

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
    time_print("Computing the item bias for each item", style="Info")
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
    time_print("Converting values to the right format and subtracting the user bias...", style="Info")
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
    time_print("in average every user evaluated ", str(tot/c)+" ", " items and ", str(zeros) +
               " of the target users evaluated no items.", style="Info")

    # Create a sparse matrix from the interactions
    time_print("Creating the sparse matrix with the data", style="Info")
    user_rating_matrix = sps.csc_matrix((data, (row_ind, col_ind)), shape=(row_number, col_number))
    pearson_urm = sps.csc_matrix((data_pearson, (row_ind, col_ind)), shape=(row_number, col_number))
    data.clear()
    row_ind.clear()
    col_ind.clear()

    if args.recommend_on_pearson == 1:
        user_rating_matrix = pearson_urm

    # returns the non-normalized User Rating Matrix
    return user_rating_matrix, target_users, users, items, position_uid_dic, uid_position_dic, position_iid_dic,\
        iid_position_dic, row_number, col_number, user_rated_items, item_rating_users, pearson_urm, removed_positions


def recommend(similarity_matrix, user_rating_matrix, target_users, new_uid_position,
              position_iid_dic, position_uid_dic, uid_position_dic, user_rated_items, rec_length, expired_items,
              name="Generic Rec-Sys", item_rating_user=None):
    # This function takes a sparse similarity matrix, a list of users and the number of desired recommendations
    # and returns a dictionary u_id -> {(rec_items_id, estimated_rating)}
    counter = 0
    tot = len(target_users)
    non_profiled_users = []
    # variable to count the users for which is not possible to provide user based recommendations
    non_profiled_users_number = 0
    uri_keys = set(user_rated_items.keys())
    transposed_similarity = similarity_matrix.transpose(copy=True)
    # To each key (user_id) is associated a tuple composed by : (item_id,est_rating)
    rec_dictionary = dict()
    # For every user in target_users you have to compute
    for user in target_users:
        # The row of the similarity matrix corresponding to our user
        user_rating_row = user_rating_matrix.getrow(uid_position_dic[user])
        # For every target user we have to compute the estimated rating for all the interesting items
        # where interesting means that have been evaluated at least by one of the neighbours

        # Building the interesting_item_list by merging the respective lines in the user rated items list
        interesting_items = list()
        items_rated_by_the_user = []
        if user in uri_keys:
            items_rated_by_the_user = user_rated_items[user]
        for item in items_rated_by_the_user:
            interesting_items.extend(transposed_similarity.getrow(item).nonzero()[1])

        interesting_items = list(set(interesting_items))
        recommendable_items_number = len(interesting_items)

        # marks as non profiled all the users for which there is no possibility to provide recommendations using
        # the collaborative filtering user based technique
        if recommendable_items_number == 0:
            non_profiled_users_number += 1
            non_profiled_users.append(user)

        # Puts the non-recommendable items in a set in order to get O(1) membership check
        if user not in uri_keys:
            user_rated_items[user] = []

        non_recommendable_set = set(it.chain(expired_items, user_rated_items[user]))

        # Provides recommendations for avery one by initializing the recommendation list with the unseen and not expired
        # top pops to which are assigned negative weights. Then for the user for which is possible the collaborative
        # filtering technique will estimate the personalized recommendations
        if counter % 1 == 0:
            time_print("[", name, "] ", str(counter/tot*100), "% recommendations provided ["+str(counter)
                       + "/" + str(tot) + "]", style="Info")
        counter += 1

        rec_dictionary[user] = non_personalized_recommendation(rec_length, non_recommendable_set)

        # Now for every interested item we compute the estimated rating storing only the @rec_length better ones
        for item_row in interesting_items:
            sparse_sim_row = similarity_matrix.getrow(item_row)
            similarity_matrix_row = sparse_sim_row.toarray()[0]
            similarity_matrix_row_indices = sparse_sim_row.nonzero()[1]
            friends = set(similarity_matrix_row_indices)
            # Checks the estimated rating only for valid item_ids
            if position_iid_dic[item_row] not in non_recommendable_set:
                if args.weights_policy == "mar":
                    weights = args.shrink_factor
                    for ranker in items_rated_by_the_user:
                        if ranker in friends:
                            weights += similarity_matrix_row[ranker]
                elif args.weights_policy == "constant":
                    weights = 1
                elif args.weights_policy == "man":
                    weights = sum(similarity_matrix_row) + args.shrink_factor
                if weights == 0:
                    print("Eccheccazz")
                    weights = 1

                estimated_rating_dirty =\
                    np.asarray((user_rating_row.dot(sparse_sim_row.T).todense())).ravel()
                estimated_rating = estimated_rating_dirty / weights

                new_tuple = position_iid_dic[item_row], estimated_rating[0]
                rec_dictionary[user].append(new_tuple)

        # Stores in the dictionary only the rec_length best tuples, sorted by estimated rating
        rec_dictionary[user].sort(key=lambda tup: tup[1], reverse=True)

        rec_dictionary[user] = rec_dictionary[user][:rec_length]

    time_print("[", name, "] 100% recommendations provided. "+str(non_profiled_users_number)+" of "+str(counter) +
               " are not profiled")

    return rec_dictionary, non_profiled_users


def write_recommendations(recommendations_dic, target_users, user_caption="user_id",
                          rec_item_caption="recommended_items",
                          user_items_sep=',', item_item_sep='\t',
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

    time_print("Output operations concluded")


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


def hold_out_split(hold_out_percentage, interactions_file, user_key="user_id", item_key="item_id", rating_key="rating",
                   seed=1234):

    temp_dic, row_number, col_number, target_users, users, items, position_uid_dic, \
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
    return train_dic, row_number, col_number, test_target_users, users, items, position_uid_dic, uid_position_dic, \
        position_iid_dic, iid_position_dic, removed_ratings_per_user


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


def main(interactions, target_users_file=None, k=50, user_key="user_id", item_key="item_id", rating_key="rating",
         rec_length=5, _item_bias_=1, _user_bias_=1):

    # Setting the timers
    global last_time
    global time_offset
    last_time = int(round(time.time() * 1000))
    time_offset = last_time

    # building the matrices needed in order to recommend the right items
    target_users, similarity_matrix, new_uid_position, user_rating_matrix, position_iid_dic, \
        position_uid_dic, iid_position_dic, uid_position_dic, row_number, col_number, similar_users_columns, \
        user_rated_items, item_rating_user, test_set = item_knn(interactions, target_users_file, k, user_key, item_key,
                                                                rating_key, _item_bias_, _user_bias_)

    expired_items = check_expiration()

    # Splits the work between 4 worker in order to exploit all the 4 CPUs of my machine
    pool = multiprocessing.Pool()
    tot = len(target_users)
    step = math.floor(tot / args.number_of_cpu)

    blocks = []
    for i in range(0, args.number_of_cpu):
        start = i * step
        end = (i + 1) * step
        if i == args.number_of_cpu-1:
            end = tot
        blocks.append(pool.apply_async(recommend, (similarity_matrix, user_rating_matrix,
                                                   target_users[start:end], new_uid_position,
                                                   position_iid_dic, position_uid_dic, uid_position_dic,
                                                   user_rated_items, rec_length, expired_items, str(i),
                                                   item_rating_user,)))

    results = []
    # Fetching of the results from the workers who run the tasks
    for j in range(0, args.number_of_cpu):
        results.append(blocks[j].get())

    # Merging the results obtained by the workers
    time_print("Merging the results obtained by the workers...", style="Info")

    rec_dictionary = dict(results[0][0])
    non_profiled_users = results[0][1]
    for i in range(1, args.number_of_cpu):
        non_profiled_users = np.hstack((non_profiled_users, results[i][1]))
        rec_dictionary.update(results[i][0])

    # If it is not testing then it writes the recommendations
    if args.holdout_perc == 0:
        # Writing the recommendations file
        time_print("Writing the recommendations files")
        write_recommendations(rec_dictionary, target_users,
                              prediction_file=args.prediction_file, user_items_sep=args.user_item_sep,
                              item_item_sep=args.item_item_sep)
    # If it's testing then it computes the map and prints it
    else:
        # Computing the map
        time_print("Estimating recommendation quality")
        unit = []
        for i in range(0, args.number_of_cpu):
            start = i * step
            end = (i + 1) * step
            if i == args.number_of_cpu-1:
                end = tot
            unit.append(pool.apply_async(estimate_quality,
                        (rec_dictionary, target_users[start:end], rec_length, test_set,)))
        map_summation = 0
        for j in range(0, args.number_of_cpu):
            map_summation += unit[j].get()
        real_map = map_summation / args.number_of_cpu
        time_print("The map estimated was-->MAP@" + str(rec_length) + ": " + str(real_map), style="Info")

    pool.close()
    pool.terminate()
    time_print("All work done!")

    print(len(non_profiled_users))

if args.help_me != 19835:
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
          '\n   --verbosity_level : is in {None, Log, Info,} and indicates the kind of messages that you want to '
          'be printed  ' +
          '\n   --number_of_cpu : parameter to decide on how many processor to split the computational load  ' +
          '\n   --holdout_perc : the percentage of interactions to include in the train set  ' +
          '\n   --seed : the seed of the random function used to shuffle the interactions to randomly select '
          'the train set  ' +
          '\n   --weights_policy : is in {man, mar, constant} man uses all the similarities, mar only the ones'
          ' of the rankers '
          'and constant doesn\'t divide  ' +
          '\n   --recommend_on_pearson : 1 if you want to use the debiased matrix to recommend, '
          '0 to use the biased one  ' +
          '\n   --rating_policy : is in {sum, max, boolean, decremental_a, decremental_d}'
          ' sum sums interactions, max picks the maximum '
          ',boolean consider 1 if there is at least one otherwise 0 and decremental_a (decremental_d)'
          ' shrinks new (past) interactions by the '
          'value of the past (new) ones'
          '\n   --shrink_factor : is the shrink factor to be considered'
          '\n   --log_scale : is the base of the logarithm used to scale data [if used use also recommend_on_pearson]'
          ' and remember that the higher is the base the major is the shrinkage [eg. 2 maps 1:100 in 0:6, 10 maps it in'
          '0:2')
else:
    o_file = open(args.prediction_file.replace(".csv", "[INFO].txt"), "w+")
    if args.holdout_perc != 0:
        time_print("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %")
    o_file.write("TEST SESSION. Hold out percentage : " + str(args.holdout_perc) + " %\n")
    time_print("k : " + str(args.k), style="Info")
    o_file.write("k : " + str(args.k) + "\n")
    time_print("recommendations per user : " + str(args.rec_length), style="Info")
    o_file.write("recommendations per user : " + str(args.rec_length) + "\n")
    time_print("Weights in the estimated ratings follow the policy : " + args.weights_policy, style="Info")
    o_file.write("Weights in the estimated ratings follow the policy : " + args.weights_policy + "\n")
    time_print("Shrink factor : " + str(args.shrink_factor), style="Info")
    o_file.write("Shrink factor : " + str(args.shrink_factor))
    time_print("Ratings are considered with the following policy : " + args.rating_policy, style="Info")
    o_file.write("Ratings are considered with the following policy : " + args.rating_policy + "\n")
    if args.log_scale > 0:
        time_print("Logaritmic rating' shrinkage factor : " + args.rating_policy, style="Info")
        o_file.write("Logaritmic rating' shringage factor : " + args.rating_policy + "\n")
    if args.normalize == 1:
        time_print("Normalization active", style="Info")
    o_file.write("Normalization active" + "\n")
    if args.user_bias == 1:
        time_print("User bias subtraction active", style="Info")
    o_file.write("User bias subtraction active" + "\n")
    if args.item_bias == 1:
        time_print("Item bias subtraction active", style="Info")
    o_file.write("Item bias subtraction active" + "\n")
    if args.recommend_on_pearson == 0:
        time_print("Estimated ratings are given using the biased urm", style="Info")
    o_file.write("Estimated ratings are given using the biased urm" + "\n")
    if args.number_of_cpu > 1:
        time_print("Parallelization is performed using " + str(args.number_of_cpu) + " processors", style="Info")
    o_file.write("Parallelization is performed using " + str(args.number_of_cpu) + " processors" + "\n")
    o_file.close()
    main(args.rating_file, args.target_users, k=args.k,
         _item_bias_=args.item_bias, _user_bias_=args.user_bias, rating_key=args.rating_key, rec_length=args.rec_length,
         user_key=args.user_key, item_key=args.item_key)
