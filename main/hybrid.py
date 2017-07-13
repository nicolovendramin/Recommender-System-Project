import hybrid_imports.UserBasedCF as UserBasedCF
import hybrid_imports.UserContentBased as UserContentBased
import hybrid_imports.ItemContentBased as ItemContentBased
import hybrid_imports.ials2 as ials
import hybrid_imports.ItemBasedCF as ItemBasedCF
import argparse
import datetime
import numpy as np

par = argparse.ArgumentParser()
par.add_argument('--item_cf', type=str, default="None")
par.add_argument('--icf_weight', type=float, nargs='+', default=[1])
par.add_argument('--ials_weight', type=float, nargs='+', default=[1])
par.add_argument('--rating_file', type=str, default="data/competition/interactions.csv")
par.add_argument('--target_users', type=str, default="data/competition/target_users.csv")
par.add_argument('--icb_weight', type=float, nargs='+', default=[0.6])
par.add_argument('--ucf_weight', type=float, nargs='+', default=[1.2])
par.add_argument('--ucb_weight', type=float, nargs='+', default=[0.2])
par.add_argument('--user_cf', type=str, default="None")
par.add_argument('--item_cb', type=str, default="None")
par.add_argument('--user_cb', type=str, default="None")
par.add_argument('--ials', type=str, default="None")
par.add_argument('--holdout_perc', type=int, default=0)
par.add_argument('--rec_length', type=int, default=5)
par.add_argument('--seed', type=int, default=1234)
parsed = par.parse_args()


def parse_recommender_arguments(strings):
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
    args = parser.parse_args(strings)
    return args


def main():
    to_append = ""
    if parsed.holdout_perc != 0:
        to_append = " --holdout_perc "+str(parsed.holdout_perc)+" --seed "+str(parsed.seed)
    if parsed.rating_file != "data/competition/interactions.csv":
        to_append = to_append + " --rating_file "+str(parsed.rating_file)
    if parsed.target_users != "data/competition/target_users.csv":
        to_append = to_append + " --target_users "+str(parsed.target_users)
    dictionaries = []
    weights = []
    maps = []
    decays = []
    techs = []
    targets = []
    test = []
    i = 0
    max_weight = max(len(list(parsed.ials_weight)), len(list(parsed.icb_weight)), len(list(parsed.ucf_weight)),
                     len(list(parsed.ucb_weight)), len(list(parsed.icf_weight)))
    if parsed.ials != "None":
        argos = parse_recommender_arguments((parsed.ials+to_append).split(" "))
        dic, targets, test, map_, tech = ials.rec(argos)
        dictionaries.append(dic)
        maps.append(map_)
        techs.append(tech)
        decays.append(constant_decay)
        w_num = len(parsed.ials_weight)
        for w in range(0, max_weight):
            weights.append([])
            weights[w].append(parsed.ials_weight[min(w, w_num-1)])
        i += 1
    if parsed.item_cf != "None":
        argos = parse_recommender_arguments((parsed.item_cf+to_append).split(" "))
        dic, targets, test, map_, tech = ItemBasedCF.rec(argos)
        dictionaries.append(dic)
        maps.append(map_)
        techs.append(tech)
        decays.append(constant_decay)
        w_num = len(parsed.icf_weight)
        for w in range(0, max_weight):
            weights[w].append(parsed.icf_weight[min(w, w_num-1)])
        i += 1
    if parsed.item_cb != "None":
        argos = parse_recommender_arguments((parsed.item_cb+to_append).split(" "))
        dic, targets, test, map_, tech = ItemContentBased.rec(argos)
        dictionaries.append(dic)
        maps.append(map_)
        techs.append(tech)
        decays.append(constant_decay)
        w_num = len(parsed.icb_weight)
        for w in range(0, max_weight):
            weights[w].append(parsed.icb_weight[min(w, w_num-1)])
        i += 1
    if parsed.user_cf != "None":
        argos = parse_recommender_arguments((parsed.user_cf+to_append).split(" "))
        dic, targets, test, map_, tech = UserBasedCF.rec(argos)
        dictionaries.append(dic)
        maps.append(map_)
        techs.append(tech)
        decays.append(constant_decay)
        w_num = len(parsed.ucf_weight)
        for w in range(0, max_weight):
            weights[w].append(parsed.ucf_weight[min(w, w_num-1)])
        i += 1
    if parsed.user_cb != "None":
        argos = parse_recommender_arguments((parsed.user_cb+to_append).split(" "))
        dic, targets, test, map_, tech = UserContentBased.rec(argos)
        dictionaries.append(dic)
        maps.append(map_)
        techs.append(tech)
        decays.append(gradual_decay)
        w_num = len(parsed.ucb_weight)
        for w in range(0, max_weight):
            weights[w].append(parsed.ucb_weight[min(w, w_num-1)])
        i += 1

    print(weights)
    for a in range(0, max_weight):
        print("Combination number: ", a+1)
        print(techs)
        print(weights[a])
        rec_dictionary = combine(dictionaries, weights[a], i, decays)
        if parsed.holdout_perc != 0:
            mappo = estimate_quality(rec_dictionary, targets, parsed.rec_length, test)
            for o in range(len(techs)):
                print("Estimated map for ", techs[o], " : ", maps[o])
            print("The estimated map for the hybrid was: ", mappo)
            if mappo > max(maps):
                print("The hybrid is better than all the sub-components")
            elif mappo < min(maps):
                print("The hybrid is worse than all the sub-components")
            else:
                w, b = [], []
                for o in range(len(maps)):
                    if maps[o] > mappo:
                        b.append(techs[o])
                    else:
                        w.append(techs[o])
                print("The hybrid is better than ", w, " and worse than ", b)
        write_recommendations(rec_dictionary, targets)
    print("work done ", i, " techniques, ", techs, " combined with the linear method on ",
          max_weight, "weight combinations ", weights)


def combine(dictionaries, weights, num, decays):
    if num < 1:
        return {}
    elif num == 1:
        print("No work to do")
        return dictionaries[0]
    else:
        final_dic = {}
        for key in dictionaries[0].keys():
            lists = []
            final_list = []
            for i in range(num):
                lists.append(multiply_list_of_tuples(dictionaries[i][key], weights[i], decays[i]))
            for i in range(num):
                final_list = extend_with_repetitions(final_list, lists[i])
            final_list.sort(key=lambda tup: tup[1], reverse=True)
            final_dic[key] = final_list[:parsed.rec_length]
        return final_dic


def extend_with_repetitions(list_one, list_two):
    len_one = len(list_one)
    len_two = len(list_two)
    extended_list = []
    for i in range(len_one):
        pos = in_list(list_one[i], list_two)
        if pos != -1:
            if list_one[pos][1] > 0:
                uno = list_one[pos][1]
            else:
                uno = 0.1
            if list_two[pos][1] > 0:
                due = list_two[pos][1]
            else:
                due = 0.1
            new_tuple = list_one[i][0], uno + due
            extended_list.append(new_tuple)
        else:
            extended_list.append(list_one[i])
    for i in range(len_two):
        if in_list(list_two[i], extended_list) == -1:
            extended_list.append(list_two[i])
    return extended_list


def in_list(element, list_):
    for i in range(len(list_)):
        if list_[i][0] == element[0]:
            return i
    return -1


def multiply_list_of_tuples(list_one, weight, decay):
    for i in range(len(list_one)):
        tup = list_one[i]
        list_one[i] = (tup[0], tup[1] * weight * decay(i))

    return list_one


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
    assert float(0) <= map_score <= float(1)
    return map_score


def constant_decay(i):
    return 1


def linear_decay(i):
    return 1 - (i / parsed.rec_length)


def gradual_decay(i):
    return 3/(3 + i)


main()
