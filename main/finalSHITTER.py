
import pandas as pd
import argparse
import datetime
import random

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
parser.add_argument('--rec1', type=str, default="rec1.csv")
parser.add_argument('--rec2', type=str, default="rec2.csv")
parser.add_argument('--rec3', type=str, default="rec3.csv")
parser.add_argument('--rec4', type=str, default="rec4.csv")
parser.add_argument('--rec5', type=str, default="rec5.csv")
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
        print(best_choices)
        for recommendation in best_choices:
            line += str(recommendation) + item_item_sep
            dic_line += "("+str(recommendation)+","+str(recommendation)+"),"

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


def main(args):
    targets, rec1, rec2, rec3 = get_targets()
    rec_dic = {}
    for target in targets:
        print(target, "---",rec1[target], rec2[target], rec3[target])
        listino = []
        possibilities = []
        lista = [rec1[target], rec2[target], rec3[target]]
        for i in range(5):
            notgo = True
            times = 0
            for l in range(3):
                possibilities.append(lista[l][i])
            while(notgo):
                which_dict = random.randint(0, 2)
                if(times >= 3):
                    decision = possibilities[random.randint(0,len(possibilities)-1)]
                else:
                    decision = lista[which_dict][i]
                notgo = decision in listino
                if(not notgo):
                    listino.append(decision)
                else:
                    times += 1
        for i in range(4):
            flip = random.randint(0, 323)
            if flip == 34:
                temp = listino[i]
                listino[i] = listino[i+1]
                listino[i+1] = temp
        rec_dic[target] = []
        for i in range(5):
            rec_dic[target].append(listino[i])
    print(rec_dic)
    write_recommendations(rec_dic, targets)

    return rec_dic


def get_targets():
    with open(args.target_users, 'r') as j:
        target_reader = pd.read_csv(args.target_users, delimiter='\t')
    target_users = target_reader[args.user_key].as_matrix()
    with open(args.rec1, 'r') as j:
        rec1 = pd.read_csv(args.rec1, delimiter='\t')
    recommendations_one = rec1.as_matrix()
    with open(args.rec2, 'r') as j:
        rec2 = pd.read_csv(args.rec2, delimiter='\t')
    recommendations_two = rec2.as_matrix()
    with open(args.rec3, 'r') as j:
        rec3 = pd.read_csv(args.rec3, delimiter='\t')
    recommendations_three = rec3.as_matrix()
    return target_users, get_dic(recommendations_one), get_dic(recommendations_two), get_dic(recommendations_three)


def get_dic(reccomandations_one):
    rec_dic_one = {}
    for i in reccomandations_one:
        rec_dic_one[int(i[0].split(",")[0])] = []
        for j in range(0,5):
            jey = i[0].split(",")[1].split(" ")[j]
            if(jey is not " "):
                rec_dic_one[int(i[0].split(",")[0])].append(int(jey))
    return rec_dic_one

main(args)
