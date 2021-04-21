import os
import gc
import time
import pickle
import tensorflow as tf
import sys
import operator
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

def load_dataset():
    userprofile = defaultdict(list)
    ratings_list = []
    # Load dataset
    data = pd.read_csv("./data/ratings.dat", sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])

    # Create list of unique movie IDs
    data['movieid_cat'] = data.movieid.astype('category').cat.codes
    print(data.head())
    
    # Order by timestamp and set rating column to 1 to indicate an interaction
    for i, (idx, row) in enumerate(data.sort_values(by='timestamp').iterrows()):
        uid, iid, rating, timestamp, iid_cat = row
        #print(idx, uid, iid, rating, timestamp)
        
        output = (uid, userprofile[uid], [], iid_cat, 1, timestamp)

        userprofile[uid].append(iid_cat)
        ratings_list.append(output)

    # Split into test, validation and training sets
    train_set, test_set = train_test_split(ratings_list, test_size=0.2, random_state=1)

    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2

    # Count number of unique user and movie IDs
    user_count = data['userid'].nunique()
    item_count = data['movieid_cat'].nunique()
    print(f"item_count = {item_count}, user_count = {user_count}")

    print("len(train_set): ", len(train_set))
    print("len(test_set): ", len(test_set))
    print("len(val_set): ", len(val_set))

    with open("./data/dataset_movielens.pkl", "wb") as f:
        pickle.dump(train_set, f)
        pickle.dump(val_set, f)
        pickle.dump(test_set, f)
        pickle.dump([], f)
        pickle.dump((user_count, item_count, 0), f)

def trim_dataset(k_item=1, k_user=1):
    userprofile = defaultdict(list)
    ratings_list = []
    
    # Load dataset
    data = pd.read_csv("./data/ratings.dat", sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])
    
    print(data.head())
    
    # Remove users/movies with less than specified number of ratings
    while (len(data[data['movieid'].map(data['movieid'].value_counts()) < int(k_item)]) > 0 or len(data[data['userid'].map(data['userid'].value_counts()) < int(k_user)]) > 0):
        data = data[data['movieid'].map(data['movieid'].value_counts()) >= int(k_item)]
        data = data[data['userid'].map(data['userid'].value_counts()) >= int(k_user)]

    print("Length: ", len(data))
    # Create list of unique movie and user IDs
    data['movieid_cat'] = data.movieid.astype('category').cat.codes
    data['userid_cat'] = data.userid.astype('category').cat.codes

    # Order by timestamp and set rating column to 1 to indicate an interaction
    for i, (idx, row) in enumerate(data.sort_values(by='timestamp').iterrows()):
        uid, iid, rating, timestamp, iid_cat, uid_cat = row
        #print(idx, uid, iid, rating, timestamp)
        
        output = (uid_cat, userprofile[uid_cat], [], iid_cat, 1, timestamp)

        userprofile[uid_cat].append(iid_cat)
        ratings_list.append(output)

    # Split into test, validation and training sets
    train_set, test_set = train_test_split(ratings_list, test_size=0.2, random_state=1)

    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2

    # Count number of unique user and movie IDs
    user_count = data['userid_cat'].nunique()
    item_count = data['movieid_cat'].nunique()
    density = ((len(data)/item_count)/user_count) * 100
    print(f"item_count = {item_count}, user_count = {user_count}")
    print(f"Density: {density}%")
    print("len(train_set): ", len(train_set))
    print("len(test_set): ", len(test_set))
    print("len(val_set): ", len(val_set))

    with open(f"./data/dataset_movielens_modified_{k_item}_{k_user}.pkl", "wb") as f:
        pickle.dump(train_set, f)
        pickle.dump(val_set, f)
        pickle.dump(test_set, f)
        pickle.dump([], f)
        pickle.dump((user_count, item_count, 0), f)
    return

def get_movies_categories():
    df_movies = pd.read_csv("./data/movies.dat", sep='::', names=['movieid', 'title', 'category'])
    enc = MultiLabelBinarizer()
    
    enc.fit(df_movies.category.apply(lambda x: x.split('|')))
    df_movies['categories_oh'] = df_movies.apply(lambda row: enc.transform([row.category.split("|")]), axis=1)

    print('All categories: ', enc.classes_)

    return enc, df_movies

def filter_by_category():
    enc, df_movies = get_movies_categories()
    # display(df_movies.head(4))

    category = ["Comedy"]

    target_categories = enc.transform([category]) # choose your categories of interest here eg: just one [["Comedy"]] or multiple: [["Comedy", "Drama"]]

    #mask = df_movies.apply(lambda row: np.count_nonzero(np.logical_and(target_categories, row.categories_oh)) >, axis=1) # any category match
    mask = df_movies.apply(lambda row: (target_categories == row.categories_oh).all(), axis=1) # exact category match

    filter_movies_df = df_movies[mask]

    userprofile = defaultdict(list)
    ratings_list = []

    data = pd.read_csv("./data/ratings.dat", sep='::', names=['userid', 'movieid', 'rating', 'timestamp'])

    if filter_movies_df is not None:
        print("Length: ", len(data))
        data = data[data.movieid.isin(filter_movies_df.movieid)]

    print("Length: ", len(data))
    # display(filter_movies_df)
    data['movieid_cat'] = data.movieid.astype('category').cat.codes
    data['userid_cat'] = data.userid.astype('category').cat.codes

    # Order by timestamp and set rating column to 1 to indicate an interaction
    for i, (idx, row) in enumerate(data.sort_values(by='timestamp').iterrows()):
        uid, iid, rating, timestamp, iid_cat, uid_cat = row
        #print(idx, uid, iid, rating, timestamp)
        
        output = (uid_cat, userprofile[uid_cat], [], iid_cat, 1, timestamp)

        userprofile[uid_cat].append(iid_cat)
        ratings_list.append(output)

    # Split into test, validation and training sets
    train_set, test_set = train_test_split(ratings_list, test_size=0.2, random_state=1)

    train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=1) # 0.25 x 0.8 = 0.2

    # Count number of unique user and movie IDs
    user_count = data['userid_cat'].nunique()
    item_count = data['movieid_cat'].nunique()
    print(f"item_count = {item_count}, user_count = {user_count}")
    print("len(train_set): ", len(train_set))
    print("len(test_set): ", len(test_set))
    print("len(val_set): ", len(val_set))

    with open(f"./data/dataset_movielens_modified_{category}.pkl", "wb") as f:
        pickle.dump(train_set, f)
        pickle.dump(val_set, f)
        pickle.dump(test_set, f)
        pickle.dump([], f)
        pickle.dump((user_count, item_count, 0), f)
    return
    

if __name__ == "__main__": 
    # print("Argument 0: ", sys.argv[0])
    # print("Argument 1: ", sys.argv[1])
    # print("Argument 2: ", sys.argv[2])
    # print("Argument 3: ", sys.argv[3])
    if sys.argv[1] == "0":
        load_dataset()
    elif sys.argv[1] == "1":
        trim_dataset(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "2":
        filter_by_category()