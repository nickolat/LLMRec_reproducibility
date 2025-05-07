import os
import pickle
from utility.parser import parse_args
import pandas as pd
import time
import requests
from utility.logging import Logger
import numpy as np

# Set path variables
key = " Insert your key here"
args = parse_args() # Change in the parser data_path and dataset
file_path = args.data_path + args.dataset + '/'
embeddingModel = "text-embedding-ada-002"

endpoint = os.getenv("ENDPOINT_URL", " Insert your Endpoint URL here")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", key)

# Load item attributes
df = pd.read_csv(file_path + 'movies.csv', names=['id', 'year', 'title']) # all movies
urls = pd.read_csv(file_path + 'urls.csv', names=['id', 'title', 'url']) # 'url' is a string of a base64 encoded image

def dictionary(file, file_name):
    if os.path.exists(file_path + file_name):
        print(f"The file {file_name} exists.")
        file = pickle.load(open(file_path + file_name,'rb'))
    else:
        print(f"The file {file_name} does not exist.")
        pickle.dump(file, open(file_path + file_name,'wb'))
    return file

# Train dataset
train_mat = pickle.load(open(file_path + 'train_mat', 'rb')) # Train Mat is a sparse matrix of interactions
adjacency_list_dict = {
    index: train_mat[index].nonzero()[1] for index in range(train_mat.shape[0])
}

# Load candidate indices from the file
with open(file_path + 'candidate_indices', 'rb') as file:
    candidate_indices = pickle.load(file)

# Create a dictionary mapping each index to its corresponding value
candidate_indices_dict = {
    index: candidate_indices[index]
    for index in range(candidate_indices.shape[0])
}


