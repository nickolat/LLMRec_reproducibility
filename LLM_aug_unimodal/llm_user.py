import os
import pandas as pd
import pickle
import time
import numpy as np
from tqdm import tqdm
from utils import *


def construct_prompting(item_attribute, item_list):
    """
    Function that creates the prompt to use in the LLM request, for the default dataset (Netflix).
    :param item_attribute: pandas DataFrame containing item attributes
    :param item_list: list of items' ids for the current user
    """
    # create history string
    history_string = "User history:\n"
    for index in item_list:
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += str(year) + ", "
        history_string += title + "\n"
    # output format
    output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    # create prompt
    prompt = "You are required to generate user profile based on the history of user, that each movie with year, title.\n"
    prompt += history_string
    prompt += output_format
    return prompt


def construct_prompting_amazon(item_attribute, item_list):
    """
    Function that creates the prompt to use in the LLM request, for the Amazon-music dataset.
    :param item_attribute: pandas DataFrame containing item attributes
    :param item_list: list of items' ids for the current user
    """
    # create history string
    history_string = "User history:\n"
    for index in item_list:
        title = item_attribute['title'][index]
        genres = item_attribute['genres'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += str(title) + ", "
        history_string += str(genres) + "\n"
    # output format
    output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked artists\':liked artists, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    # create prompt
    prompt = "You are required to generate user profile based on the history of user, that has each movie with title, genres.\n"
    prompt += history_string
    prompt += output_format
    return prompt


def LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, client):
    """
    Function that sends the request to the LLM.
    :param toy_item_attribute: pandas DataFrame containing item attributes
    :param adjacency_list_dict: dictionary containing adjacency list
    :param index: int index indicating current user
    :param model_type: string indicating the specific LLM used
    :param augmented_user_profiling_dict: dictionary where to save augmented user profiling data
    :param client: string indicating the LLM client ('azure', 'azure-llama', 'openai')
    """
    if index in augmented_user_profiling_dict:
        return 0
    else:
        try:
            print(f"Index: {index}")
            if dataset == "amazon":
                prompt = construct_prompting_amazon(toy_item_attribute, adjacency_list_dict[index])
            else:
                prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index])

            messages = [{"role": "user", "content": prompt}]
            params={
                "model": model_type,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.1,
                "stream": False
            }

            if client=='azure':
                content = generate_completion_azure(params=params)
            elif client=='azure-llama':
                content = generate_completion_azure_llama(params=params)
            else:
                print("Using OpenAI client")
                content = generate_completion_openai(params=params)

            print(f"content: {content}")
            augmented_user_profiling_dict[index] = content
            if (index % 10 == 0) or (index == len(adjacency_list_dict) - 1):
                pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict','wb'))

        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
            LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, client)
        except ValueError as ve:
            print("An error occurred while parsing the response:", str(ve))
            time.sleep(5)
            LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, client)
        except KeyError as ke:
            print("An error occurred while accessing the response:", str(ke))
            time.sleep(5)
            LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, client)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
            LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, client)
        return 1


def LLM_request_embedding(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, client):
    """
    Function that sends the request to the LLM to obtain embeddings.
    :param augmented_user_profiling_dict: dictionary containing augmented user profiling data
    :param index: int index indicating current user
    :param model_type: string indicating the specific LLM used
    :param augmented_user_init_embedding: dictionary where to save augmented user profile embeddings
    :param client: string indicating the LLM client ('azure', 'azure-llama', 'openai')
    """
    if index in augmented_user_init_embedding:
        return 0
    else:
        try:
            params={
                "model": model_type,
                "input": str(augmented_user_profiling_dict[index])
            }
            if client=='azure' or client=='azure-llama':
                content = generate_embeddings_azure(params=params)
            else:
                print("Using Openai API")
                content = generate_embeddings_openai(params=params)

            augmented_user_init_embedding[index] = np.array(content)
            if (index % 20 == 0) or (index == len(augmented_user_profiling_dict) - 1):
                pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding','wb'))

        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
            LLM_request_embedding(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, client)
        except ValueError as ve:
            print("An error occurred while parsing the response:", str(ve))
            time.sleep(5)
            LLM_request_embedding(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, client)
        except KeyError as ke:
            print("An error occurred while accessing the response:", str(ke))
            time.sleep(5)
            LLM_request_embedding(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, client)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
            LLM_request_embedding(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, client)
        return 1



### step1: generate user profiling ##################################################################################
def generate_user_profiling(g_model_type, client):
    """
    Function that creates the necessary dictionaries and calls LLM_request
    :param g_model_type: string indicating the specific LLM used
    :param client: string indicating the LLM client used ('azure', 'azure-llama', 'openai')
    """
    ### read item_attribute
    toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=item_attribute_cols)

    ### write augmented dict
    augmented_user_profiling_dict = {}
    if os.path.exists(file_path + "augmented_user_profiling_dict"):
        print(f"The file augmented_user_profiling_dict exists.")
        augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict','rb'))
    else:
        print(f"The file augmented_user_profiling_dict does not exist.")
        pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict','wb'))

    ### read adjacency_list (user: list of item indices)
    adjacency_list_dict = {}
    train_mat = pickle.load(open(file_path + 'train_mat','rb'))
    for index in range(train_mat.shape[0]):
        data_x, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y

    print(f"## Start generating user profiling using model type: {g_model_type}\n")
    # LLM request
    for index in tqdm(range(0, len(adjacency_list_dict.keys()))):
        print(index)
        LLM_request(toy_item_attribute, adjacency_list_dict, index, g_model_type, augmented_user_profiling_dict, client)



# ### step2: generate user embedding ################################################################################
def generate_user_embedding(model_type_embedding, client):
    """
    Function that creates the necessary dictionaries and calls LLM_request_embedding
    :param model_type_embedding: string indicating the specific LLM used for embeddings generation
    :param client: string indicating the LLM client used ('azure', 'azure-llama', 'openai')
    """
    ### read user_profile
    augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict','rb'))

    ### write augmented_user_init_embedding
    augmented_user_init_embedding = {}
    if os.path.exists(file_path + "augmented_user_init_embedding"):
        print(f"The file augmented_user_init_embedding exists.")
        augmented_user_init_embedding = pickle.load(open(file_path + 'augmented_user_init_embedding','rb'))
    else:
        print(f"The file augmented_user_init_embedding does not exist.")
        pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding','wb'))

    for index,value in tqdm(enumerate(augmented_user_profiling_dict.keys())):
        print(index)
        LLM_request_embedding(augmented_user_profiling_dict, index, model_type_embedding, augmented_user_init_embedding, client)



# # ### step3: get user embedding ##################################################################################
def generate_user_embedding_final():
    """
    Function that generates the final file for augmented user profile embeddings.
    """
    augmented_user_init_embedding = pickle.load(open(file_path + 'augmented_user_init_embedding','rb'))
    augmented_user_init_embedding_list = []
    for i in range(len(augmented_user_init_embedding)):
        augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
    augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
    pickle.dump(augmented_user_init_embedding_final, open(file_path + 'augmented_user_init_embedding_final','wb'))



# #######################################################################################################

# 1) USER PROFILE GENERATION
generate_user_profiling(model_type, client)

# 2) EMBEDDINGS GENERATION
generate_user_embedding(model_type_embedding, client)
generate_user_embedding_final()
