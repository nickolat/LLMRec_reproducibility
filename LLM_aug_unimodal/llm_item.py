import os
import time
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from utils import *


def construct_prompting(item_attribute, indices):
    """
    Function that creates the prompt to use in the LLM request, for the default dataset (Netflix).
    :param item_attribute: pandas DataFrame containing item attributes
    :param indices: item indices to get information on
    """
    # pre string
    pre_string = "You are now a search engine, and required to provide the inquired information of the given movies below:\n"
    # create item list
    item_list_string = ""
    for index in indices:
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        item_list_string += "["
        item_list_string += str(index)
        item_list_string += "] "
        item_list_string += str(year) + ", "
        item_list_string += title + "\n"
    # output format
    output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    # create prompt
    prompt = pre_string + item_list_string + output_format
    return prompt


def construct_prompting_amazon(item_attribute, indices):
    """
    Function that creates the prompt to use in the LLM request, for the Amazon-music dataset.
    :param item_attribute: pandas DataFrame containing item attributes
    :param indices: item indices to get information on
    """
    # pre string
    pre_string = "You are now a search engine, and required to provide the inquired information of the given music album below:\n"
    # create item list
    item_list_string = ""
    for index in indices:
        title = item_attribute['title'][index]
        genres = item_attribute['genres'][index]
        item_list_string += "["
        item_list_string += str(index)
        item_list_string += "] "
        item_list_string += str(title) + ", "
        item_list_string += str(genres) + "\n"
    # output format
    output_format = "The inquired information is : artist, country, language.\nAnd please output them in form of: \nartist::country::language\nplease output only the content in the form above, i.e., artist::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    # create prompt
    prompt = pre_string + item_list_string + output_format
    return prompt


def LLM_request(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt, client):
    """
    Function that sends the request to the LLM.
    :param toy_item_attribute: pandas DataFrame containing item attributes
    :param index: int index indicating current item
    :param model_type: string indicating the specific LLM used
    :param augmented_attribute_dict: dictionary where to save augmented item attributes
    :param error_cnt: int indicating error counter
    :param client: string indicating the LLM client ('azure', 'azure-llama', 'openai')
    """
    if index in augmented_attribute_dict:
        return 0
    else:
        try: 
            print(f"Index: {index}")
            if dataset=="amazon":
                prompt = construct_prompting_amazon(toy_item_attribute, [index])
            else:
                prompt = construct_prompting(toy_item_attribute, [index])

            messages = [{"role": "user", "content": prompt}]
            params={
                "model": model_type,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.6,
                "top_p": 0.1,
                "stream": False
            }

            if not client:
                content = "unknown::unknown::unknown"
            elif client=='azure':
                content = generate_completion_azure(params=params)
            elif client=='azure-llama':
                content = generate_completion_azure_llama(params=params)
            else:
                print("Using Openai API")
                content = generate_completion_openai(params=params)

            print(f"content: {content}")
            if not content:
                content = "unknown::unknown::unknown"
            elements = content.split("::")
            director = elements[0]
            country = elements[1]
            language = elements[2]
            augmented_attribute_dict[index] = {}
            augmented_attribute_dict[index][0] = director
            augmented_attribute_dict[index][1] = country
            augmented_attribute_dict[index][2] = language

            if (index % 10 == 0) or (index == toy_item_attribute.shape[0] - 1):
                pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))

        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
            LLM_request(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt, client)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            error_cnt += 1
            if error_cnt==5:
                client = None
            LLM_request(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt, client)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            error_cnt += 1
            if error_cnt==5:
                client = None
            LLM_request(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt, client)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            error_cnt += 1
            if error_cnt==5:
                client = None
            LLM_request(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt, client)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
            LLM_request(toy_item_attribute, index, model_type, augmented_attribute_dict, error_cnt, client)
        return 1



def LLM_request_embedding(toy_augmented_item_attribute, index, model_type, augmented_atttribute_embedding_dict, client):
    """
    Function that sends the request to the LLM to obtain embeddings.
    :param toy_augmented_item_attribute: dictionary containing augmented item attributes
    :param index: int index indicating current item
    :param model_type: string indicating the specific LLM used
    :param augmented_atttribute_embedding_dict: dictionary where to save augmented item attributes embeddings
    :param client: string indicating the LLM client ('azure', 'azure-llama', 'openai')
    """
    flag_dump = False
    print(f"Index: {index}")
    for value in augmented_atttribute_embedding_dict.keys():
        if index in augmented_atttribute_embedding_dict[value]:
            continue
        else:
            flag_dump = True
            print(value)
            try:
                params={
                    "model": model_type,
                    "input": str(toy_augmented_item_attribute[value][index])
                }
                if client=='azure' or client=='azure-llama':
                    content = generate_embeddings_azure(params=params)
                else:
                    print("Using Openai API")
                    content = generate_embeddings_openai(params=params)

                augmented_atttribute_embedding_dict[value][index] = content

            except requests.exceptions.RequestException as e:
                print("An HTTP error occurred:", str(e))
                time.sleep(5)
                LLM_request_embedding(toy_augmented_item_attribute, index, model_type, augmented_atttribute_embedding_dict, client)
            except ValueError as ve:
                print("An error occurred while parsing the response:", str(ve))
                time.sleep(5)
                LLM_request_embedding(toy_augmented_item_attribute, index, model_type, augmented_atttribute_embedding_dict, client)
            except KeyError as ke:
                print("An error occurred while accessing the response:", str(ke))
                time.sleep(5)
                LLM_request_embedding(toy_augmented_item_attribute, index, model_type, augmented_atttribute_embedding_dict, client)
            except Exception as ex:
                print("An unknown error occurred:", str(ex))
                time.sleep(5)
                LLM_request_embedding(toy_augmented_item_attribute, index, model_type, augmented_atttribute_embedding_dict, client)

    if flag_dump:
        if (index % 20 == 0) or (index == toy_augmented_item_attribute.shape[0] - 1):
            pickle.dump(augmented_atttribute_embedding_dict, open(file_path + 'augmented_atttribute_embedding_dict','wb'))



# ############################# step 1: built item attribute #########################################################
def built_item_attribute(model_type, client):
    """
    Function that creates the augmented attributes dictionary, and then calls LLM_request.
    :param model_type: string indicating the specific LLM used
    :param client: string indicating the LLM client used ('azure', 'azure-llama', 'openai')
    """
    error_cnt = 0
    ### write augmented dict
    augmented_attribute_dict = {}
    if os.path.exists(file_path + "augmented_attribute_dict"):
        print(f"The file augmented_attribute_dict exists.")
        augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb'))
    else:
        print(f"The file augmented_attribute_dict does not exist.")
        pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))

    ### read item attribute file
    toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=item_attribute_cols)

    print(f"## Start generating augmented item attributes using model type: {model_type}\n")
    for i in tqdm(range(0, toy_item_attribute.shape[0])):
        LLM_request(toy_item_attribute, i, model_type, augmented_attribute_dict, error_cnt, client)



# ############################# step 2: generate new csv ###########################################################
def generate_augmented_csv():
    """
    Function that generates the augmented item attributes csv file.
    """
    raw_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=item_attribute_cols)
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb'))
    director_list, country_list, language_list = [], [], []
    for i in range(len(augmented_attribute_dict)):
        director_list.append(augmented_attribute_dict[i][0])
        country_list.append(augmented_attribute_dict[i][1])
        language_list.append(augmented_attribute_dict[i][2])
    director_series = pd.Series(director_list)
    country_series = pd.Series(country_list)
    language_series = pd.Series(language_list)

    if dataset=="amazon":
        raw_item_attribute['artist'] = director_series
    else:
        raw_item_attribute['director'] = director_series
    raw_item_attribute['country'] = country_series
    raw_item_attribute['language'] = language_series
    raw_item_attribute.to_csv(file_path + 'augmented_item_attribute_agg.csv', index=False, header=None)


# ############################# step 3: generate item atttribute embedding #########################################
def generate_item_attribute_embedding(model_type_embedding, client):
    """
    Function that creates the necessary dictionaries, and then it calls LLM_request_embedding
    :param model_type_embedding: string indicating the specific LLM used for embeddings generation
    :param client: string indicating the LLM client used ('azure', 'azure-llama', 'openai')
    """
    ### write augmented dict
    augmented_atttribute_embedding_dict = {}
    for attr_name in item_attribute_cols+item_attribute_augm:
        if attr_name == 'id':
            continue
        augmented_atttribute_embedding_dict[attr_name] = {}

    ### read augmented item attribute file
    toy_augmented_item_attribute = pd.read_csv(file_path + 'augmented_item_attribute_agg.csv',
                                               names=item_attribute_cols+item_attribute_augm)

    file_name = "augmented_atttribute_embedding_dict"
    if os.path.exists(file_path + file_name):
        print(f"The file augmented_atttribute_embedding_dict exists.")
        augmented_atttribute_embedding_dict = pickle.load(open(file_path + file_name,'rb'))
    else:
        print(f"The file augmented_atttribute_embedding_dict does not exist.")
        pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))

    print(f'## Start Embedding phase with model: {model_type_embedding}\n')
    for i in tqdm(range(0, toy_augmented_item_attribute.shape[0])):
        LLM_request_embedding(toy_augmented_item_attribute, i, model_type_embedding, augmented_atttribute_embedding_dict, client)


# ############################# step 4: get separate embedding matrix ################################################
def generate_augmented_total_embed_dict():
    """
    Function that generates the final file for augmented item attributes embeddings.
    """
    augmented_total_embed_dict = {}
    for attr_name in item_attribute_cols+item_attribute_augm:
        if attr_name == 'id':
            continue
        augmented_total_embed_dict[attr_name] = []

    augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb'))
    for value in augmented_atttribute_embedding_dict.keys():
        for i in range(len(augmented_atttribute_embedding_dict[value])):
            augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])
        augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])
    pickle.dump(augmented_total_embed_dict, open(file_path + 'augmented_total_embed_dict','wb'))


# ##############################################################################################################

## 1) ITEM ATTRIBUTES GENERATION
built_item_attribute(model_type, client)
generate_augmented_csv()

## 2) EMBEDDINGS GENERATION
generate_item_attribute_embedding(model_type_embedding, client)
generate_augmented_total_embed_dict()
