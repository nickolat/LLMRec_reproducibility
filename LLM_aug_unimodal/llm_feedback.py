import os
import pandas as pd
import pickle
import time
from tqdm import tqdm
from utils import *


def construct_prompting(item_attribute, item_list, candidate_list):
    """
    Function that creates the prompt to use in the LLM request, for the default dataset (Netflix).
    :param item_attribute: pandas DataFrame containing item attributes
    :param item_list: list of items' ids for the current user
    :param candidate_list: list of item candidate ids for the current user
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
    # create candidates
    candidate_string = "Candidates:\n"
    for index in candidate_list:
        year = item_attribute['year'][index.item()]
        title = item_attribute['title'][index.item()]
        candidate_string += "["
        candidate_string += str(index.item())
        candidate_string += "] "
        candidate_string += str(year) + ", "
        candidate_string += title + "\n"
    # output format
    output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    # create prompt
    prompt = ""
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt


def construct_prompting_amazon(item_attribute, item_list, candidate_list):
    """
    Function that creates the prompt to use in the LLM request, for the Amazon-music dataset.
    :param item_attribute: pandas DataFrame containing item attributes
    :param item_list: list of items' ids for the current user
    :param candidate_list: list of item candidate ids for the current user
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
    # create candidates
    candidate_string = "Candidates:\n"
    for index in candidate_list:
        title = item_attribute['title'][index.item()]
        genres = item_attribute['genres'][index.item()]
        candidate_string += "["
        candidate_string += str(index.item())
        candidate_string += "] "
        candidate_string += str(title) + ", "
        candidate_string += str(genres) + "\n"
    # output format
    output_format = "Please output the index of user\'s favorite and least favorite music album only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    # create prompt
    prompt = ""
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt


def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client):
    """
    Function that sends the request to the LLM.
    :param toy_item_attribute: pandas DataFrame containing item attributes
    :param adjacency_list_dict: dictionary containing adjacency list
    :param candidate_indices_dict: dictionary containing item candidate indices
    :param index: int index indicating current user
    :param model_type: string indicating the specific LLM used
    :param augmented_sample_dict: dictionary where to save augmented samples
    :param client: string indicating the LLM client ('azure', 'azure-llama', 'openai')
    """

    if index in augmented_sample_dict:
        print(f"g:{index}")
        return 0
    else:
        try:
            print(f"Index: {index}")

            if dataset == "amazon":
                system = "You are a music recommendation system and required to recommend user with music albums based on user history that has each album with title, genres.\n"
                prompt = construct_prompting_amazon(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])
            else:
                system = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years).\n"
                prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])

            params = {
                "model": model_type,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
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
                print("Using Openai API")
                content = generate_completion_openai(params=params)

            print(f"content: {content}")
            samples = content.split("::")
            pos_sample = int(samples[0])
            neg_sample = int(samples[1])
            augmented_sample_dict[index] = {}
            augmented_sample_dict[index][0] = pos_sample
            augmented_sample_dict[index][1] = neg_sample

            if (index % 10 == 0) or (index == len(adjacency_list_dict) - 1):
                pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict', 'wb'))

        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(5)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client)
        except EOFError as ke:
            print("EOFError: : Ran out of input error occurred while accessing the response:", str(ke))
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(5)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict, client)
        return 1



##### DATA LOADING ########################

## candidate indices name from baseline model
candidate_indices = 'candidate_indices'

### read candidate indices
candidate_indices = pickle.load(open(file_path + candidate_indices,'rb'))
candidate_indices_dict = {}
for index in range(candidate_indices.shape[0]):
    candidate_indices_dict[index] = candidate_indices[index]
### read adjacency_list
adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + 'train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y
### read item_attribute
toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=item_attribute_cols)
### write augmented dict
augmented_sample_dict = {}
if os.path.exists(file_path + "augmented_sample_dict"):
    print(f"The file augmented_sample_dict exists.")
    augmented_sample_dict = pickle.load(open(file_path + 'augmented_sample_dict','rb'))
else:
    print(f"The file augmented_sample_dict does not exist.")
    pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict','wb'))


######## LLM REQUEST ####################################################################

print(f"Using model_type: {model_type}")
for index in tqdm(range(0, len(adjacency_list_dict))):
    LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index,
                model_type, augmented_sample_dict, client)
