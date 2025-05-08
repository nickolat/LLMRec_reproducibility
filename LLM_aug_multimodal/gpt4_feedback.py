import os
import base64
from openai import AzureOpenAI
import pandas as pd
from utilities import *

gpt4_feedback_dict = dictionary({},"gpt4_feedback_dict")
file_name = file_path + "gpt4_feedbackLogging.txt"
logger = Logger(filename=file_name, is_debug=args.debug)

def construct_prompting(item_list, candidate_list):

    """
    Function that creates the prompt to use as input in the request to GPT
    :param item_list: list of products' ids consumed by the user (to change in case of different datasets)
    :param candidate_list: list of to be recommended products' ids for each user' (to change in case of different datasets)
    """

    history_string = "\nUser history:\n"

    links = []
    links_ = []

    for index in item_list:
        url = urls['url'][index]
        title = df['title'][index]

        year = df['year'][index]
        history_string += f"[{index}] {title}, {year} (title, year) \n"

        links.append({"type": "image_url", "image_url": {"url": url}})

    candidate_string = "Candidates:\n"
    for index in candidate_list:
        url = urls['url'][index]
        title = df['title'][index]

        year = df['year'][index]
        candidate_string += f"[{index}] {title}, {year} (title, year) \n"
        product = "movie"

        links_.append({"type": "image_url", "image_url": {"url": url}})

    output_format = (
        f"Please output the index of the user's favorite and least favorite {product} "
        "only from Candidates, but not from User History. "
        "Use exclusively index from Candidates at the beginning of each line.\n"
        "This is the output format: Two numbers separated by '::' (no brackets, only numbers). "
        f"Example --> most_favorite_{product}::least_favourite_{product}\n"
        "Please provide no additional text or reasoning.\n\n"
    )

    prompt = (
            f"You are a {product} recommendation system and required to recommend users with {product} based on their User History. "
            f"You are given, for each user: \n1) both textual and visual information (listed in the same order) about {product} they consumed (in User History); \n"
            f"2) both textual and visual information (listed in the same order) about the possible candidates from which to choose the index of the {product} to recommend (in Candidates) \n."
            + history_string + candidate_string + output_format
    )
    return prompt, links, links_

def LLM_request(idx):
    """
    Function that sends the request to GPT
    :param idx: index of the product to get information on
    """

    # If idx has already been processed:
    if idx in gpt4_feedback_dict:
        return

    # Otherwise:
    try:

        item_list = adjacency_list_dict[idx]
        candidate_list = np.array(candidate_indices_dict[idx])
        prompt, links, links_ = construct_prompting(item_list, candidate_list)

        req = [
            {"type": "text", "text": prompt},
            {"type": "text", "text": f"The {len(item_list)} following base64 encoded images are related to User History. "
                                     "Images of products are given to you in the same order of the listing in User History. "
                                     "So, the first base64 encoded image corresponds to the first product title, the second base64 encoded image corresponds to the second product title and so on."}
        ]
        req.extend(links)
        req.append({"type": "text", "text": f"The {len(candidate_list)} following base64 encoded images are related to Candidates. "
                                     "Images of products are given to you in the same order of the listing in Candidates. "
                                            "So, the first base64 encoded image corresponds to the first product title, the second base64 encoded image corresponds to the product title and so on. "})
        req.extend(links_)

        # Initialize Azure OpenAI client with key-based authentication
        try:
            client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=subscription_key,
                api_version="2024-05-01-preview",
            )

        # Generate the completion
            completion = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": req}],
                max_tokens=1024,
                temperature=0.6,
                top_p=0.1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )

            content = completion.choices[0].message.content
            logger.logging(f"Content received: {content}")

            pos_sample, neg_sample = map(int, content.split("::"))
            if pos_sample not in candidate_list or neg_sample not in candidate_list:
                raise Exception(f"Products are not chosen from candidates: {candidate_list}")

            gpt4_feedback_dict[idx] = {0: pos_sample, 1: neg_sample}

        except Exception as e:
            print(f"Error generating completion: {e}")

        with open(file_path + 'gpt4_feedback_dict', 'wb') as f:
            pickle.dump(gpt4_feedback_dict, f)

    except Exception as e:
        print(f"Error generating completion: {e}")

for index in range(len(adjacency_list_dict)):
    logger.logging(f"Processing index: {index}")
    LLM_request(index)

for i in range(len(adjacency_list_dict)):
    if i not in gpt4_feedback_dict:
        gpt4_feedback_dict[i] = {0: np.nan, 1: np.nan}

with open(file_path + 'gpt4_feedback_dict', 'wb') as f:
        pickle.dump(gpt4_feedback_dict, f)
