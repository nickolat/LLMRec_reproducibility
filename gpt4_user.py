from utilities import *
from openai import AzureOpenAI

# Generates augmented user attributes (gpt4_user_dict) and augmented embeddings (gpt4_UserEmbedding_dict) using gpt-4-turbo

# Set model to be used and dictionaries
gpt4_user_dict = dictionary({},"gpt4_user_dict")
gpt4_embedding_dict = dictionary({}, "gpt4_UserEmbedding_dict")
file_name = file_path + "gpt4_userLogging.txt"
logger = Logger(filename=file_name, is_debug=args.debug)

endpoint = os.getenv("ENDPOINT_URL", " Insert your endpoint URL here ")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", key)

def construct_prompting(item_list):

    """
    Function that creates the prompt to use as input in the request to GPT
    :param item_list: list of products' ids watched by the user (to change in case of different datasets)
    """

    history_string = "User history:\n"
    links = []

    for index in item_list:
        url = urls['url'][index]
        title = df['title'][index]

        year = df['year'][index]
        history_string += f"{title}, {year} (title, year) \n"
        product = "movie"

        links.append({"type": "image_url", "image_url": {"url": url}})

    output_format = (
        "Please output the following information of user, output format:"
        "\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, " # Change format for different Dataset
        "\'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', "
        "but make an educated guess based on the available information and fill in the specific content.\n"
        "Please output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. "
        "Reiterating once again!! Please only output the content after \"output format: \", "
        "and do not include any other content such as introduction or acknowledgments.\n\n"
    )

    prompt = (
            "You are required to generate user profile based on the history of products consumed by the user. "
            f"Each product has textual and visual information: \n"
            f"1) Textual information consists in User History. \n"
            f"2) Visual information is the associated {product} covers in base64 encoded images. "
            f"{product} covers are given to you in the same order of the listing in User History."
            + history_string + output_format
    )
    return prompt, links


def LLM_request(idx):

    """
    Function that sends the request to GPT
    :param idx: index of the product to get information on
    """

    # If idx has already been processed:
    if idx in gpt4_user_dict:
        return

    # Otherwise:
    try:

        item_list = adjacency_list_dict[idx]
        prompt, links = construct_prompting(item_list)

        req =  [
            {"type": "text", "text": prompt},
            {"type": "text",
             "text": f"The {len(item_list)} following base64 encoded images are related to User History. "
                     "Images of products are given to you in the same order of the listing in User History. "
                     "So, the first base64 encoded image corresponds to the first product title, the second base64 encoded image corresponds to the second product title and so on."}

        ]
        req.extend(links)

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
            logger.logging(f"User n° {idx} --> {content}")

            gpt4_user_dict[idx] = content

        except Exception as e:
            print(f"Error generating completion: {e}")

        with open(file_path + 'gpt4_user_dict', 'wb') as f:
            pickle.dump(gpt4_user_dict, f)

    except Exception as e:
        print(f"Error generating completion: {e}")

for index in range(len(adjacency_list_dict)):
    logger.logging(f"Processing index: {index}")
    LLM_request(index)

for i in range(len(adjacency_list_dict)):
    if i not in gpt4_user_dict:
        # Change string for different Dataset
        gpt4_user_dict[i] = "{'age': Unknown, 'gender': Unknown, 'liked genre': Unknown, 'disliked genre': Unknown, 'liked directors': Unknown, 'country': Unknown, 'language': Unknown}"

with open(file_path + 'gpt4_user_dict', 'wb') as f:
        pickle.dump(gpt4_user_dict, f)

def LLM_embedding(idx):

    """
    Function that sends the request to GPT to get the embedding for each side information of the product
    """

    # If idx has already been processed:
    if idx in gpt4_embedding_dict:
        return

    # Otherwise:
    try:
        print("I'm getting the embedding for user n°: " + str(idx))

        client = AzureOpenAI(
            api_key=subscription_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )

        embedding = client.embeddings.create(input=str(gpt4_user_dict[idx]), model="text-embedding-ada-002").data[0].embedding

        gpt4_embedding_dict[idx] = np.array(embedding)
        pickle.dump(gpt4_embedding_dict, open(file_path + 'gpt4_UserEmbedding_dict', 'wb'))

    except Exception as e:
        print(f"Error generating completion: {e}")

for index in range(len(adjacency_list_dict)):
    print(f"Processing index: {index}")
    LLM_embedding(index)

