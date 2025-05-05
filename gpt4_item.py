from utilities import *
from openai import AzureOpenAI

# Generates augmented item attributes (gpt4_item_dict) and augmented embeddings (gpt4_ItemEmbedding_dict) using gpt-4-turbo

# Set model to be used and dictionaries
gpt4_item_dict = dictionary({},"gpt4_item_dict")
gpt4_embedding_dict = dictionary({'year':{}, 'title':{}, 'director':{}, 'country':{}, 'language':{}},"gpt4_ItemEmbedding_dict") # Netflix Dataset
file_name = file_path + "gpt4_itemLogging.txt"
logger = Logger(filename=file_name, is_debug=args.debug)

endpoint = os.getenv("ENDPOINT_URL", " Insert your Endpoint URL here ")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", key)

def construct_prompting(idx):
    """
    Function that creates the prompt to use as input in the request to GPT
    :param idx: index (of the dataset) of the product to get information on (to change in case of different datasets)
    """

    # Construct prompt
    title = df['title'][idx]
    link = urls['url'][idx]

    year = df['year'][idx]
    prod = f"{title}, {year} (title, year)"
    product = "movie"

    prompt = (f"The inquired information of the given {product} is: director, country, language. \n" # Change the format for different dataset
              f"The textual information is the following: {prod}. \n"
              f"The visual information provided is the {product} cover (in the base64 encoded image). \n"
              f"Please output them in the following format: director::country::language. \n" # Change the format for different dataset
              f"Output only and exclusively the content in the specified format, with no extra text, no reasoning, and no indices.")

    return prompt, link

def LLM_request(idx):

    """
    Function that sends the request to GPT
    :param idx: index of the product to get information on
    """

    # If idx has already been processed:
    if idx in gpt4_item_dict:
        return

    # Otherwise:
    try:

        prompt, link = construct_prompting(idx)
        req =  [
            {"type": "text", "text": prompt},
            {"type": "image_url","image_url": {"url": link}},
        ]

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
            logger.logging(f"{df.iloc[idx]['title']}, {df.iloc[idx]['year']}--> {content}")

            elements = content.split("::")
            gpt4_item_dict[idx] = {0: elements[0], 1: elements[1], 2: elements[2]}

        except Exception as e:
            print(f"Error generating completion: {e}")

        with open(file_path + 'gpt4_item_dict', 'wb') as x:
            pickle.dump(gpt4_item_dict, x)

    except Exception as e:
        print(f"Error generating completion: {e}")

for idx in range(len(df)):
    logger.logging(f"Processing index: {idx}") # Print the index for tracking
    LLM_request(idx)

director_list, country_list, language_list = [], [], []
for i in range(len(df)):
    if i in gpt4_item_dict:

        if gpt4_item_dict[i][0]!='' and 'director' not in gpt4_item_dict[i][0]:
            director_list.append(gpt4_item_dict[i][0])
        else:
            director_list.append('Unknown')
            gpt4_item_dict[i][0] = 'Unknown'

        if gpt4_item_dict[i][1]!='' and 'country' not in gpt4_item_dict[i][1]:
            country_list.append(gpt4_item_dict[i][1])
        else:
            country_list.append('Unknown')
            gpt4_item_dict[i][1] = 'Unknown'

        if gpt4_item_dict[i][2]!='' and 'language' not in gpt4_item_dict[i][2]:
            language_list.append(gpt4_item_dict[i][2])
        else:
            language_list.append('Unknown')
            gpt4_item_dict[i][2] = 'Unknown'
    else:
        director_list.append('Unknown')
        country_list.append('Unknown')
        language_list.append('Unknown')
        gpt4_item_dict[i] = {0:'Unknown', 1:'Unknown', 2:'Unknown'}

df['director'] = pd.Series(director_list)
df['country'] = pd.Series(country_list)
df['language'] = pd.Series(language_list)

# Creates a gpt4_item.csv file where augmented information about movies is saved
df.to_pickle(file_path+"augmented_movies")
with open(file_path + 'gpt4_item_dict', 'wb') as x:
    pickle.dump(gpt4_item_dict, x)

def LLM_embedding(idx, v):

    """
    Function that sends the request to GPT to get the embedding for each side information of the product
    """

    # If idx has already been processed:
    if idx in gpt4_embedding_dict[v]:
        return

    # Otherwise:
    try:
        print("I'm getting the embedding for: " + v + " of the product.")

        client = AzureOpenAI(
            api_key=subscription_key,
            api_version="2023-05-15",
            azure_endpoint=endpoint
        )

        embedding = client.embeddings.create(input = str(df[v][idx]), model="text-embedding-ada-002").data[0].embedding

        gpt4_embedding_dict[v][idx] = np.array(embedding)
        pickle.dump(gpt4_embedding_dict, open(file_path + 'gpt4_ItemEmbedding_dict','wb'))

    except Exception as e:
        print(f"Error generating completion: {e}")

for idx in range(len(df)):
    print(f"Processing index: {idx}")
    for value in ['year', 'title', 'director', 'country', 'language']:
        LLM_embedding(idx, value)

