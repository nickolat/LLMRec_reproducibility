from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
import requests

######## SETTINGS ########################################################

## DATASET ('netflix' or 'amazon')
dataset = "netflix"

## LLM ('gpt35' or 'llama')
llm = 'gpt35'

## KEYS FOR LLM API
OPENAI_KEY = "" # only if you use OPENAI API
# Note: we used three different AZURE endpoints and keys for embeddings, GPT, LLama
KEY_EMB = "Insert your key for embeddings"
KEY_GPT = "Insert your key for GPT"
KEY_LLAMA = "Insert your key for LLAMA"
## ENDPOINTS URLs
endpoint_emb = "Insert your endpoint for embeddings"
endpoint_gpt35 = "Insert your endpoint for GPT"
endpoint_llama = "Insert your endpoint for LLAMA"

##########################################################################

# DATA PATH
file_path = f"../data/{dataset}/"

# item attribute columns and augmented features
if dataset == "amazon":
    item_attribute_cols = ['id','title','genres']
    item_attribute_augm = ['artist', 'country', 'language']
elif dataset == "netflix":
    item_attribute_cols = ['id','year','title']
    item_attribute_augm = ['director', 'country', 'language']

# client and model_type settings
if llm == 'gpt35':
    client = 'azure'
    model_type = "gpt-35-turbo-16k"
elif llm == 'llama':
    client = 'azure-llama'
    model_type = "Meta-Llama-3.1-405B-Instruct"
else:
    client = 'openai'
    model_type = "gpt-35-turbo-16k"
model_type_embedding = "text-embedding-ada-002"


############## AZURE clients

# for EMBEDDINGS
client_embedding = AzureOpenAI(
    azure_endpoint = endpoint_emb,
    api_key = KEY_EMB,
    api_version = "2023-05-15"
)

# for GPT
client_gpt = AzureOpenAI(
    azure_endpoint = endpoint_gpt35,
    api_key = KEY_GPT,
    api_version = "2024-08-01-preview"
)

# for LLAMA
client_llama = ChatCompletionsClient(endpoint=endpoint_llama, credential=AzureKeyCredential(KEY_LLAMA))


############ Using AZURE API

# for GPT
def generate_completion_azure(params: dict):
    completion = client_gpt.chat.completions.create(
        model=params["model"],
        messages=params["messages"],
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        stream=False
    )
    return completion.choices[0].message.content

# for LLama
def generate_completion_azure_llama(params: dict):
    payload = {
        "messages": params["messages"],
        "max_tokens": params.get("max_tokens", None),
        "temperature": params.get("temperature", None),
        "top_p": params.get("top_p", None)
    }
    return client_llama.complete(payload).choices[0].message.content

# for embeddings
def generate_embeddings_azure(params: dict):
    return client_embedding.embeddings.create(input = [params["input"]], model=params["model"]).data[0].embedding


########### Using OPENAI API
def generate_completion_openai(params: dict):
    url = "https://api.openai.com/v1/chat/completions"
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + OPENAI_KEY
    }
    response = requests.post(url=url, headers=headers,json=params)
    message = response.json()
    content = message['choices'][0]['message']['content']
    return content

def generate_embeddings_openai(params: dict):
    url = "https://api.openai.com/v1/embeddings"
    headers={
        "Authorization": "Bearer " + OPENAI_KEY
    }
    response = requests.post(url=url, headers=headers,json=params)
    message = response.json()
    content = message['data'][0]['embedding']
    return content
