import json
import os
from openai import AzureOpenAI
import pickle
import ast

endpoint = os.getenv("ENDPOINT_URL", " Insert your endpoint here ")
deployment = os.getenv("DEPLOYMENT_NAME", "gpt-35-turbo-16k")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", " Insert your key here ")

def get_gpt_response_w_system(prompt):
    # Initialize Azure OpenAI client with key-based authentication
    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-08-01-preview",
        )
    except Exception as e:
        print(f"Error initializing AzureOpenAI client: {e}")
        exit(1)

    global system_prompt
    try:
        completion = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        user = completion.choices[0].message.content
        print(f"Content received: {user}")
        return user
    except Exception as e:
        print(f"Error generating completion: {e}")
        return None

# Read the system_prompt (Instruction) for user profile generation
system_prompt = ""
with open('./netflix_item.txt', 'r') as f:
    for line in f.readlines():
        system_prompt += line

# Read the example prompts of items
with open('./netflix_prompts.json', 'r') as f:
    example_prompts = json.load(f)

for i in range(len(example_prompts)):
    example_prompts[i] = example_prompts[i]['prompt']

try:
    dictionary = pickle.load(open("../../data/netflix/itm_prf.pkl", "rb"))
except Exception as e:
    dictionary = {}

# Maximum number of retries
max_retries = 2

for i in range(len(example_prompts)):
    print(f"Evaluating: {i}")
    if i not in dictionary:
        retries = 0
        while retries < max_retries:
            response = get_gpt_response_w_system(example_prompts[i])
            if response is None:
                print(f"Failed to get response for prompt {i}. Retrying...")
                retries += 1
                continue

            try:
                parsed_response = ast.literal_eval(response)
                dictionary.update({i: parsed_response})
                pickle.dump(dictionary, open("../../data/netflix/itm_prf.pkl", "wb"))
                break  # Exit the retry loop if successful
            except (SyntaxError, ValueError) as e:
                print(f"Failed to parse response for prompt {i}: {e}. Retrying...")
                retries += 1

        if retries == max_retries:
            print(f"Max retries reached for prompt {i}. Skipping...")

for i in range(len(example_prompts)):
    if i not in dictionary:
        dictionary.update({i: {'profile': "None", "reasoning": "None"}})
        pickle.dump(dictionary, open("../../data/netflix/itm_prf.pkl", "wb"))

