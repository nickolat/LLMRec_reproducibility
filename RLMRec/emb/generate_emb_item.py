import json
from openai import AzureOpenAI
import os
import numpy as np
import pickle

endpoint = os.getenv("ENDPOINT_URL", " Insert your endpoint here")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", " Insert your key here")

def get_gpt_emb(prompt):

    client = AzureOpenAI(
        api_key=subscription_key,
        api_version="2023-05-15",
        azure_endpoint=endpoint
    )

    embedding = client.embeddings.create(input=str(prompt), model="text-embedding-ada-002").data[0].embedding

    return np.array(embedding)

# Read generated profiles
profiles = pickle.load(open("../../data/netflix/itm_prf.pkl", "rb"))
profiles = {k: profiles[k] for k in sorted(profiles)}

responses = []

for i in range(len(profiles)):
    print("Evaluating profile {}".format(i))
    emb = get_gpt_emb(profiles[i]['profile'])
    responses.append(emb)
    pickle.dump(np.array(responses),open("../../data/netflix/itm_emb_np.pkl", "wb"))

