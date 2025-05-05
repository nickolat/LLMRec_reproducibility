import tqdm
import argparse
from config import *
import pandas as pd
from characteristics.io.loader import TsvLoader
from characteristics.io.writer import TsvWriter
from characteristics.Dataset import GraphDataset
import networkx

parser = argparse.ArgumentParser(description="Run generate characteristics.")
parser.add_argument('--dataset', type=str, default='amazon_35_MMSSL')
parser.add_argument('--characteristics', type=str, nargs='+', default=ACCEPTED_CHARACTERISTICS)

def compute_characteristics_on_dataset():

    path = DATA_FOLDER+'/{}'.format(input_dataset)+'/new_dataset.tsv'

    # load dataset
    loader = TsvLoader(path)
    dataset = GraphDataset(loader.load())

    d_characteristics = {}
    iterator = tqdm.tqdm(characteristics)
    for characteristic in iterator:
        d_characteristics.update({characteristic: dataset.get_metric(characteristic)})

    return d_characteristics

def compute_characteristics():
    # compute characteristics
    characteristics_dataset = []
    row = compute_characteristics_on_dataset()
    if row is not None:
        characteristics_dataset.append(row)
    return characteristics_dataset


if __name__ == '__main__':

    args = parser.parse_args()

    # find datasets
    input_dataset = args.dataset

    characteristics = args.characteristics
    computed_characteristics = compute_characteristics()

    # store results
    computed_characteristics = pd.DataFrame(computed_characteristics)
    writer = TsvWriter(main_directory=OUTPUT_FOLDER, drop_header=False)
    writer.write(computed_characteristics,
                 file_name=f'characteristics',
                 directory=input_dataset)
