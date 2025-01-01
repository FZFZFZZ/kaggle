import train, process
import numpy as np
import pandas as pd

def main():
    file_path = "playground-series-s4e12/train.csv"
    data = process.load_data(file_path)

    #process.analyse(data, file_path)
    processed_data = process.clean_data(data)

    model = train.train_and_evaluate_model(processed_data)

    #prep_submit(model)

def prep_submit(model):
    pass

if __name__ == "__main__":
    main()