# Load the Pandas libraries with alias pd
import pandas as pd

def main():
    # Read data from files
    training_data1 = pd.read_csv("tcdml1920-rec-click-pred--training(1).csv")
    training_data2 = pd.read_csv("tcdml1920-rec-click-pred--training(2).csv")
    training_data = training_data1.append(training_data2)

    df3 = training_data[(training_data['set_clicked'] == 1)]
    # Save dataframe to submission file as csv
    df3.to_csv(r'only_ones.csv')

if __name__ == '__main__':
    main()