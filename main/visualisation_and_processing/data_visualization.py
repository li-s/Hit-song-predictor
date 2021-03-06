import pandas as pd
import matplotlib.pyplot as plt

def visualise():
    raw_df = pd.read_csv("../../data/data.csv")
    print(f"Number of rows in entire dataset: {raw_df.shape[0]}")
    print(f"Number of columns in entire dataset: {raw_df.shape[1]}", end="\n\n")

    for idx, feature_name in enumerate(raw_df.columns):
        print(f"Feature {idx + 1}. {feature_name}")

    raw_df.hist(figsize=(20, 20), bins='auto', color='#0504aa')

    plt.show()

if __name__ == '__main__':
    visualise()