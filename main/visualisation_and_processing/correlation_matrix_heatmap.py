import matplotlib.pyplot as plt
from preprocess import preprocess_data
import seaborn as sns

def correlation_matrix():
    df_final = preprocess_data(0)
    ax = plt.subplots(figsize=(20, 20))
    data = df_final.copy()
    corr = data.corr()

    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True, annot=True
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )

    desired_num_features = len(df_final.columns) - 1
    corr['is_in_billboard'] = corr['is_in_billboard'].apply(abs)
    corr = corr.sort_values('is_in_billboard', ascending=False)

    # add one to extract features because 1st feature will be popularity itself
    extracted_features_list = corr['is_in_billboard'].head(desired_num_features + 1).index.values
    print("Number of features (excluding target variable column) extracted:", len(extracted_features_list) - 1)
    print("Features to extract:", extracted_features_list[1:])

    processed_data = data[extracted_features_list]
    plt.show()

if __name__ == "__main__":
    correlation_matrix()