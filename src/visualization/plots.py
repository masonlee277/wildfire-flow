# Define a function to plot histograms for features
def plot_feature_histograms(X_train, X_val, num_features=10):
    # Select the first num_features columns
    feature_columns = X_train.columns[:num_features]

    # Determine the number of rows and columns for subplots
    num_rows = math.ceil(num_features / 2)
    num_cols = 2

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(feature_columns):
        sns.histplot(X_train[col], bins=30, kde=True, color='blue', label='Training Data', ax=axes[i], stat='density', alpha=0.5)
        sns.histplot(X_val[col], bins=30, kde=True, color='orange', label='Validation Data', ax=axes[i], stat='density', alpha=0.5)

        axes[i].set_title(f'Feature: {col}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


