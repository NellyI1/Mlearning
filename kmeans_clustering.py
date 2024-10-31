# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.cluster import KMeans  # K-Means algorithm for clustering
import matplotlib.pyplot as plt  # For plotting the results
from sklearn.preprocessing import StandardScaler  # To standardize the data
import os  # To interact with the file system

def load_data(directory_path):
    """
    Load CSV files from the specified directory into a single DataFrame.

    Parameters:
        directory_path (str): The path to the directory containing CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame containing all loaded data.
    """
    data_frames = []  # List to hold DataFrames for each CSV file
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):  # Check if the file is a CSV
            df = pd.read_csv(os.path.join(directory_path, filename))  # Load the CSV file
            data_frames.append(df)  # Add DataFrame to the list
            print(f"Loaded data from {filename} successfully.")  # Confirmation message
    return pd.concat(data_frames, ignore_index=True)  # Combine all DataFrames into one

def preprocess_data(data):
    """
    Preprocess the data by filling NaN values and standardizing numeric columns.

    Parameters:
        data (pd.DataFrame): The DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Keep only numeric columns
    print("Numeric Data Types:\n", numeric_data.dtypes)  # Show data types of numeric columns
    numeric_data.fillna(numeric_data.mean(), inplace=True)  # Fill NaNs with the mean of each column
    print("NaN values filled with mean.")  # Confirmation message

    # Check for any remaining NaN values
    if numeric_data.isnull().values.any():
        print("Warning: There are still NaN values in the numeric data.")
        print(numeric_data.isnull().sum())  # Show remaining NaN values
    else:
        print("No remaining NaN values in the numeric data.")  # Confirmation message

    # Standardize the data
    scaler = StandardScaler()  # Create a StandardScaler object
    data_scaled = scaler.fit_transform(numeric_data)  # Fit and transform the numeric data
    print("Data has been standardized.")  # Confirmation message
    return data_scaled  # Return the scaled data

def apply_kmeans(data_scaled, n_clusters=3):
    """
    Apply K-Means clustering to the scaled data.

    Parameters:
        data_scaled (np.array): The standardized data to cluster.
        n_clusters (int): The number of clusters to form.

    Returns:
        np.array: The labels assigned to each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Initialize K-Means with specified clusters
    kmeans.fit(data_scaled)  # Fit K-Means to the scaled data
    return kmeans.labels_  # Return the cluster labels

# Main script execution
if __name__ == "__main__":
    # Step 1: Load data
    directory_path = '/Users/ifeomaigbokwe/Desktop/dataset_csv/customer_segmentation_files/'  # Path to file directory
    data = load_data(directory_path)  # Load the data from the specified directory
    print("Combined Data Preview:\n", data.head())  # Preview the combined data

    # Step 2: Preprocess the data
    data_scaled = preprocess_data(data)  # Preprocess the data and scale it

    # Step 3: Apply K-Means clustering
    labels = apply_kmeans(data_scaled, n_clusters=3)  # Apply K-Means clustering

    # Step 4: Add Cluster Labels to Data and Save the Result
    data['Cluster'] = labels  # Append the cluster labels to the original data
    output_file_path = os.path.join(directory_path, 'clustered_data.csv')  # Specify output path for the clustered data
    data.to_csv(output_file_path, index=False)  # Save clustered data to a new CSV file
    print(f"Clustered data saved as '{output_file_path}'.")  # Confirmation message

    # Step 5: Visualize the Clusters
    # For simplicity, we'll plot the first two features (you can adjust as needed)
    plt.figure(figsize=(8, 6))  # Set the size of the plot
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', marker='o')  # Scatter plot of clusters
    plt.title("K-Means Clustering Visualization")  # Title of the plot
    plt.xlabel("Feature 1 (Standardized)")  # Label for the x-axis
    plt.ylabel("Feature 2 (Standardized)")  # Label for the y-axis
    plt.colorbar(label="Cluster")  # Show color bar
    plt.show()  # Display the plot