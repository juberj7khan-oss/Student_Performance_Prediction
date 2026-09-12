import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/student_data.csv")

# Display the dataset
print(data)

# Display basic information
print("\nDataset Information:")
print(data.info())

# Display statistical summary
print("\nStatistical Summary:")
print(data.describe())