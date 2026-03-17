import pandas as pd
import numpy as np
import os


def generate_safety_data(num_samples=10000):
    # This ensures we find the project root folder correctly
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.path.join(base_path, 'data')

    # Create the data folder if it somehow disappeared
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    np.random.seed(42)
    data = []

    for _ in range(num_samples):
        label = np.random.choice([0, 1])
        if label == 1:
            speed, accel, jerk = np.random.uniform(15, 50), np.random.uniform(12, 25), np.random.uniform(5, 15)
            risk_score, p_dist = np.random.uniform(0.7, 1.0), np.random.uniform(5, 15)
        else:
            speed, accel, jerk = np.random.uniform(0, 5), np.random.uniform(9.0, 10.5), np.random.uniform(0, 2)
            risk_score, p_dist = np.random.uniform(0, 0.3), np.random.uniform(0, 2)

        data.append([speed, accel, jerk, risk_score, p_dist, label])

    columns = ['speed', 'accel', 'jerk', 'risk_score', 'police_dist', 'label']
    df = pd.DataFrame(data, columns=columns)

    # Use the absolute path to save the file
    file_path = os.path.join(data_folder, 'safety_dataset.csv')
    df.to_csv(file_path, index=False)
    print(f"✅ Successfully generated {num_samples} samples in: {file_path}")


if __name__ == "__main__":
    generate_safety_data()