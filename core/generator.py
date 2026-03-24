import pandas as pd
import numpy as np
import os


def generate_safety_data(num_samples=10000):
    # core/generator.py lives inside project_root/core, so parent is project root.
    base_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(base_path)
    data_folder = os.path.join(base_path, 'data')

    # Create the data folder if it somehow disappeared
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    np.random.seed(2026)
    data = []

    for _ in range(num_samples):
        label = np.random.choice([0, 1])
        if label == 1:
            # Harder synthetic positives with partial overlap for realism.
            speed = np.random.uniform(10, 42)
            accel = np.random.uniform(10.5, 23)
            jerk = np.random.uniform(3.5, 14)
            risk_score = np.random.beta(6, 2)
            p_dist = np.random.uniform(4, 15)
        else:
            # Hard negatives: occasionally elevated speed/accel but low risk profile.
            speed = np.random.uniform(0, 18)
            accel = np.random.uniform(8.6, 12.8)
            jerk = np.random.uniform(0, 5.5)
            risk_score = np.random.beta(2, 7)
            p_dist = np.random.uniform(0, 8)

        data.append([speed, accel, jerk, risk_score, p_dist, label])

    columns = ['speed', 'accel', 'jerk', 'risk_score', 'police_dist', 'label']
    df = pd.DataFrame(data, columns=columns)

    # Use the absolute path to save the file
    file_path = os.path.join(data_folder, 'safety_dataset.csv')
    df.to_csv(file_path, index=False)
    print(f"✅ Successfully generated {num_samples} samples in: {file_path}")


if __name__ == "__main__":
    generate_safety_data()