import numpy as np
import json
import os

def generate_synthetic_data(n_samples=1000, n_features=16, class_sep=1.0, random_state=None):
    """
    Generates synthetic data for binary classification.

    Parameters:
    - n_samples: int, total number of samples to generate.
    - n_features: int, the number of features (dimension of each vector).
    - class_sep: float, the separation between the two classes.
    - random_state: int, seed for random number generator.

    Returns:
    - X: np.array, feature matrix of shape (n_samples, n_features).
    - y: np.array, labels of shape (n_samples,).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate data for class 0
    X_class_0 = np.random.randn(n_samples // 2, n_features) - class_sep
    
    # Generate data for class 1
    X_class_1 = np.random.randn(n_samples // 2, n_features) + class_sep
    
    # Stack them together
    X = np.vstack((X_class_0, X_class_1))
    
    # Create labels
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    
    # Shuffle the dataset
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


def save_to_json(X, y, filename='data.json'):
    """
    Saves the feature matrix X and labels y to a JSON file.

    Parameters:
    - X: np.array, feature matrix.
    - y: np.array, labels.
    - filename: str, the name of the file to save the data to.
    """
    data = {
        "features": X.tolist(),  # Convert numpy array to list
        "labels": y.tolist()
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Data successfully saved to {filename}")

n_samples = 1000
X, y = generate_synthetic_data(n_samples=n_samples, n_features=16, class_sep=2.0, random_state=42)


train_sample = n_samples//4*3
batch_mask = np.random.choice(n_samples,train_sample)
anti_batch_mask = np.setdiff1d(np.arange(n_samples),batch_mask)

X_train = X[batch_mask]
y_train = y[batch_mask]

X_eval = X[anti_batch_mask]
y_eval = y[anti_batch_mask]

file_name_train = os.path.join(os.path.dirname(__file__), 'traindata')
file_name_eval = os.path.join(os.path.dirname(__file__), 'evaldata')
save_to_json(X_train, y_train,filename=file_name_train)
save_to_json(X_eval, y_eval,filename=file_name_eval)

# Print the shapes of the generated data
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)