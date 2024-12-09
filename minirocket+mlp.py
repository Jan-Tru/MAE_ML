import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sktime.transformations.panel.rocket import MiniRocketMultivariate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# from sktime.datatypes import from_3d_numpy_to_nested

def load_all_data(base_dir, fixed_length=500):
    """
    Load and process all data from the specified base directory.

    Returns:
    - X: numpy array of shape (n_instances, n_channels, fixed_length)
    - y: numpy array of class labels corresponding to the data.
    """
    # Participants P1 to P11
    group_numbers = range(1, 12)
    # Sessions S1 to S4
    session_numbers = range(1, 5)

    # Initialize lists to store data and labels
    X_list = []
    y_list = []

    for group_number in group_numbers:
        for session_number in session_numbers:
            session_folder = os.path.join(base_dir, f"P{group_number}_S{session_number}")

            if not os.path.exists(session_folder):
                print(f"Session folder not found: {session_folder}")
                continue

            # List all CSV files in the folder
            all_files = [f for f in os.listdir(session_folder) if f.endswith('.csv')]

            for file_name in all_files:
                file_path = os.path.join(session_folder, file_name)
                # Load CSV file
                df = pd.read_csv(file_path)

                # Convert header to first row
                header_as_row = pd.DataFrame([list(df.columns)], columns=df.columns)
                df = pd.concat([header_as_row, df], ignore_index=True)

                # Rename columns
                if df.shape[1] == 6:
                    df.columns = [
                        "EMG_submental", "EMG_intercostal", "EMG_diaphragm",
                        "pneumotachometry", "contact_microphone", "class_labels"
                    ]
                else:
                    print(f"Warning: {file_name} does not have 6 columns. Skipping file.")
                    continue

                # Process class labels
                # Convert class_labels column to numeric
                df['class_labels'] = pd.to_numeric(df['class_labels'], errors='coerce')
                df = df.dropna(subset=['class_labels'])
                df['class_labels'] = df['class_labels'].astype(int)

                # Filter out rows with class_labels not in [0,1,2,3,4]
                df = df[df['class_labels'].isin([0, 1, 2, 3, 4])]

                # Combine labels 1 and 2 into 'swallow' class (label 1)
                df['class_labels'] = df['class_labels'].replace(2, 1)

                # Filter out class_label 0 (null)
                df = df[df['class_labels'] != 0]

                # Now, we can extract the data and labels
                # Group by consecutive labels
                df['label_change'] = df['class_labels'].diff().fillna(0).abs()
                df['group_id'] = df['label_change'].cumsum()

                for group_id, group_df in df.groupby('group_id'):
                    label = group_df['class_labels'].iloc[0]
                    # Get the feature columns
                    features = group_df[[
                        "EMG_submental", "EMG_intercostal", "EMG_diaphragm",
                        "pneumotachometry", "contact_microphone"
                    ]]

                    # Each channel time series
                    channels = []
                    for col in features.columns:
                        ts = group_df[col].astype(float).values
                        # Pad or truncate ts to fixed_length
                        if len(ts) < fixed_length:
                            # Pad with zeros at the end
                            ts_padded = np.pad(ts, (0, fixed_length - len(ts)), 'constant')
                        else:
                            # Truncate
                            ts_padded = ts[:fixed_length]
                        channels.append(ts_padded)
                    # Now channels is a list of arrays of length fixed_length
                    # Stack channels to get shape (n_channels, fixed_length)
                    instance = np.vstack(channels)  # shape (n_channels, fixed_length)
                    X_list.append(instance)
                    y_list.append(label)

    # Convert X_list and y_list to numpy arrays
    X = np.array(X_list)  # shape (n_instances, n_channels, fixed_length)
    y = np.array(y_list)

    return X, y


def from_3d_numpy_to_nested(X, column_names=None, cells_as_numpy=False):
    """Convert numpy 3D Panel to nested pandas Panel.

    Convert NumPy ndarray with shape (n_instances, n_columns, n_timepoints)
    into nested pandas DataFrame (with time series as pandas Series in cells)

    Parameters
    ----------
    X : np.ndarray
        3-dimensional Numpy array to convert to nested pandas DataFrame format

    column_names: list-like, default = None
        Optional list of names to use for naming nested DataFrame's columns

    cells_as_numpy : bool, default = False
        If True, then nested cells contain NumPy array
        If False, then nested cells contain pandas Series


    Returns
    -------
    df : pd.DataFrame
    """
    df = pd.DataFrame()
    # n_instances, n_variables, _ = X.shape
    n_instances, n_columns, n_timepoints = X.shape

    container = np.array if cells_as_numpy else pd.Series

    if column_names is None:
        column_names = _make_column_names(n_columns)

    else:
        if len(column_names) != n_columns:
            msg = " ".join(
                [
                    f"Input 3d Numpy array as {n_columns} columns,",
                    f"but only {len(column_names)} names supplied",
                ]
            )
            raise ValueError(msg)

    for j, column in enumerate(column_names):
        df[column] = [container(X[instance, j, :]) for instance in range(n_instances)]
    return df


def _make_column_names(column_count):
    return [f"var_{i}" for i in range(column_count)]


def _get_column_names(X):
    if isinstance(X, pd.DataFrame):
        return X.columns
    else:
        return _make_column_names(X.shape[1])


# Main code
base_dir = '/Users/janga/pycharmProjects/MAE_ML/Processed'

# Load and process the data
X, y = load_all_data(base_dir, fixed_length=500)  # Adjust fixed_length as needed

# Check class distribution
print("Class distribution:")
print(pd.Series(y).value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert numpy arrays to nested pandas DataFrames
X_train_df = from_3d_numpy_to_nested(X_train)
X_test_df = from_3d_numpy_to_nested(X_test)

# Transform the data using MiniRocketMultivariate
transformer = MiniRocketMultivariate()
transformer.fit(X_train_df)
X_train_transform = transformer.transform(X_train_df)
X_test_transform = transformer.transform(X_test_df)

# Train the MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,), activation='relu', solver='adam',
    max_iter=200, random_state=42
)

mlp.fit(X_train_transform, y_train)

# Predict on the test data
y_pred = mlp.predict(X_test_transform)

# Evaluate the classifier
print("Classification report:")
print(classification_report(y_test, y_pred, digits=4))

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Swallow', 'Cough', 'Speech']
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
