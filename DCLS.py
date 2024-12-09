import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
                df['class_labels'] = pd.to_numeric(df['class_labels'], errors='coerce')
                df = df.dropna(subset=['class_labels'])
                df['class_labels'] = df['class_labels'].astype(int)

                # Filter out rows with class_labels not in [0,1,2,3,4]
                df = df[df['class_labels'].isin([0, 1, 2, 3, 4])]

                # Combine labels 1 and 2 into 'swallow' class (label 1)
                df['class_labels'] = df['class_labels'].replace(2, 1)

                # Filter out class_label 0 (null)
                df = df[df['class_labels'] != 0]


                # Group by consecutive labels
                df['label_change'] = df['class_labels'].diff().fillna(0).abs()
                df['group_id'] = df['label_change'].cumsum()

                for group_id, group_df in df.groupby('group_id'):
                    label = group_df['class_labels'].iloc[0]
                    features = group_df[[
                        "EMG_submental", "EMG_intercostal", "EMG_diaphragm",
                        "pneumotachometry", "contact_microphone"
                    ]]

                    channels = []
                    for col in features.columns:
                        ts = group_df[col].astype(float).values
                        ts_padded = np.pad(ts, (0, max(0, fixed_length - len(ts))), 'constant')[:fixed_length]
                        channels.append(ts_padded)
                    instance = np.vstack(channels)  # Shape (n_channels, fixed_length)
                    X_list.append(instance)
                    y_list.append(label)

    X = np.array(X_list)  # Shape (n_instances, n_channels, fixed_length)
    y = np.array(y_list)
    return X, y


# Load the dataset
base_dir = '/Users/janga/pycharmProjects/MAE_ML/Processed'
fixed_length = 500
X, y = load_all_data(base_dir, fixed_length=fixed_length)


# Preprocess the data
def preprocess_data(X, y):
    X = np.transpose(X, (0, 2, 1))  # Transpose to shape (n_instances, fixed_length, n_channels)
    X = X / np.max(np.abs(X), axis=(1, 2), keepdims=True)  # Normalize
    X, y = shuffle(X, y, random_state=42)  # Shuffle
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # Encode labels
    return X, y_encoded, le


X, y, label_encoder = preprocess_data(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model parameters
num_f1, num_f2 = 64, 128  # Feature maps
k_size = 3  # Kernel size
num_units1, num_units2 = 128, 64  # LSTM units
num_units = 128  # Fully connected layer
num_classes = len(np.unique(y))  # Number of classes


# Define the model
def create_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=num_f1, kernel_size=k_size, activation='relu', padding='same')(inputs)
    x = layers.Conv1D(filters=num_f2, kernel_size=k_size, activation='relu', padding='same')(x)
    x = layers.LSTM(num_units1, return_sequences=True)(x)
    x = layers.LSTM(num_units2)(x)
    x = layers.Dense(num_units, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)


# Compile and train the model
model = create_model(input_shape=(fixed_length, X.shape[2]), num_classes=num_classes)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
y_pred = model.predict(X_test).argmax(axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str), zero_division=1))


# Example usage for predictions
def predict_activity_class(X_hat):
    return model.predict(X_hat).argmax(axis=1)


X_hat = X_test[:10]
predictions = predict_activity_class(X_hat)
print("Predictions:", predictions)


# Evaluate the model
y_pred = model.predict(X_test).argmax(axis=1)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = label_encoder.classes_

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)

# Annotate the Confusion Matrix
thresh = conf_matrix.max() / 2
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], "d"),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()