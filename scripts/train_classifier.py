import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Levenshtein import distance as levenshtein_distance
from sklearn.utils import resample
import joblib
import os
import sys

# Add backend to sys.path to import config
# This assumes the script is in 'scripts/' and 'backend/' is a sibling
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from app.core.config import settings # Now you can import settings

def train_model():
    print(f"Loading data from: {settings.PROCESSED_DATA_CSV_PATH}")
    if not os.path.exists(settings.PROCESSED_DATA_CSV_PATH):
        print(f"Error: Training data CSV not found at {settings.PROCESSED_DATA_CSV_PATH}")
        print("Make sure the CSV is in backend/app/assets/ or update config.py")
        return

    df = pd.read_csv(settings.PROCESSED_DATA_CSV_PATH)
    df['expected_phonemes'] = df['expected_phonemes'].astype(str).fillna('')
    df['predicted_phonemes'] = df['predicted_phonemes'].astype(str).fillna('')

    print("Calculating Levenshtein distances...")
    df['distance'] = df.apply(lambda r: levenshtein_distance(r['expected_phonemes'], r['predicted_phonemes']), axis=1)
    df['len_expected'] = df['expected_phonemes'].apply(len)
    df['normalized_distance'] = df.apply(lambda r: r['distance'] / r['len_expected'] if r['len_expected'] > 0 else 0, axis=1)


    if 'is_correct' not in df.columns:
        print("Column 'is_correct' not found. Auto-labeling: normalized_distance <= 0.2 is 'correct' (1).")
        df['is_correct'] = df['normalized_distance'].apply(lambda x: 1 if x <= 0.2 else 0)
    else:
         df['is_correct'] = df['is_correct'].astype(int) # Ensure it's integer

    print(f"Value counts for 'is_correct' before upsampling:\n{df['is_correct'].value_counts()}")

    # Balance classes by upsampling minority
    # Assuming 1 is 'correct' and 0 is 'incorrect'
    # The notebook upsampled 'correct'. Let's make it more general.
    label_counts = df['is_correct'].value_counts()
    minority_class_count = label_counts.min()
    majority_class_count = label_counts.max()

    if minority_class_count == 0 or minority_class_count == majority_class_count : # if one class missing or already balanced
        balanced_df = df
        print("Dataset is already balanced or one class is missing. No upsampling performed.")
    else:
        minority_class_label = label_counts.idxmin()
        majority_class_label = label_counts.idxmax()

        df_minority = df[df['is_correct'] == minority_class_label]
        df_majority = df[df['is_correct'] == majority_class_label]

        df_minority_upsampled = resample(df_minority,
                                         replace=True,    # sample with replacement
                                         n_samples=len(df_majority), # to match majority class
                                         random_state=42) # reproducible results

        balanced_df = pd.concat([df_majority, df_minority_upsampled])
        print(f"Upsampled class '{minority_class_label}' from {len(df_minority)} to {len(df_minority_upsampled)}.")


    print(f"Value counts for 'is_correct' after balancing:\n{balanced_df['is_correct'].value_counts()}")

    features_cols = ['distance', 'len_expected', 'normalized_distance']
    features = balanced_df[features_cols]
    labels = balanced_df['is_correct']

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42,
        stratify=labels if labels.nunique() > 1 and min(labels.value_counts()) >= 2 else None # Stratify if possible
    )

    print("Training Logistic Regression model...")
    model = LogisticRegression(solver='liblinear', random_state=42) # As in notebook
    model.fit(X_train, y_train)

    # Ensure assets directory exists before saving
    os.makedirs(os.path.dirname(settings.PHONEME_CLASSIFIER_PATH), exist_ok=True)
    joblib.dump(model, settings.PHONEME_CLASSIFIER_PATH)
    print(f"Model trained and saved to: {settings.PHONEME_CLASSIFIER_PATH}")

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy on test set: {accuracy:.4f}")

if __name__ == "__main__":
    train_model()