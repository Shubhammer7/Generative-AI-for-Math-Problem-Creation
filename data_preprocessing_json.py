import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

def preprocess_text(text):
    text = text.lower().replace('\n', ' ')
    return text

data_dir = 'data/MATH'
categories = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']

problems = []
solutions = []

for category in categories:
    train_path = os.path.join(data_dir, 'train', category)
    test_path = os.path.join(data_dir, 'test', category)

    for subdir in [train_path, test_path]:
        for filename in os.listdir(subdir):
            if filename.endswith('.json'):
                file_path = os.path.join(subdir, filename)
                with open(file_path, 'r') as file:
                    try:
                        data = json.load(file)
                        problem = data.get('problem', None)
                        solution = data.get('solution', None)
                        if problem and solution:
                            problems.append(preprocess_text(problem))
                            solutions.append(preprocess_text(solution))
                    except json.JSONDecodeError:
                        print(f"Failed to decode JSON from file: {filename}")
                    except Exception as e:
                        print(f"An error occurred: {e}")

if problems and solutions:
    df = pd.DataFrame({'Problem': problems, 'Solution': solutions})
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs('data/processed', exist_ok=True)
    train_data.to_csv('data/processed/train_cleaned.csv', index=False)
    test_data.to_csv('data/processed/test_cleaned.csv', index=False)

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(problems + solutions)

    # Save the tokenizer
    with open('data/processed/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Tokenizer has been created and saved.")

print("Preprocessing completed.")
