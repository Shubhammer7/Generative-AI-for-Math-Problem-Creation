import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained generator model
generator = load_model('models/generator_epoch_150.h5')

# Set the noise dimension and number of problems to generate
noise_dim = 100
num_problems = 10

# Generate random noise
noise = np.random.normal(0, 1, (num_problems, noise_dim))
print("Noise generated for input to generator:")

# Generate problems using the generator model
generated_vectors = generator.predict(noise)
print("Generated vectors from the generator:")
print(generated_vectors)

def vector_to_text(vector):
    text = ''.join([chr(int((char + 1) * 47.5 + 32)) for char in vector if 32 <= (char + 1) * 47.5 + 32 <= 126])
    return text

# Convert the numeric vectors back to text problems
generated_problems = [vector_to_text(vector) for vector in generated_vectors]
print("Converted generated vectors to text problems:")
print(generated_problems)

# Save the generated problems to a file
with open('data/generated_problems.txt', 'w') as f:
    for i, problem in enumerate(generated_problems):
        f.write(f"Generated Problem {i+1}: {problem}\n")

print("Generated problems saved to data/generated_problems.txt")
