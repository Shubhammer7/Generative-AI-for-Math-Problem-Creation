import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Embedding, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pickle

# Load preprocessed data
train_data = pd.read_csv('data/processed/train_cleaned.csv')
test_data = pd.read_csv('data/processed/test_cleaned.csv')

# Load the tokenizer
with open('data/processed/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Convert text to sequences
train_problems = tokenizer.texts_to_sequences(train_data['Problem'])
train_solutions = tokenizer.texts_to_sequences(train_data['Solution'])

# Pad sequences to ensure consistent input length
max_sequence_length = max(max(len(seq) for seq in train_problems), max(len(seq) for seq in train_solutions))
train_problems = np.array([np.pad(seq, (0, max_sequence_length - len(seq)), 'constant') for seq in train_problems])
train_solutions = np.array([np.pad(seq, (0, max_sequence_length - len(seq)), 'constant') for seq in train_solutions])

# GAN parameters
vocab_size = len(tokenizer.word_index) + 1
sequence_length = max_sequence_length
noise_dim = 100

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=noise_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(sequence_length, activation='tanh'))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=sequence_length))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator()

# Define the GAN model
z = Input(shape=(noise_dim,))
problem = generator(z)
discriminator.trainable = False
valid = discriminator(problem)

gan = Model(z, valid)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training the GAN
def train_gan(gan, generator, discriminator, train_data, epochs=10000, batch_size=64, noise_dim=100):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_problems = train_data[idx]

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        gen_problems = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_problems, valid)
        d_loss_fake = discriminator.train_on_batch(gen_problems, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, valid)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")
            if epoch % 1000 == 0:
                save_generated_problems(generator, epoch)

def save_generated_problems(generator, epoch, examples=5):
    noise = np.random.normal(0, 1, (examples, noise_dim))
    gen_problems = generator.predict(noise)
    problems = tokenizer.sequences_to_texts(gen_problems)
    with open(f'data/generated_problems_epoch_{epoch}.txt', 'w') as f:
        for problem in problems:
            f.write("%s\n" % problem)

train_gan(gan, generator, discriminator, train_problems, epochs=10000, batch_size=64, noise_dim=noise_dim)
