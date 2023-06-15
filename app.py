from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

app = FastAPI()

#LoadData
with open('./models/word2int.pkl', 'rb') as f:
    word2int = pickle.load(f)
with open('./models/vectors.pkl', 'rb') as f:
    vectors = pickle.load(f)

    
# Assuming you have a target_word_vector representing the input word vector
target_word_vector = vectors[word2int['career']]

# Normalize the target_word_vector
normalized_target = tf.nn.l2_normalize(target_word_vector, axis=0)

# Normalize all vectors
normalized_vectors = tf.nn.l2_normalize(vectors, axis=1)

# Calculate cosine similarity
cos_sim = tf.matmul(normalized_vectors, tf.expand_dims(normalized_target, axis=1))

# Endpoint method
@app.get("/api/searching/{input_word}")
def get_similar_words(input_word: str):
    target_word_vector = vectors[word2int[input_word]]
    normalized_target = tf.nn.l2_normalize(target_word_vector, axis=0)
    normalized_vectors = tf.nn.l2_normalize(vectors, axis=1)
    cos_sim = tf.matmul(normalized_vectors, tf.expand_dims(normalized_target, axis=1))
    cos_sim_values = cos_sim.numpy()
    int2word = {v: k for k, v in word2int.items()}
    k = 100  # Number of similar words to display
    most_similar_indices = np.argsort(-cos_sim_values[:, 0])[:k]
    similar_words = []
    for idx in most_similar_indices:
        if idx in int2word:
            word = int2word[idx]
            similarity = (cos_sim_values[idx, 0] + 1) / 2
            similar_words.append({"Word": word, "Cosine Similarity": similarity})
    return {"similar_words": similar_words}