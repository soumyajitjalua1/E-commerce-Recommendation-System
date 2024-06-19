import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot

def build_collaborative_model(num_users, num_products, embedding_size=50):
    try:
        user_input = Input(shape=(1,), name='user_input')
        user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
        user_vec = Flatten(name='user_flatten')(user_embedding)

        product_input = Input(shape=(1,), name='product_input')
        product_embedding = Embedding(input_dim=num_products, output_dim=embedding_size, name='product_embedding')(product_input)
        product_vec = Flatten(name='product_flatten')(product_embedding)

        dot_product = Dot(axes=1, name='dot_product')([user_vec, product_vec])
        model = Model(inputs=[user_input, product_input], outputs=dot_product)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        print("Collaborative filtering model built successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while building the collaborative filtering model: {e}")
        return None

def train_collaborative_model(model, train_data):
    try:
        user_ids = train_data['User ID'].values
        product_ids = train_data['Product ID'].values
        ratings = train_data['Rating'].values
        
        model.fit([user_ids, product_ids], ratings, epochs=5, batch_size=64, validation_split=0.2)
        print("Model trained successfully.")
        return model
    except KeyError as e:
        print(f"KeyError: The column '{e}' is missing in the training data.")
        return None
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
        return None
