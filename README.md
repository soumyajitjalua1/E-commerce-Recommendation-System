# E-commerce Recommendation System

This repository contains the code for an E-commerce Recommendation System built using Streamlit, Pandas, Scikit-learn, and TensorFlow. The system includes collaborative filtering, content-based filtering, and a hybrid recommendation model.

## Project Structure

ecommerce_recommendation/
- ├── app.py     # Main Streamlit app
- ├── data_preprocessing.py 
- ├── collaborative_filtering.py 
- ├── content_based_filtering.py 
- ├── hybrid_recommendation.py 
- ├── requirements.txt 
- ├── data/
   - ├── user_data.csv 
   - ├── products.csv
   - ├── interaction.csv 

### `app.py`

This is the main Streamlit application file. It loads and preprocesses the data, builds and trains the collaborative filtering model, and generates recommendations based on user input.

### `data_preprocessing.py`

This module contains functions for loading and preprocessing the user, product, and interaction data. It standardizes and encodes the necessary features and merges the data for use in the collaborative filtering model.

### `collaborative_filtering.py`

This module defines functions for building and training a collaborative filtering model using TensorFlow. The model uses user and product embeddings to predict user ratings for products.

### `content_based_filtering.py`

This module contains functions for generating content-based recommendations using TF-IDF vectorization and cosine similarity based on product descriptions.

### `hybrid_recommendation.py`

This module combines collaborative filtering and content-based filtering to provide hybrid recommendations. It uses both the trained collaborative filtering model and content similarity scores to generate recommendations.

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ecommerce_recommendation.git

2. Install the required packages:
    ```bash
    pip install -r requirements.txt

3.Ensure the data files (user_data.csv, products.csv, interaction.csv) are placed in the data/ directory.

  ### 4. Run the Streamlit app:
      ```bash
      streamlit run app.py
    
### 5. Data Files
  1. user_data.csv: Contains information about users, including age and location.
  2. products.csv: Contains product details, including descriptions and categories.
  3. interaction.csv: Contains user-product interaction data, such as ratings.

### 6. Usage
  1. Load and preprocess the data by running the Streamlit app.
  2. Build and train the collaborative filtering model.
  3. Enter a user ID and product ID to get recommendations.
  4. View the recommended products based on the hybrid recommendation system.

### 7. App link
    https://e-commerce-recommendation-system-7v6ow6ztxxbhkcxmmd4kzz.streamlit.app/

### 8.  License
  This project is licensed under the MIT License - see the LICENSE file for details.
