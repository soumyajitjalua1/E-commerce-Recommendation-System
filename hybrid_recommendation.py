import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_content_based_recommendations(product_id, product_data):
    try:
        if 'Description' not in product_data.columns:
            raise KeyError("Product data must contain 'Description' column.")

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(product_data['Description'])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        idx = product_data.index[product_data['Product ID'] == product_id]
        if idx.empty:
            raise ValueError(f"Product ID {product_id} not found in product data.")
        
        idx = idx[0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        product_indices = [i[0] for i in sim_scores]

        return product_data.iloc[product_indices], tfidf_matrix, cosine_sim
    
    except Exception as e:
        print(f"An error occurred in content-based recommendations: {e}")
        return None, None, None

def hybrid_recommendation(user_id, product_id, collaborative_model, product_data, top_n=10):
    try:
        user_vector = np.array([user_id] * len(product_data))
        product_vector = np.array(product_data['Product ID'].values)
        
        cf_scores = collaborative_model.predict([user_vector, product_vector]).flatten()
        print(f"Collaborative Filtering Scores: {cf_scores[:10]}")  # Debugging output

        cb_recommendations, tfidf_matrix, cosine_sim = get_content_based_recommendations(product_id, product_data)
        if cb_recommendations is None:
            print("Error: Could not generate content-based recommendations.")
            return None

        idx = product_data.index[product_data['Product ID'] == product_id]
        if idx.empty:
            raise ValueError(f"Product ID {product_id} not found in product data.")
        
        idx = idx[0]
        cb_scores = cosine_sim[idx]
        print(f"Content-Based Scores: {cb_scores[:10]}")  # Debugging output

        combined_scores = cf_scores + cb_scores

        top_indices = combined_scores.argsort()[-top_n:][::-1]
        print(f"Top Indices: {top_indices}")  # Debugging output

        recommendations = product_data.iloc[top_indices]
        recommendations = recommendations.sort_values(by='Product ID')  # Sort by Product ID

        return recommendations
    
    except Exception as e:
        print(f"An error occurred in hybrid recommendation: {e}")
        return None
