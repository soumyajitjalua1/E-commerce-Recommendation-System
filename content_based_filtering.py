from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def get_content_based_recommendations(product_id, product_data):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(product_data['Description'])

        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        if 'Product ID' not in product_data.columns:
            print("Error: 'Product ID' column not found in product_data.")
            return None

        idx = product_data.index[product_data['Product ID'] == product_id][0]

        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]

        product_indices = [i[0] for i in sim_scores]

        return product_data.iloc[product_indices]
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage (assuming product_data is defined somewhere)
# example_product_data = pd.DataFrame({
#     'Product ID': [1, 2, 3, 4, 5],
#     'Description': [
#         'This is a great product.',
#         'Excellent product with many features.',
#         'A good product for everyday use.',
#         'The best product in the market.',
#         'A decent product with good value.'
#     ]
# })

# recommended_products = get_content_based_recommendations(1, example_product_data)
# if recommended_products is not None:
#     print(recommended_products)
# else:
#     print("No recommendations could be generated.")
