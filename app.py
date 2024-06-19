import streamlit as st
import pandas as pd
from data_preprocessing import load_and_preprocess_data
from collaborative_filtering import build_collaborative_model, train_collaborative_model
from content_based_filtering import get_content_based_recommendations
from hybrid_recommendation import hybrid_recommendation

def main():
    st.set_page_config(page_title="E-commerce Recommendation System")
    st.title("E-commerce Recommendation System")

    st.subheader("Data Loading and Preprocessing")
    with st.spinner('Loading data...'):
        user_data, product_data, merged_data, category_encoder = load_and_preprocess_data()

    if user_data is None or product_data is None or merged_data is None or category_encoder is None:
        st.error("Failed to load data. Please check your data files.")
        return

    st.success("Data loaded successfully.")
    st.write("User Data Sample:",user_dat.head())
    st.write("Product Data Sample:", product_data.head())
    st.write("Merged Data Sample:", merged_data.head())

    st.subheader("Building and Training Collaborative Filtering Model")
    num_users = merged_data['User ID'].nunique()
    num_products = merged_data['Product ID'].nunique()
    collaborative_model = build_collaborative_model(num_users, num_products)

    if collaborative_model is None:
        st.error("Failed to build collaborative filtering model.")
        return

    trained_model = train_collaborative_model(collaborative_model, merged_data)

    if trained_model is None:
        st.error("Failed to train collaborative filtering model.")
        return

    st.success("Collaborative filtering model built and trained successfully.")

    st.subheader("Get Recommendations")
    user_id = st.text_input("Enter User ID:")
    product_id = st.text_input("Enter Product ID:")
    top_n = st.number_input("Number of recommendations:", min_value=1, max_value=20, value=10)

    if st.button("Get Recommendations"):
        if user_id == '' or product_id == '':
            st.warning("Please enter both User ID and Product ID.")
        else:
            try:
                user_id = int(user_id)
                product_id = int(product_id)

                st.write(f"Generating recommendations for User ID: {user_id} and Product ID: {product_id}")

                recommendations = hybrid_recommendation(user_id, product_id, trained_model, product_data, top_n)

                if recommendations is None:
                    st.error("Failed to generate recommendations.")
                else:
                    recommendations['Category'] = category_encoder.inverse_transform(recommendations['Category'])
                    st.write("Recommendations:", recommendations[['Product ID', 'Product Name', 'Category', 'Price']])
            except ValueError:
                st.error("Please enter valid User ID and Product ID.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    main()
