import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data():
    try:
        # Load datasets
        user_data = pd.read_csv(r'C:\Users\soumy\LLM\GenAi Project\recomedation_project\data\user_data.csv')
        product_data = pd.read_csv(r'C:\Users\soumy\LLM\GenAi Project\recomedation_project\data\products.csv')
        interaction_data = pd.read_csv(r'C:\Users\soumy\LLM\GenAi Project\recomedation_project\data\interaction.csv')
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e.strerror} - {e.filename}")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None, None, None, None

    try:
        # Preprocess user data
        user_data['Age'] = StandardScaler().fit_transform(user_data['Age'].values.reshape(-1, 1))
        user_data['Location'] = LabelEncoder().fit_transform(user_data['Location'])
        print("User data preprocessed successfully.")
    except KeyError as e:
        print(f"KeyError: The column '{e}' is missing in the user data.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while preprocessing user data: {e}")
        return None, None, None, None

    try:
        # Preprocess product data
        category_encoder = LabelEncoder()
        product_data['Category'] = category_encoder.fit_transform(product_data['Category'])
        print("Product data preprocessed successfully.")
    except KeyError as e:
        print(f"KeyError: The column '{e}' is missing in the product data.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while preprocessing product data: {e}")
        return None, None, None, None

    try:
        # Handle missing values in interaction data
        interaction_data['Rating'].fillna(0, inplace=True)

        # Merge data for collaborative filtering
        merged_data = pd.merge(interaction_data, user_data, on='User ID')
        merged_data = pd.merge(merged_data, product_data, on='Product ID')
        print("Data merged successfully.")
    except KeyError as e:
        print(f"KeyError: The column '{e}' is missing in the data for merging.")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while merging data: {e}")
        return None, None, None, None

    return user_data, product_data, merged_data, category_encoder
