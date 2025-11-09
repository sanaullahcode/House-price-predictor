import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# Database Simulation Class
class FirebaseDB:
    def _init_(self):
        try:
            self.users = {}
            # Demo users add karte hain
            self.add_user("demo", "demo@example.com", "demo123")
            self.add_user("sana", "sana@example.com", "sana123") 
            self.add_user("hassan", "hassan@example.com", "hassan123")
            self.add_user("shahid", "shahid@example.com", "shahid123")
            print("Database initialized successfully")
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def add_user(self, username, email, password):
        try:
            if username in self.users:
                return False
                
            user_data = {
                'username': username,
                'email': email,
                'password': password,
                'created_at': time.time()
            }
            
            self.users[username] = user_data
            return True
        except Exception as e:
            print(f"Error adding user: {e}")
            return False
    
    def verify_user(self, username, password):
        try:
            if username in self.users:
                user_data = self.users[username]
                if user_data['password'] == password:
                    return True
            return False
        except Exception as e:
            print(f"Error verifying user: {e}")
            return False
    
    def user_exists(self, username):
        try:
            return username in self.users
        except Exception as e:
            print(f"Error checking user: {e}")
            return False

# Machine Learning Model
def create_sample_dataset():
    """Sample house price dataset banate hain"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'area': np.random.normal(1500, 500, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'stories': np.random.randint(1, 3, n_samples),
        'parking': np.random.randint(0, 3, n_samples),
        'location_quality': np.random.randint(1, 6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Realistic price generate karte hain
    base_price = 50000
    price = (
        base_price +
        df['area'] * 100 +
        df['bedrooms'] * 15000 +
        df['bathrooms'] * 10000 +
        df['stories'] * 20000 +
        df['parking'] * 5000 +
        df['location_quality'] * 25000 +
        np.random.normal(0, 10000, n_samples)
    )
    
    df['price'] = price.astype(int)
    return df

def train_house_price_model():
    """House price prediction model train karte hain"""
    st.info("Machine Learning Model Training Shuru...")
    df = create_sample_dataset()
    
    # Features aur target prepare karte hain
    X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'location_quality']]
    y = df['price']
    
    # Data ko split karte hain
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model train karte hain
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions banate hain
    y_pred = model.predict(X_test)
    
    # Metrics calculate karte hain
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.success(f"Model Training Complete - MAE: ${mae:,.2f}, R²: {r2:.4f}")
    
    return model

def predict_price(model, input_data):
    """Trained model se prediction karte hain"""
    try:
        # Input ko DataFrame mein convert karte hain
        input_df = pd.DataFrame([input_data])
        
        # Prediction karte hain
        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def validate_inputs(area, bedrooms, bathrooms, stories, parking, location_quality):
    """User inputs validate karte hain"""
    if area <= 0:
        return "Area must be positive"
    if bedrooms < 1 or bedrooms > 10:
        return "Bedrooms must be between 1 and 10"
    if bathrooms < 1 or bathrooms > 5:
        return "Bathrooms must be between 1 and 5"
    if stories < 1 or stories > 4:
        return "Stories must be between 1 and 4"
    if parking < 0 or parking > 5:
        return "Parking must be between 0 and 5"
    if location_quality < 1 or location_quality > 5:
        return "Location quality must be between 1 and 5"
    return None

# Main Application
def main():
    # Page configuration
    st.set_page_config(
        page_title="House Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize database and model
    if 'db' not in st.session_state:
        st.session_state.db = FirebaseDB()
    
    if 'model' not in st.session_state:
        st.session_state.model = train_house_price_model()
    
    # Session state management
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'page' not in st.session_state:
        st.session_state.page = 'login'

    st.title("🏠 House Price Prediction System")
    st.markdown("---")
    
    # Navigation
    if st.session_state.logged_in:
        st.sidebar.success(f"Welcome, {st.session_state.username}!")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.rerun()
        
        # Main application after login
        show_prediction_interface()
    else:
        # Login/Signup interface
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Login", use_container_width=True):
                st.session_state.page = 'login'
        with col2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state.page = 'signup'
        
        if st.session_state.page == 'login':
            show_login()
        else:
            show_signup()

def show_login():
    st.header("🔐 User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not username or not password:
                st.error("Please fill in all fields")
            elif st.session_state.db.verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid username or password")

def show_signup():
    st.header("📝 User Registration")
    
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if not all([username, email, password, confirm_password]):
                st.error("Please fill in all fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif st.session_state.db.user_exists(username):
                st.error("Username already exists")
            else:
                if st.session_state.db.add_user(username, email, password):
                    st.success("Registration successful! Please login.")
                    st.session_state.page = 'login'
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error("Registration failed. Please try again.")

def show_prediction_interface():
    st.header("🏡 House Price Prediction")
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Property Details")
        area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500, step=100)
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.slider("Bathrooms", min_value=1, max_value=5, value=2)
    
    with col2:
        st.subheader("🏗 Additional Features")
        stories = st.slider("Stories", min_value=1, max_value=4, value=2)
        parking = st.slider("Parking Spaces", min_value=0, max_value=5, value=1)
        location_quality = st.slider("Location Quality (1-5)", min_value=1, max_value=5, value=3,
                                   help="1: Poor location, 5: Excellent location")
    
    # Prediction button
    if st.button("🏠 Predict House Price", use_container_width=True):
        # Validate inputs
        validation_error = validate_inputs(area, bedrooms, bathrooms, stories, parking, location_quality)
        if validation_error:
            st.error(validation_error)
        else:
            # Prepare input data
            input_data = {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'stories': stories,
                'parking': parking,
                'location_quality': location_quality
            }
            
            # Make prediction
            with st.spinner("Predicting house price..."):
                prediction = predict_price(st.session_state.model, input_data)
            
            if prediction is not None:
                # Display result
                st.markdown("---")
                st.success(f"## Predicted House Price: *${prediction:,.2f}*")
                
                # Show feature analysis
                show_feature_analysis(input_data, prediction)

def show_feature_analysis(input_data, prediction):
    st.subheader("📈 Feature Analysis")
    
    # Visualization ke liye DataFrame banate hain
    features_df = pd.DataFrame({
        'Feature': list(input_data.keys()),
        'Value': list(input_data.values())
    })
    
    # Feature importance chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(features_df, x='Feature', y='Value', 
                        title="Input Features Visualization",
                        color='Value', color_continuous_scale='viridis')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Price breakdown estimation
        breakdown_data = {
            'Component': ['Base Structure', 'Area Premium', 'Bedrooms/Bathrooms', 'Location', 'Additional Features'],
            'Amount': [
                prediction * 0.4,
                prediction * 0.25,
                prediction * 0.15,
                prediction * 0.12,
                prediction * 0.08
            ]
        }
        
        breakdown_df = pd.DataFrame(breakdown_data)
        fig_pie = px.pie(breakdown_df, values='Amount', names='Component',
                        title="Price Breakdown Analysis")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Additional insights
    st.subheader("💡 Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Price per Sq Ft", f"${(prediction/input_data['area']):.2f}")
    with col2:
        st.metric("Price per Bedroom", f"${(prediction/input_data['bedrooms']):,.2f}")
    with col3:
        st.metric("Location Premium", f"${(prediction * 0.12):,.2f}")

if _name_ == "_main_":
    main()