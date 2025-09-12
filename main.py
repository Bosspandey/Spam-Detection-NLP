import streamlit as st
import pandas as pd
from src.spam_detector import SpamDetector
from src.utils import plot_confusion_matrix, display_metrics
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Spam Filtering System",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-weight: bold;
        text-align: center;
    }
    .spam {
        background-color: #ffcccc;
        color: #ff0000;
    }
    .ham {
        background-color: #ccffcc;
        color: #008000;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #7f7f7f;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Initialize the spam detector
    if 'spam_detector' not in st.session_state:
        st.session_state.spam_detector = SpamDetector()
        st.session_state.model_trained = False
    
    # Header
    st.markdown('<h1 class="main-header">📩 Spam Filtering System with NLP</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="highlight">
    This project demonstrates an automated system to filter spam SMS and emails using Natural Language Processing (NLP). 
    The system uses TF-IDF for feature extraction and a Naive Bayes classifier for spam detection.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", 
                                   ["Home", "Data Overview", "Train Model", "Test the Filter", "About"])
    
    if app_mode == "Home":
        st.markdown("""
        ## Welcome to the Spam Filtering System
        
        This application helps you automatically classify SMS and emails as spam or legitimate (ham) using
        Natural Language Processing techniques.
        
        ### Features:
        - Text preprocessing (tokenization, stopword removal, stemming)
        - TF-IDF vectorization for feature extraction
        - Naive Bayes classification model
        - Real-time spam prediction
        - Performance metrics evaluation
        
        ### How to use:
        1. Navigate to **Data Overview** to view and upload data
        2. Go to **Train Model** to train the spam detection model
        3. Use **Test the Filter** to check if a message is spam or ham
        """)
        
    elif app_mode == "Data Overview":
        st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
        
        # Data upload section
        st.subheader("Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                df = pd.read_csv(uploaded_file)
                st.write("Data preview:")
                st.dataframe(df.head())
                
                # Basic info about the dataset
                st.subheader("Dataset Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(df))
                with col2:
                    if 'label' in df.columns:
                        spam_count = df['label'].sum() if df['label'].dtype == 'int' else len(df[df['label'] == 'spam'])
                        st.metric("Spam Messages", spam_count)
                with col3:
                    if 'label' in df.columns:
                        ham_count = len(df) - spam_count
                        st.metric("Ham Messages", ham_count)
                
                # Show column information
                st.subheader("Column Information")
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("Please upload a CSV file to proceed")
        
    elif app_mode == "Train Model":
        st.markdown('<h2 class="sub-header">Train the Spam Detection Model</h2>', unsafe_allow_html=True)
        
        # Option to use sample data or upload data
        data_option = st.radio("Choose data source:", 
                              ("Use sample SMS data", "Upload your own data"))
        
        if data_option == "Use sample SMS data":
            # For demonstration, we'll create sample data
            sample_data = {
                'text': [
                    "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
                    "Your package has been delivered. Track your shipment with code 123ABC.",
                    "URGENT: Your bank account needs verification. Click this link to secure your account.",
                    "Meeting reminder: Tomorrow at 10 AM in conference room B.",
                    "Hi, how are you? It's been a while since we last talked.",
                    "You've been selected for our exclusive investment opportunity with high returns.",
                    "Your Netflix subscription is about to expire. Update your payment information now.",
                    "Mom: Can you pick up some milk on your way home?",
                    "FREE entry to our prize draw. Text STOP to unsubscribe.",
                    "Your recent order #45678 has been shipped. Expected delivery: Friday."
                ],
                'label': [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  # 1 for spam, 0 for ham
            }
            df = pd.DataFrame(sample_data)
            st.write("Sample data preview:")
            st.dataframe(df)
            
        else:
            uploaded_file = st.file_uploader("Upload your training data (CSV)", type="csv")
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.write("Data preview:")
                st.dataframe(df.head())
            else:
                st.info("Please upload a CSV file to train the model")
                return
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Train the model
                    accuracy, cm, report, X_test, y_test, y_pred = st.session_state.spam_detector.train_model(df, 'text', 'label')
                    st.session_state.model_trained = True
                    
                    st.success("Model trained successfully!")
                    
                    # Display results
                    st.subheader("Model Performance")
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Confusion Matrix:**")
                        st.write(pd.DataFrame(cm, 
                                             index=['Actual Ham', 'Actual Spam'], 
                                             columns=['Predicted Ham', 'Predicted Spam']))
                    
                    with col2:
                        st.write("**Classification Report:**")
                        st.text(report)
                    
                    # Plot confusion matrix
                    fig = plot_confusion_matrix(y_test, y_pred)
                    st.pyplot(fig)
                    
                    # Option to save the model
                    if st.button("Save Model"):
                        st.session_state.spam_detector.save_model("models/spam_classifier.pkl")
                        st.success("Model saved successfully!")
                        
                except Exception as e:
                    st.error(f"Error training model: {e}")
    
    elif app_mode == "Test the Filter":
        st.markdown('<h2 class="sub-header">Test the Spam Filter</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("Please train the model first in the 'Train Model' section.")
            return
        
        # Input options
        input_method = st.radio("Choose input method:", 
                               ("Type a message", "Use sample messages"))
        
        if input_method == "Type a message":
            user_input = st.text_area("Enter a message to check if it's spam:", 
                                     height=150,
                                     placeholder="Type or paste your SMS/email content here...")
        else:
            sample_messages = [
                "Congratulations! You've won a free iPhone. Click here to claim now!",
                "Your Amazon order #12345 has been shipped and will arrive tomorrow.",
                "Hi John, are we still meeting for lunch tomorrow at 1 PM?",
                "URGENT: Your bank account has been compromised. Verify your details immediately."
            ]
            selected_message = st.selectbox("Select a sample message:", sample_messages)
            user_input = selected_message
        
        if st.button("Check for Spam") and user_input:
            # Make prediction
            try:
                prediction, probability = st.session_state.spam_detector.predict_text(user_input)
                
                # Display results
                st.subheader("Result")
                if prediction == 1:
                    st.markdown(f'<div class="result-box spam">🚨 This message is classified as SPAM</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box ham">✅ This message is classified as LEGITIMATE (HAM)</div>', unsafe_allow_html=True)
                
                # Show confidence
                st.write("**Confidence:**")
                spam_confidence = probability[1] * 100
                ham_confidence = probability[0] * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Spam Confidence", f"{spam_confidence:.2f}%")
                with col2:
                    st.metric("Ham Confidence", f"{ham_confidence:.2f}%")
                
                # Show processed text
                with st.expander("View processed text"):
                    processed_text = st.session_state.spam_detector.preprocessor.preprocess_text(user_input)
                    st.write(processed_text)
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    elif app_mode == "About":
        st.markdown('<h2 class="sub-header">About the Project</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Project Overview
        This project addresses the challenge of manually sorting through large volumes of daily SMS and emails, 
        which is time-consuming and inefficient. By utilizing Natural Language Processing (NLP), this system 
        automatically filters spam messages with high accuracy.
        
        ### Methodology
        1. **Text Preprocessing**: 
           - Tokenization, lowercasing, special character removal
           - Stopword removal and stemming
        2. **Feature Extraction**:
           - TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
        3. **Classification**:
           - Naive Bayes classifier for spam detection
        
        ### Technologies Used
        - Python for backend processing
        - NLTK for natural language processing
        - Scikit-learn for machine learning
        - Streamlit for the web interface
        
        ### Potential Enhancements
        - Use larger and more diverse datasets
        - Implement deep learning models (RNN, LSTM)
        - Add multi-language support
        - Create browser extensions for real email filtering
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <hr>
        <p>Spam Filtering System with NLP | College Project | 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()