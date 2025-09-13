import streamlit as st
import pandas as pd
from src.spam_detector import SpamDetector
from src.utils import plot_confusion_matrix, display_metrics
import matplotlib.pyplot as plt
import io

# Set page configuration
st.set_page_config(
    page_title="Spam Detection System Using NLP",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .sub-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }

    .highlight {
        background: transparent !important;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        border-left: 5px solid #3498db;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .highlight:hover {
        transform: translateY(-2px);
    }

    .card {
        background: transparent !important;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }

    .result-box {
        padding: 25px;
        border-radius: 12px;
        margin: 25px 0;
        font-weight: 600;
        text-align: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }

    .spam {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #c0392b;
        border: 2px solid #e74c3c;
    }

    .ham {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #27ae60;
        border: 2px solid #2ecc71;
    }

    .metric-card {
        background: none !important;
        background-color: transparent !important;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }

    .footer {
        text-align: center;
        margin-top: 60px;
        color: #7f8c8d;
        font-size: 0.9rem;
        padding: 20px;
        background: transparent !important;
        border-radius: 10px;
        border-top: 3px solid #3498db;
    }

    .sidebar-content {
        padding: 20px 0;
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }

    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e1e8ed;
        padding: 10px;
        transition: border-color 0.3s ease;
        color: white !important;
        background: none !important;
        background-color: transparent !important;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
        background: none !important;
        background-color: transparent !important;
    }

    .stSelectbox>div>div {
        border-radius: 8px;
        border: 2px solid #e1e8ed;
        background: none !important;
        background-color: transparent !important;
    }

    .stRadio>div {
        background: none !important;
        background-color: transparent !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    # Initialize the spam detector
    if 'spam_detector' not in st.session_state:
        st.session_state.spam_detector = SpamDetector()
        st.session_state.model_trained = False
    
    # Header
    st.markdown('<h1 class="main-header">📩 Spam Detection System Using NLP</h1>', unsafe_allow_html=True)

    # Sidebar for navigation with icons
    st.sidebar.markdown("""
    <style>
    .sidebar-content {
        padding: 20px 0;
    }
    .sidebar-icon {
        font-size: 1.2rem;
        margin-right: 8px;
        vertical-align: middle;
    }
    .sidebar-item {
        display: flex;
        align-items: center;
        padding: 8px 0;
        font-weight: 600;
        color: #34495e;
        cursor: pointer;
        transition: color 0.3s ease;
    }
    .sidebar-item:hover {
        color: #667eea;
    }
    </style>
    """, unsafe_allow_html=True)

    nav_items = {
        "Home": "🏠",
        "Data Overview": "📊",
        "Train Model": "⚙️",
        "Test the Filter": "🔍",
        "About": "ℹ️"
    }

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section",
                                   list(nav_items.keys()),
                                   format_func=lambda x: f"{nav_items[x]}  {x}")
    
    # Introduction
    st.markdown("""
    <div class="highlight">
    This project demonstrates an automated system to filter spam SMS and emails using Natural Language Processing (NLP). 
    The system uses TF-IDF for feature extraction and a Naive Bayes classifier for spam detection.
    </div>
    """, unsafe_allow_html=True)
    
    if app_mode == "Home":
        st.markdown("""
        <div class="card">
        <h3>Welcome to the Spam Detection System Using NLP</h3>
        <p>This application helps you automatically classify SMS and emails as spam or legitimate (ham) using
        Natural Language Processing techniques.</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
            <h4>🚀 Features</h4>
            <ul>
            <li>Text preprocessing (tokenization, stopword removal, stemming)</li>
            <li>TF-IDF vectorization for feature extraction</li>
            <li>Naive Bayes classification model</li>
            <li>Real-time spam prediction</li>
            <li>Performance metrics evaluation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
            <h4>📋 How to use</h4>
            <ol>
            <li>Navigate to <strong>Data Overview</strong> to view and upload data</li>
            <li>Go to <strong>Train Model</strong> to train the spam detection model</li>
            <li>Use <strong>Test the Filter</strong> to check if a message is spam or ham</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
        
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
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{len(df)}</h3>
                        <p>Total Samples</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    if 'label' in df.columns:
                        spam_count = df['label'].sum() if df['label'].dtype == 'int' else len(df[df['label'] == 'spam'])
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{spam_count}</h3>
                            <p>Spam Messages</p>
                        </div>
                        """, unsafe_allow_html=True)
                with col3:
                    if 'label' in df.columns:
                        ham_count = len(df) - spam_count
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{ham_count}</h3>
                            <p>Ham Messages</p>
                        </div>
                        """, unsafe_allow_html=True)
                
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

                    # Accuracy metric
                    st.markdown(f"""
                    <div class="metric-card" style="text-align: center; margin-bottom: 30px;">
                        <h2 style="color: #27ae60; margin: 0;">{accuracy:.2%}</h2>
                        <p style="margin: 5px 0 0 0; font-weight: 600;">Model Accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("""
                        <div class="card">
                        <h4>Confusion Matrix</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.write(pd.DataFrame(cm,
                                             index=['Actual Ham', 'Actual Spam'],
                                             columns=['Predicted Ham', 'Predicted Spam']))

                    with col2:
                        st.markdown("""
                        <div class="card">
                        <h4>Classification Report</h4>
                        </div>
                        """, unsafe_allow_html=True)
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

                # Calculate confidence scores
                ham_confidence = probability[0] * 100  # Probability of being ham
                spam_confidence = probability[1] * 100  # Probability of being spam

                # Display results
                st.subheader("Result")
                if prediction == 1:
                    st.markdown(f'<div class="result-box spam">🚨 This message is classified as SPAM</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box ham">✅ This message is classified as LEGITIMATE (HAM)</div>', unsafe_allow_html=True)

                # Show confidence with styled metric cards
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #ff9a9e; color: #c0392b;">
                        <h3>{spam_confidence:.2f}%</h3>
                        <p>Spam Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="background: #a8edea; color: #27ae60;">
                        <h3>{ham_confidence:.2f}%</h3>
                        <p>Ham Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Show processed text
                with st.expander("View processed text"):
                    processed_text = st.session_state.spam_detector.preprocessor.preprocess_text(user_input)
                    st.write(processed_text)

            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    elif app_mode == "About":
        st.markdown('<h2 class="sub-header">About the Project</h2>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="card">
            <h4>📋 Project Overview</h4>
            <p>This project addresses the challenge of manually sorting through large volumes of daily SMS and emails,
            which is time-consuming and inefficient. By utilizing Natural Language Processing (NLP), this system
            automatically filters spam messages with high accuracy.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <h4>🔬 Methodology</h4>
            <ol>
            <li><strong>Text Preprocessing</strong>:
               <ul>
               <li>Tokenization, lowercasing, special character removal</li>
               <li>Stopword removal and stemming</li>
               </ul>
            </li>
            <li><strong>Feature Extraction</strong>:
               <ul>
               <li>TF-IDF (Term Frequency-Inverse Document Frequency) vectorization</li>
               </ul>
            </li>
            <li><strong>Classification</strong>:
               <ul>
               <li>Naive Bayes classifier for spam detection</li>
               </ul>
            </li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
            <h4>🛠️ Technologies Used</h4>
            <ul>
            <li><strong>Python</strong> for backend processing</li>
            <li><strong>NLTK</strong> for natural language processing</li>
            <li><strong>Scikit-learn</strong> for machine learning</li>
            <li><strong>Streamlit</strong> for the web interface</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card">
            <h4>🚀 Potential Enhancements</h4>
            <ul>
            <li>Use larger and more diverse datasets</li>
            <li>Implement deep learning models (RNN, LSTM)</li>
            <li>Add multi-language support</li>
            <li>Create browser extensions for real email filtering</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px; margin-bottom: 10px;">
            <span style="font-size: 1.2rem;">📧</span>
            <span style="font-weight: 600; color: #3498db;">Spam Detection System Using NLP</span>
            <span style="font-size: 1.2rem;">🎓</span>
        </div>
        <p style="margin: 0; font-size: 0.9rem;">College Project | 2025 | 👨‍💻 Author ❤️Bosspandey</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()