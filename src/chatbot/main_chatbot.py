# Libraries
import joblib
import streamlit as st
import sys
import os
import pandas as pd
import stat

# Configure the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Components
from mistralai import Mistral
from src.chatbot.llm_model import MistralLLM
from src.chatbot.rag import Data, RAG
from src.chatbot.path_manager import get_path_manager
from src.chatbot.api_key_manager import get_api_key_manager

# Initialize managers
path_mgr = get_path_manager()
api_mgr = get_api_key_manager()

# ml functions

async def load_ml_utils(model_path: str, scaler_path: str, label_encoder_path: str):
    """Load a machine learning model from the specified path."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        
        return model, scaler, label_encoder
    except Exception as e:
        st.error(f"❌ Error loading ML model: {str(e)}")
        return None, None, None

# utils functions
def start_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I am Punch IQ AI, your boxing match prediction assistant. 🥊"
        },
        {
            "role": "assistant", 
            "content": "Please provide the details of the fighters, and I will predict the outcome of the match. You can ask me about:\nFighter comparisons\n• Match predictions\n• Boxing statistics\n• Fighter profiles"
        }
    ]

def init_chatbot():
    if "messages" not in st.session_state:
        start_conversation()

def reset_chatbot():
    start_conversation()
    st.rerun()

def init_fighters_info():
    if "fighters_info" not in st.session_state:
        st.session_state.fighters_info = {
            "fighterA": {
                "name": "",
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "kos": 0
            },
            "fighterB": {
                "name": "",
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "kos": 0
            }
        }


# Initialize RAG system (cached)
@st.cache_resource
def initialize_rag_system(api_key, dataset_path, model_choice, embedding_model, chunk_size):
    """Initializes the RAG system (cached to avoid reloading)"""
    try:
        # Initialize components
        mistral_client = Mistral(api_key=api_key)
        llm = MistralLLM(client=mistral_client, model=model_choice)
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            st.error(f"❌ Dataset file not found: {dataset_path}")
            return None, None, False
        
        # Load data
        data = Data(dataset_path)
        
        # Create the RAG system
        rag = RAG(
            data=data, 
            mistral_client=mistral_client, 
            embedding_model=embedding_model, 
            chunk_size=chunk_size
        )
        
        # Use path manager for consistent file paths
        embeddings_path = path_mgr.get_absolute_path('fighters_embeddings')
        vector_db_path = path_mgr.get_absolute_path('faiss_db')
        
        st.info(f"📁 Looking for files in: {path_mgr.get_absolute_path('data_dir')}")

        # Check file permissions and existence
        embeddings_exist = check_file_accessible(embeddings_path)
        vector_db_exist = check_file_accessible(vector_db_path)

        if embeddings_exist and vector_db_exist:
            try:
                # Load existing embeddings and vector database
                rag.load_embeddings(embeddings_path)
                rag.load_vector_database(vector_db_path)
                st.success("✅ RAG system loaded from cache")
            except Exception as load_error:
                st.warning(f"⚠️ Error loading cached files: {str(load_error)}")
                st.info("🔄 Recreating RAG system...")
                return create_new_rag_system(rag, embeddings_path, vector_db_path)
        else:
            # Create embeddings and vector database
            st.info("🔄 First time setup - creating embeddings...")
            return create_new_rag_system(rag, embeddings_path, vector_db_path)
        
        return llm, rag, True
    
    except Exception as e:
        st.error(f"❌ Error initializing RAG: {str(e)}")
        # Display diagnostic information
        path_mgr.display_diagnostic()
        return None, None, False

def check_file_accessible(file_path):
    """Check if file exists and is accessible for reading"""
    try:
        if not os.path.exists(file_path):
            return False
        
        # Check read permissions
        if not os.access(file_path, os.R_OK):
            st.warning(f"⚠️ No read permission for: {file_path}")
            return False
            
        # Try to get file stats
        os.stat(file_path)
        return True
        
    except (OSError, PermissionError) as e:
        st.warning(f"⚠️ File access error for {file_path}: {str(e)}")
        return False

def create_new_rag_system(rag, embeddings_path, vector_db_path):
    """Create new RAG system with proper error handling"""
    try:
        # Ensure directories exist
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
        
        with st.spinner("🔄 Initializing RAG system..."):
            rag.set_chunks()
            rag.chunks_to_embeddings()
            
            # Try to save embeddings
            try:
                rag.save_embeddings(embeddings_path)
                st.success("✅ Embeddings saved successfully")
            except Exception as e:
                st.warning(f"⚠️ Could not save embeddings: {str(e)}")
            
            # Try to save vector database
            try:
                rag.set_vector_database(vector_db_path)
                st.success("✅ Vector database created successfully")
            except Exception as e:
                st.warning(f"⚠️ Could not save vector database: {str(e)}")
                st.info("💡 RAG will work but won't persist between sessions")
        
        st.success("✅ RAG system successfully initialized")
        return rag.llm if hasattr(rag, 'llm') else None, rag, True
        
    except Exception as e:
        st.error(f"❌ Error creating RAG system: {str(e)}")
        return None, None, False

# Function to generate a response with or without tools (RAG, ML models)
def generate_response(prompt, result_ml, llm, rag=None, use_rag=False, nb_chunks=3):

    if use_rag and rag is not None:
        try:
            # Search for relevant chunks
            search_results = rag.search_similar_chunks(prompt, nb_chunks_by_search=nb_chunks)
            retrieved_chunks = search_results['chunks']
            
            # Create context with retrieved data
            context = """
            You are Punch IQ AI, an expert assistant in boxing and fight prediction.
            Use the following information about the fighters to answer the question.
            """
            
            # Build enriched prompt with RAG
            retrieved_data = "\n".join([f"Chunk {i+1}: {chunk[:500]}..." for i, chunk in enumerate(retrieved_chunks)])
            
            constraints = [
                "Use simple and professional language",
                "Be precise and detailed",
                "Base your response on the provided data"
            ]
            
            rag_prompt = rag.get_prompt_template(
                context=context,
                retrieved_data=retrieved_data,
                result_ml=result_ml,  # Pass ML result to RAG
                task=prompt,
                constraints=constraints
            )
            
            print(rag_prompt)  # Debugging output to verify the prompt content
            
            response = llm.run(rag_prompt)
            
            return {
                "response": response,
                "used_rag": True,
                "chunks_used": len(retrieved_chunks),
                "retrieved_chunks": retrieved_chunks[:2]  # Display only the first 2
            }
            
        except Exception as e:
            # Fallback to response without RAG in case of error
            st.warning(f"⚠️ RAG error: {str(e)}. Using standard mode.")
            
    # Standard mode without RAG
    boxing_context = f"""
    You are Punch IQ AI, an expert assistant in boxing and fight prediction.
    Your role is to analyze boxing matches and provide predictions based on:
    - The boxers' statistics
    - Their fight history
    - Their fighting style
    - Their strengths and weaknesses
    
    User's question: {prompt}
    
    Respond in a professional and detailed manner.
    """
    
    response = llm.run(boxing_context)
    
    return {
        "response": response,
        "used_rag": False,
        "chunks_used": 0,
        "retrieved_chunks": []
    }


# Streamlit page configuration
st.set_page_config(
    page_title="Punch IQ AI",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded"
)


with st.sidebar:
    st.header("🔧 Chatbot Settings")
    st.markdown("---")
    
    # API configuration
    saved_key = api_mgr.get_mistral_key()
    default_key = saved_key if saved_key else ""
    
    api_key = st.text_input(
        "🔑 Mistral API Key", 
        key="chatbot_api_key", 
        type="password",
        value=default_key,
        help="Enter your Mistral API key to use the chatbot"
    )
    
    # Option to save the key
    if api_key and api_key != saved_key:
        if st.button("💾 Save API Key", help="Save key securely for future use"):
            if api_mgr.save_api_key("mistral", api_key):
                st.success("✅ API Key saved securely")
                st.rerun()
    
    if api_key:
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ API Key required")
        if saved_key:
            st.info("💡 Using saved API key")
    
    # Display key status in expander
    api_mgr.display_key_status()
    
    st.markdown("---")
    
    # RAG configuration
    st.subheader("🧠 RAG Configuration")
    use_rag = st.checkbox("Enable RAG system",
                          value=True,
                          help="Uses the boxer database for more accurate responses")
    
    if use_rag:
        # Chemins absolus par défaut
        default_dataset_path = path_mgr.get_default_dataset()
        
        if not default_dataset_path:
            st.warning("⚠️ No dataset found in data directory")
            path_mgr.display_diagnostic()
        
        dataset_path = st.text_input(
            "📊 Dataset path", 
            value=default_dataset_path or "Please check data directory",
            help="Path to the CSV file containing boxer data"
        )
        
        # Vérifier si le fichier existe
        if dataset_path and os.path.exists(dataset_path):
            st.success(f"✅ Dataset found: {os.path.basename(dataset_path)}")
        elif dataset_path:
            st.error(f"❌ Dataset not found: {dataset_path}")
            
            # Proposer des alternatives
            csv_files = path_mgr.find_csv_files()
            if csv_files:
                st.info("💡 Available CSV files:")
                for csv_file in csv_files:
                    if st.button(f"Use {csv_file['name']}", key=f"csv_{csv_file['name']}"):
                        st.session_state.selected_dataset = csv_file['path']
                        st.rerun()
        
        # Advanced RAG options
        with st.expander("⚙️ Advanced RAG Settings"):
            chunk_size = st.slider("Chunk size", 512, 4096, 2048, help="Size of text segments")
            nb_chunks = st.slider("Number of chunks to retrieve", 1, 10, 3, help="Number of relevant segments to use")
            embedding_model = st.selectbox(
                "Embedding model",
                ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"],
                help="Model used to create embeddings"
            )
    
    st.markdown("---")
    
    # Model settings
    st.subheader("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Mistral Model",
        [
            "mistral-tiny",
            # "mistral-small",
            # "mistral-medium",
            "open-mistral-7b"
        ],
        index=0,
        help="Choose the Mistral model to use"
    )
    
    # Button to clear chat history
    if st.button("🗑️ Clear history", help="Clears all messages in the conversation"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am Punch IQ AI, your boxing match prediction assistant."
            },
            {
                "role": "assistant", 
                "content": "Please provide the details of the fighters, and I will predict the outcome of the match."
            }
        ]
        st.rerun()

st.title("🥊 Punch IQ AI")
st.caption("🚀 AI solution that predicts the outcome of a boxing match based on the fighters' characteristics.")

# Chatbot interface customization
st.markdown(
    """
    <style>
        * {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stChatMessage {
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stChatMessageUser {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .stChatMessageAssistant {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        .chat-header {
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a pull-down menu for inputs about two distinct fighters
with st.expander("Fighters Information for Prediction", expanded=True):
    # st.header("Fighter Information")
    use_fighters_info = st.checkbox("Enable ML Outcome Prediction",
                        value=False,
                        help="Uses the boxer infos for more precise target prediction")
    # Create two columns for Fighter 1 and Fighter 2
    col1, col2 = st.columns(2)

    # Fighter A Information
    with col1:
        st.subheader("Fighter A")
        fighterA_name = st.text_input("Name", key="fighterA_name")
        fighterA_wins = st.number_input("Wins", min_value=0, key="fighterA_wins")
        fighterA_draws = st.number_input("Draws", min_value=0, key="fighterA_draws")
        fighterA_losses = st.number_input("Losses", min_value=0, key="fighterA_losses")
        fighterA_kos = st.number_input("KOs", min_value=0, key="fighterA_kos")

    # Fighter B Information
    with col2:
        st.subheader("Fighter B")
        fighterB_name = st.text_input("Name", key="fighterB_name")
        fighterB_wins = st.number_input("Wins", min_value=0, key="fighterB_wins")
        fighterB_draws = st.number_input("Draws", min_value=0, key="fighterB_draws")
        fighterB_losses = st.number_input("Losses", min_value=0, key="fighterB_losses")
        fighterB_kos = st.number_input("KOs", min_value=0, key="fighterB_kos")

# Initialize fighters information history
init_fighters_info()

# Initialize message history
init_chatbot()

# Display message history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Handle user input
async def handle_user_input():
    if prompt := st.chat_input("Ask me about boxing..."):
        
        if not api_key:
            st.error("🔑 Please enter your Mistral API key in the sidebar to continue.")
            st.stop()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Initialize variables
        result_ml = None
        model = None
        scaler = None
        label_encoder = None
        
        if not use_fighters_info:
            st.warning("⚠️ ML Outcome Prediction is deactivated. Please enable it for fight predictions.")
            st.stop()
        else:
            # Store fighter information in session state
            st.session_state.fighters_info["fighterA"]["name"] = fighterA_name
            st.session_state.fighters_info["fighterA"]["wins"] = fighterA_wins
            st.session_state.fighters_info["fighterA"]["draws"] = fighterA_draws
            st.session_state.fighters_info["fighterA"]["losses"] = fighterA_losses
            st.session_state.fighters_info["fighterA"]["kos"] = fighterA_kos
            print(st.session_state.fighters_info["fighterA"])
            
            if fighterA_wins < fighterA_kos:
                st.warning("⚠️ Wins cannot be less than KOs. Please check the values for Fighter A.")
                st.stop()

            st.session_state.fighters_info["fighterB"]["name"] = fighterB_name
            st.session_state.fighters_info["fighterB"]["wins"] = fighterB_wins
            st.session_state.fighters_info["fighterB"]["draws"] = fighterB_draws
            st.session_state.fighters_info["fighterB"]["losses"] = fighterB_losses
            st.session_state.fighters_info["fighterB"]["kos"] = fighterB_kos
            print(st.session_state.fighters_info["fighterB"])
            
            if fighterB_wins < fighterB_kos:
                st.warning("⚠️ Wins cannot be less than KOs. Please check the values for Fighter B.")
                st.stop()
            
            try:
                model, scaler, label_encoder = await load_ml_utils(
                    f'model/ML/boxing_model.pkl',
                    f'model/ML/scaler.pkl',
                    f'model/ML/label_encoder.pkl'
                )
            except Exception as e:
                st.error(f"❌ Error loading ML models: {str(e)}")
                st.stop()
                        
            if model is not None and scaler is not None and label_encoder is not None:
                # predict the outcome
                wins_diff = fighterA_wins - fighterB_wins
                losses_diff = fighterA_losses - fighterB_losses
                drawn_diff = fighterA_draws - fighterB_draws
                ko_rate_diff = fighterA_kos - fighterB_kos

                features = pd.DataFrame(
                    [[wins_diff, losses_diff, drawn_diff, ko_rate_diff]],
                    columns=['wins_diff', 'losses_diff', 'drawn_diff', 'ko_rate_diff']
                )
            
                features_scaled = scaler.transform(features)
                pred = model.predict(features_scaled)
                result_ml = label_encoder.inverse_transform(pred)[0]
            else:
                st.error("❌ Error loading the ML model. Please check the model files.")
                st.stop()

        if not st.session_state.fighters_info["fighterA"]["name"] or not st.session_state.fighters_info["fighterB"]["name"]:
            st.warning("⚠️ Please provide names for both fighters.")
            st.stop()

        # Initialize system based on configuration
        with st.chat_message("assistant"):
            try:
                if use_rag and 'dataset_path' in locals() and os.path.exists(dataset_path):
                    # RAG mode enabled
                    llm, rag, success = initialize_rag_system(
                        api_key, dataset_path, model_choice, embedding_model, chunk_size
                    )
                    
                    if success and rag is not None:
                        with st.spinner("🔮 Generating response with RAG..."):

                            if not result_ml:
                                st.warning("⚠️ No ML result available. Using standard response generation.")
                                result_ml = "No prediction available"
                                
                            result = generate_response(prompt, result_ml, llm, rag, use_rag=True, nb_chunks=nb_chunks)

                            # Display response
                            st.write(result["response"])
                            
                            # Display RAG information
                            if result["used_rag"]:
                                with st.expander(f"📚 Sources used ({result['chunks_used']} chunks)"):
                                    for i, chunk in enumerate(result["retrieved_chunks"]):
                                        st.text_area(
                                            f"Source {i+1}",
                                            chunk[:300] + "..." if len(chunk) > 300 else chunk,
                                            height=100,
                                            disabled=True
                                        )
                            
                            response_content = result["response"]
                    else:
                        # Fallback to standard mode if RAG fails
                        st.warning("⚠️ RAG system failed to initialize. Using standard mode.")
                        mistral_client = Mistral(api_key=api_key)
                        llm = MistralLLM(client=mistral_client, model=model_choice)
                        result = generate_response(prompt, result_ml or "No prediction available", llm, use_rag=False)
                        st.write(result["response"])
                        response_content = result["response"]
                        
                else:
                    # Standard mode without RAG
                    with st.spinner("🤖 Generating response..."):
                        mistral_client = Mistral(api_key=api_key)
                        llm = MistralLLM(client=mistral_client, model=model_choice)
                        
                        if not result_ml:
                            st.warning("⚠️ No ML result available. Using standard response generation.")
                            result_ml = "No prediction available"
                        
                        result = generate_response(prompt, result_ml, llm, use_rag=False)
                        st.write(result["response"])
                        
                        if not use_rag:
                            st.info("💡 Standard mode enabled. Enable RAG for more accurate responses.")
                        elif 'dataset_path' in locals() and not os.path.exists(dataset_path):
                            st.warning(f"⚠️ Dataset not found: {dataset_path}")
                        
                        response_content = result["response"]
                
                # Add response to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response_content
                    }
                )
                        
            except Exception as error:
                error_msg = f"❌ Error: {str(error)}"
                st.error(error_msg)
                st.info("💡 Check your API key and configuration.")
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer with information and statistics
import asyncio

async def main():
    await handle_user_input()

asyncio.run(main())
st.markdown("---")
st.markdown("---")

# Session statistics
if "messages" in st.session_state and len(st.session_state.messages) > 2:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💬 Messages", len(st.session_state.messages) - 2)  # -2 for welcome messages
    
    with col2:
        rag_status = "🧠 RAG Enabled" if use_rag else "📝 Standard Mode"
        st.metric("🔧 Mode", rag_status)
    
    with col3:
        st.metric("🤖 Model", model_choice)

st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🥊 Punch IQ AI - Your AI assistant for boxing match predictions<br>
    </div>
    """,
    unsafe_allow_html=True
)