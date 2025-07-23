# **PunchIQ AI**

An AI chatbot that predicts the outcome of boxing matches based on fighters' characteristics. It uses a RAG (Retrieval-Augmented Generation) system to deliver accurate answers from a comprehensive fighter database.

## Requirements

### Create a virtual environment

```bash
python -m venv myenv
```

### Activate the environment
- On `Windows`:
    ```bash
    .\myenv\Scripts\activate
    ```
- On `Linux` or `MacOS`:
    ```bash
    source myenv/bin/activate
    ```

### Install required modules
```bash
pip install -r requirements.txt
```

## Launch Notebooks

1. Select the kernel/environment you created
2. Press 'Run All' or run cells individually

## 🚀 Features

### 1. **Intelligent RAG System**
- 🧠 Semantic search in the boxer database
- 📊 Vector embeddings for precise retrieval
- 🔍 Relevant data extraction for queries

### 2. **Operating Modes**
- **RAG Mode**: Answers enriched with boxer data
- **Standard Mode**: Answers based on model knowledge

### 3. **Flexible Configuration**
- 🔧 Choose Mistral model (tiny, small, medium, 7b)
- ⚙️ Customizable RAG parameters
- 📊 Control chunk size and number of sources

### File Structure

```
PunchIQ_AI/
├── src/chatbot/
│   ├── main.py    # Streamlit interface
│   ├── rag.py             # RAG system
│   └── llm_model.py       # Mistral model
├── data/
│   ├── fighters_cleaned.csv
│   ├── fighters_embeddings.npy (generated)
│   └── faiss_vectorized_db (generated)
├── test/
│   └── test_chatbot.py        # Test script
```

## 🎯 Usage

### 1. **Start the Chatbot**
```bash
streamlit run src/chatbot/main.py
```

### 2. **Initial Configuration**
1. **Mistral API Key**: Enter your key in the sidebar
2. **RAG Activation**: Enable to use the database
3. **Dataset Path**: Check path to `fighters_cleaned.csv`
4. **Model**: Choose based on speed/quality needs

### 3. **Advanced RAG Parameters**
- **Chunk size**: 512-4096 characters (default: 2048)
- **Number of chunks**: 1-10 sources (default: 3)
- **Embedding model**: all-MiniLM-L6-v2 (fast) or L12-v2 (precise)

## 🎯 Usage Examples

### Typical Session
```
User: "Compare Anthony Joshua and Tyson Fury"
Bot: [Detailed analysis with database sources]

User: "Who would win this fight?"
Bot: [Prediction based on retrieved statistics]

User: "Explain why"
Bot: [Technical explanation with data references]
```

## 💬 Supported Question Examples

### Boxer Comparisons
```
"Compare Anthony Joshua and Tyson Fury"
"Who would win between Canelo and GGG?"
"Analyze the strengths and weaknesses of Mike Tyson"
```

### Fight Predictions
```
"Predict the outcome of Joshua vs Wilder"
"How would a Fury vs Usyk fight play out?"
"Who is favored between Crawford and Spence?"
```

### Stats and Profiles
```
"What are Floyd Mayweather's stats?"
"Show me Manny Pacquiao's record"
"Roberto Duran's fighting style"
```

### Technical Questions
```
"Explain different boxing styles"
"What makes a good jab?"
"How do you analyze a boxing match?"
```

## ⚙️ Optimal Configuration

- **Model**: mistral-tiny
- **Chunks**: 1024 characters, 2 sources
<!-- - **Embedding**: all-MiniLM-L6-v2 -->

<!-- ## 🧪 Testing

### Full Test
```bash
python test_chatbot.py
```

### Manual Tests
1. **Basic test**: "Who is Anthony Joshua?"
2. **RAG test**: "Compare two boxers from your dataset"
3. **Prediction test**: "Predict a specific fight" -->

## 🔄 Data Updates

### Add New Boxers
1. Edit `fighters_cleaned.csv`
2. Delete `fighters_embeddings.npy` and `faiss_vectorized_db`
3. Restart the chatbot (embeddings will be recreated)

### Embedding Optimization
1. Adjust chunk size for your dataset
2. Try different embedding models
3. Evaluate answer relevance

## 📈 Best Practices

### Effective Questions
- ✅ Be specific with boxer names
- ✅ Use clear context
- ✅ Ask one question at a time

### Recommended Configuration
- ✅ Always enable RAG for boxer-related questions
- ✅ Use standard mode for general questions
- ✅ Adjust parameters as needed
