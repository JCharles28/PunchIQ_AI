from mistralai import Mistral
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
import numpy as np
import faiss, os
from sentence_transformers import SentenceTransformer
from llm_model import MistralLLM

class Data:
    def __init__(self, path_source):
        self.path_source = path_source
        self.documents = CSVLoader(file_path=self.path_source).load()
        
    def get_docs(self):
        return self.documents
    
    def documents_to_text(self):
        """
        Convertit les documents en texte brut.
        """
        return [doc.page_content for doc in self.documents]
    
    def save_documents(self, path: str, format: str):
        """
        Enregistre les documents dans un fichier CSV.
        """
        if format != 'csv' and format != 'txt':
            raise ValueError("Format non supporté. Utilisez 'csv' ou 'txt'.")
        
        with open(path, 'w', encoding='utf-8') as f:
            for doc in self.documents:
                f.write(f"{doc.page_content}\n")

class RAG:
    def __init__(self, 
                    data: Data,
                        mistral_client: Mistral,
                            embedding_model: str = "all-MiniLM-L6-v2",
                                chunk_size: int = 2048):
        
        self.chunk_size = chunk_size
        self.data = data
        self.embedding_model = embedding_model
        self.client = mistral_client
        
        self.chunks = None
        self.embeddings = None
        self.index = None

    def set_chunks(self):
        """
        Divise le texte en chunks selon le notebook (caractère par caractère)
        """
        # Convertir les documents en texte complet
        text_list = self.data.documents_to_text()
        full_text = "\n".join(text_list)
        
        chunks = [
            full_text[i:i + self.chunk_size]
            for i in range(0, len(full_text), self.chunk_size)
        ]
        
        self.chunks = chunks
        return chunks

    def get_text_embedding(self, input_text):
        """
        Obtient l'embedding d'un texte via SentenceTransformer
        """
        model = SentenceTransformer(self.embedding_model)
        embedding = model.encode(input_text)
        return embedding
    
    def chunks_to_embeddings(self):
        """
        Convertit tous les chunks en embeddings
        """
        if self.chunks is None:
            self.set_chunks()
        
        if not self.chunks:
            raise ValueError("Aucun chunks disponibles pour procéder à l'embedding.")
        
        embeddings = np.array([self.get_text_embedding(chunk) for chunk in self.chunks])
        self.embeddings = embeddings
        return embeddings
    
    def save_embeddings(self, path: str, format: str = 'npy'):
        """
        Sauvegarde les embeddings au format spécifié
        """
        if format not in ['npy', 'txt']:
            raise ValueError("Format non supporté. Utilisez 'npy' ou 'txt'.")
        
        if self.embeddings is None:
            self.chunks_to_embeddings()
        
        if format == 'npy':
            np.save(path, self.embeddings)
        elif format == 'txt':
            np.savetxt(path, self.embeddings)
    
    def load_embeddings(self, path: str, format: str = 'npy'):
        """
        Charge les embeddings depuis un fichier
        """
        if format not in ['npy', 'txt']:
            raise ValueError("Format non supporté. Utilisez 'npy' ou 'txt'.")
        
        if format == 'npy':
            self.embeddings = np.load(path)
        elif format == 'txt':
            self.embeddings = np.loadtxt(path)
        
        return self.embeddings
    
    def set_vector_database(self, save_path: str = None):
        """
        Crée une base de données vectorielle FAISS
        """
        if self.embeddings is None:
            self.chunks_to_embeddings()
        
        # Créer l'index FAISS
        d = self.embeddings.shape[1]  # dimension des embeddings
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        
        # Sauvegarder si un chemin est spécifié
        if save_path:
            faiss.write_index(self.index, save_path)
        
        return self.index
    
    def load_vector_database(self, path: str):
        """
        Charge une base de données vectorielle FAISS
        """
        self.index = faiss.read_index(path)
        return self.index
    
    def search_similar_chunks(self, query: str, nb_chunks_by_search: int = 3):
        """
        Recherche les chunks les plus similaires à une requête
        """
        if self.index is None:
            raise ValueError("Base de données vectorielle non initialisée. Utilisez set_vector_database() ou load_vector_database()")
        
        if self.chunks is None:
            raise ValueError("Chunks non initialisés. Utilisez set_chunks()")
        
        # Créer l'embedding de la requête
        query_embedding = np.array([self.get_text_embedding(query)])
        
        # Rechercher les k chunks les plus similaires
        distances, indices = self.index.search(query_embedding, nb_chunks_by_search)
        
        # Récupérer les chunks correspondants
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        
        return {
            'chunks': retrieved_chunks,
            'distances': distances[0],
            'indices': indices[0]
        }
        
    def get_prompt_template(self, context: str,
                                retrieved_data: str,
                                task: str,
                                constraints: list):
        """
        Génère un prompt amélioré pour le modèle Mistral AI.
        """
        constraints_str = "\n- ".join(constraints) if constraints else "None"
        
        
        prompt = f"""
        **Instruction:** You are an expert boxing analyst and predictor, with a deep understanding of fight dynamics, fighter statistics, and the influence of machine learning predictions. Your primary goal is to provide a comprehensive and accurate analysis based *solely* on the provided information.
                            **Context Information:**
                            This section provides general background and relevant data for your analysis.
                            ---------------------
                            **General Context:** {context}
                            **Retrieved Data:** {retrieved_data}
                            ---------------------
                            Task: {task}
                            ---------------------
                            **Constraints:**
                            Adhere strictly to these constraints during your analysis and response generation:
                            - {constraints_str}

                            **Response Format:**
                            Provide your analysis in a clear, structured, and unbiased manner. Ensure all relevant points are covered as per the task.
                            """

        return prompt