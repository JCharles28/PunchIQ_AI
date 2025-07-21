from mistralai import Mistral
# from langchain.document_loaders import CSVLoader
from langchain_community.document_loaders import CSVLoader
import numpy as np
import faiss
import os
# sentence_transformers is not used in this file, but it is imported in the original code.
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
    
    def get_prompt_template(self, context: str, retrieved_data: str, result_ml: str, task: str, constraints: list):
        """
        Génère un prompt selon le template du notebook
        """
        constraints_str = ", ".join(constraints)
        
        return f"""
        Context information & required data is below.
        ---------------------
        Context: {context}
        Retrieved data: {retrieved_data}

        For a prediction based on the ML model, if a fight outcome is asked, the result is: {result_ml}
        ---------------------
        Given the context information and not prior knowledge, answer here's your task:
        {task}
        
        Constraints: {constraints_str}
        """


# class RAGSystem:

#     def __init__(self, data: Data, llm_model: MistralLLM):
#         # Initialiser les composants
#         self.data = data
#         self.rag = RAG(data, llm_model.client, embedding_model="all-MiniLM-L6-v2", chunk_size=1024)
#         self.llm = llm_model
        
#         # Paramètres par défaut
#         self.models = {
#             "default": "open-mistral-7b",
#             "fast": "mistral-tiny",
#             "latest": "mistral-large-latest"
#         }
    
#     def setup_rag_system(self, save_embeddings: bool = True, embeddings_path: str = "../data/fighters_embeddings.npy", 
#                         vector_db_path: str = "../data/faiss_vectorized_db"):
#         """
#         Configure le système RAG complet
#         """
#         # 1. Créer les chunks
#         chunks = self.rag.set_chunks()
#         print(f"✅ Créé {len(chunks)} chunks")
        
#         # 2. Créer les embeddings
#         embeddings = self.rag.chunks_to_embeddings()
#         print(f"✅ Créé {embeddings.shape[0]} embeddings de dimension {embeddings.shape[1]}")
        
#         # 3. Sauvegarder les embeddings
#         if save_embeddings:
#             self.rag.save_embeddings(embeddings_path)
#             print(f"✅ Embeddings sauvegardés dans {embeddings_path}")
        
#         # 4. Créer la base de données vectorielle
#         self.rag.set_vector_database(vector_db_path)
#         print(f"✅ Base de données vectorielle créée et sauvegardée dans {vector_db_path}")
        
#         return True
    
#     def predict_fight_outcome(self, fighters: list, k: int = 2):
#         # Construire la tâche de prédiction
#         task = f"""
#         Describe in the most detailed way the outcome of this fight between these boxers: {fighters}.
#         Be sure to clearly choose one winner and explain exactly how and why they won, including the method of victory, key moments, and what led to the final result.
#         """
        
#         # Contexte
#         context = f"""
#         Be a expert in boxing since its creation in 1880.
#         You've access to the following prediction:
#         Based on the predicted outcome of boxing fight between the fighters {fighters} is: [TO_BE_DETERMINED]
#         """
        
#         # Contraintes
#         constraints = [
#             "Use a simple language.",
#             "Be concise.",
#             "Have a sport host tone."
#         ]
        
#         # Rechercher les chunks pertinents
#         search_results = self.rag.search_similar_chunks(task, k=k)
#         retrieved_data = search_results['chunks']
        
#         # Créer le prompt final
#         prompt = self.rag.get_prompt_template(context, str(retrieved_data), task, constraints)
        
#         # Générer la réponse
#         response = self.llm.run(prompt)
        
#         return {
#             'prediction': response,
#             'retrieved_chunks': retrieved_data,
#             'distances': search_results['distances']
#         }