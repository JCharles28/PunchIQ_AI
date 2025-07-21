from mistralai import Mistral


class MistralLLM:
    """
    Classe pour gérer le modèle LLM Mistral
    """
    def __init__(self, client: Mistral, model: str = "mistral-tiny"):
        self.client = client
        self.model = model
    
    def run(self, request: str):
        """
        Exécute une requête sur le modèle LLM
        """
        messages = [
            {
                "role": "user", 
                "content": request
            }
        ]
        chat_response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        return chat_response.choices[0].message.content

