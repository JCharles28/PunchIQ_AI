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
            messages=messages,
            
            temperature=0.8, # rate (0 to 1) to control the diversity in the output content (randomness)
            
            # rate (0 to 1) that controls the set of words associating with a probability, the model can choose when it's generating the next word in the content
            # → After sorting possible next words by probability, it accumulates their probabilities until it reaches the top_p value (for example, 0.9)
            # if it a low top_p, the model will only choose from the most probable words
            # if it a high top_p, the model will choose from a wider range of words
            top_p=0.95, 
            
            
            max_tokens=1000, # maximum number of tokens to generate in the output content
            
            frequency_penalty=0.0, # rate (0 to 1) that penalizes new tokens based on their existing frequency in the content in the text
            presence_penalty=0.0, # rate (0 to 1) that penalizes new tokens based on whether they appear in the content so far

        )
        return chat_response.choices[0].message.content

