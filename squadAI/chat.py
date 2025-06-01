class ChatHistory:
    """
    This class helps to create the chat history based on the user queries and responses received during the conversation

    Methods:
    --------
    add_chat: this method adds each user query and response to the chat history
    chat: returns the entire chat history
    """

    def __init__(self):
        self.history = []

    def add_chat(self, role: str, prompt: str):
        self.history.append({"role": role, "content": prompt})

    def chat(self):
        return self.history
