import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

# Class definitions
INTENT_LABELS = {
    0: "DATA_SCIENCE",
    1: "FILE_OPERATION",
    2: "GENERAL_CHAT"
}

# Tiny training dataset to train the classifier locally in 0.05s
TRAINING_DATA = [
    # Data Science
    ("analyze csv file and plot regression graph", 0),
    ("train a pytorch neural network on dataset", 0),
    ("calculate standard deviation of columns in data", 0),
    ("run pandas dataframe operations on values", 0),
    ("plot a scatter diagram of results", 0),
    
    # File Operations
    ("create a text file called logs.txt", 1),
    ("write hello world inside test.py", 1),
    ("list files in my current directory", 1),
    ("delete files inside the workspace", 1),
    ("read content of budget.yaml", 1),
    
    # General Chat
    ("tell me a programming joke", 2),
    ("explain what is an agent in simple terms", 2),
    ("hello how are you doing today", 2),
    ("why is sky blue", 2),
    ("what is the capital of France", 2)
]

class IntentClassifierNet(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int):
        super(IntentClassifierNet, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, text_tensor, offsets):
        embedded = self.embedding(text_tensor, offsets)
        return self.fc(embedded)


class PyTorchIntentClassifier:
    def __init__(self):
        # Build vocabulary from training data
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for phrase, _ in TRAINING_DATA:
            for word in self.tokenize(phrase):
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
                    
        self.vocab_size = len(self.vocab)
        self.model = IntentClassifierNet(vocab_size=self.vocab_size, embedding_dim=16, num_classes=3)
        self._train_local_model()

    def tokenize(self, text: str) -> List[str]:
        return text.lower().replace(".", "").replace(",", "").split()

    def _text_to_tensor(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenize(text)
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        if not indices:
            indices = [self.vocab["<UNK>"]]
        return torch.tensor(indices, dtype=torch.long), torch.tensor([0], dtype=torch.long)

    def _train_local_model(self):
        """
        Trains the local PyTorch model on startup in ~50 milliseconds.
        """
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(100):  # 100 epochs is instant for 15 examples
            for phrase, label in TRAINING_DATA:
                optimizer.zero_grad()
                text_tensor, offsets = self._text_to_tensor(phrase)
                output = self.model(text_tensor, offsets)
                loss = criterion(output, torch.tensor([label], dtype=torch.long))
                loss.backward()
                optimizer.step()

    def predict(self, text: str) -> str:
        """
        Predicts the intent of the input query.
        """
        self.model.eval()
        with torch.no_grad():
            text_tensor, offsets = self._text_to_tensor(text)
            output = self.model(text_tensor, offsets)
            predicted_class = output.argmax(1).item()
            return INTENT_LABELS.get(predicted_class, "GENERAL_CHAT")
