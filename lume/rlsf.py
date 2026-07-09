import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import os

DB_WEIGHTS_PATH = "lume_rlsf_weights.pth"

class RLRouterNet(nn.Module):
    def __init__(self, input_dim: int = 3, num_tools: int = 3):
        super(RLRouterNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, num_tools),
            nn.Sigmoid()  # Tool activation probabilities
        )
        
    def forward(self, x):
        return self.fc(x)


class PyTorchRLSFRouter:
    def __init__(self, num_tools: int = 3):
        self.model = RLRouterNet(input_dim=3, num_tools=num_tools)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.05)
        self.num_tools = num_tools
        self.load_weights()

    def load_weights(self):
        if os.path.exists(DB_WEIGHTS_PATH):
            try:
                self.model.load_state_dict(torch.load(DB_WEIGHTS_PATH, weights_only=True))
                print("🧠 Loaded local RLSF policy network weights.")
            except Exception as e:
                print(f"⚠️ Failed to load RLSF weights: {e}")

    def save_weights(self):
        try:
            torch.save(self.model.state_dict(), DB_WEIGHTS_PATH)
        except Exception as e:
            print(f"⚠️ Failed to save RLSF weights: {e}")

    def select_tools(self, intent_logits: List[float]) -> Tuple[List[bool], List[torch.Tensor]]:
        """
        Takes output logits from intent classifier. Samples tool permissions.
        Returns (enabled_tools_boolean_list, saved_log_probs_for_training).
        """
        self.model.eval()
        input_tensor = torch.tensor([intent_logits], dtype=torch.float32)
        
        with torch.no_grad():
            probs = self.model(input_tensor)[0]
            
        enabled = []
        saved_probs = []
        for i in range(self.num_tools):
            prob = probs[i]
            # Sample action (Bernoulli distribution)
            m = torch.distributions.Bernoulli(prob)
            action = m.sample()
            enabled.append(action.item() == 1.0)
            
            # Save probability node for gradient descent later
            saved_probs.append(prob)
            
        return enabled, saved_probs

    def update_policy(self, intent_logits: List[float], actions: List[bool], reward: float):
        """
        Performs a single step reinforcement learning update using policy gradient descent.
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        input_tensor = torch.tensor([intent_logits], dtype=torch.float32)
        probs = self.model(input_tensor)[0]
        
        loss = 0.0
        for i in range(self.num_tools):
            prob = probs[i]
            # Avoid log(0)
            prob = torch.clamp(prob, min=1e-5, max=1-1e-5)
            
            action_taken = 1.0 if actions[i] else 0.0
            # Policy gradient loss: -log_prob(action) * reward
            log_prob = action_taken * torch.log(prob) + (1.0 - action_taken) * torch.log(1.0 - prob)
            loss += -log_prob * reward
            
        loss.backward()
        self.optimizer.step()
        self.save_weights()
        print(f"📈 Updated RLSF policy (reward: {reward}, loss: {loss.item():.4f})")
