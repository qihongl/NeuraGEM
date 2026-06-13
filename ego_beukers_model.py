"""
EGO Model for the Beukers et al. (2024) Sequence Learning Task.

Implements the Episodic Generalization and Optimization (EGO) framework 
(Giallanza et al., 2024, Study 2) adapted for next-state prediction on 
two Markov-chain graphs.

Architecture:
  - RecurrentContextModule (MGRU): state → context representation
  - EMModule (key-value memory): stores (state, context, next_state) tuples

The model learns to produce context representations that disambiguate 
the two latent graphs, enabling context-appropriate retrieval from episodic 
memory.

NOTE: EMModule and RecurrentContextModule are re-implemented here (not imported
from EmergentIntelligentControl/) per user directive to not touch that directory.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# EMModule — Episodic Memory as a differentiable key-value store
# ---------------------------------------------------------------------------

class EMModule(nn.Module):
    """
    Differentiable episodic memory that stores (state_key, context_key, value) 
    tuples and retrieves via cosine-similarity-weighted averaging.
    """
    
    def __init__(self, temperature=0.05, normalize_keys=True, weighted_retrieval=False):
        super().__init__()
        self.state_keys = None
        self.context_keys = None
        self.values = None
        self.encode_context = True
        self.temperature = temperature
        self.normalize_keys = normalize_keys
        self.state_weight = nn.Parameter(torch.zeros(1))
        self.state_weight.requires_grad = weighted_retrieval

    def norm_key(self, key):
        if self.normalize_keys:
            return key / key.norm(dim=-1, keepdim=True)
        return key

    def get_match_weights(self, state, context):
        if not self.encode_context:
            state = torch.cat([state, context], axis=-1)
        state = self.norm_key(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state_sim = torch.einsum('b a, c a -> c b', self.state_keys, state) / self.temperature
        
        if not self.encode_context:
            return state_sim

        context = self.norm_key(context)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        context_sim = torch.einsum('b a, c a -> c b', self.context_keys, context) / self.temperature
        
        w = torch.sigmoid(self.state_weight)
        return w * state_sim + (1 - w) * context_sim

    def forward(self, state, context):
        match_weights = self.get_match_weights(state, context)
        # safe_softmax: subtract epsilon to guard against numerical overflow
        safe_weights = torch.softmax(match_weights, dim=-1) - 1e-6
        return torch.einsum('a b, c a -> c b', self.values, safe_weights)

    def write(self, state_key, context_key, value):
        state_key = self.norm_key(state_key)
        context_key = self.norm_key(context_key)
        if self.state_keys is None:
            self.state_keys = state_key
            self.context_keys = context_key
            self.values = value
        else:
            self.state_keys = torch.cat((self.state_keys, state_key), dim=0)
            self.context_keys = torch.cat((self.context_keys, context_key), dim=0)
            self.values = torch.cat((self.values, value), dim=0)

    def reset(self):
        self.state_keys = None
        self.context_keys = None
        self.values = None


# ---------------------------------------------------------------------------
# RecurrentContextModule — learns context representations via MGRU
# ---------------------------------------------------------------------------

class RecurrentContextModule(nn.Module):
    """
    Minimal Gated Recurrent Unit for learning context representations.
    
    Architecture: h_new = weight * h_prev + (1 - weight) * h_update
    where weight = sigmoid(state_to_hidden_wt(x) + hidden_to_hidden_wt(h_prev))
    and   h_update = tanh(state_to_hidden(x) + hidden_to_hidden(h_prev))
    """
    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()
        self.state_to_hidden = nn.Linear(n_inputs, n_hidden)
        self.hidden_to_hidden = nn.Linear(n_hidden, n_hidden)
        self.state_to_hidden_wt = nn.Linear(n_inputs, n_hidden)
        self.hidden_to_hidden_wt = nn.Linear(n_hidden, n_hidden)
        self.hidden_to_context = nn.Linear(n_hidden, n_outputs)
        self.n_hidden_units = n_hidden
        self.hidden_state = torch.zeros((self.n_hidden_units,), dtype=torch.float)

    def forward(self, x):
        h_prev = self.hidden_state.to(x.device)
        h_update = torch.tanh(self.state_to_hidden(x) + self.hidden_to_hidden(h_prev))
        h_weight = torch.sigmoid(self.state_to_hidden_wt(x) + self.hidden_to_hidden_wt(h_prev))
        h_new = h_weight * h_prev + (1 - h_weight) * h_update
        # Detach hidden state to implement truncated BPTT (1 step back)
        self.hidden_state = h_new.detach().clone()
        return self.hidden_to_context(h_new)

    def reset_hidden(self):
        self.hidden_state = torch.zeros((self.n_hidden_units,), dtype=torch.float)


# ---------------------------------------------------------------------------
# EGOBeukersModel — full model wrapper for the Beukers task
# ---------------------------------------------------------------------------

class EGOBeukersModel(nn.Module):
    """
    EGO model adapted for the Beukers et al. next-state prediction task.
    
    Components:
      - context_module: RecurrentContextModule (state_d → hidden_d → context_d)
      - em_module: EMModule (key-value episodic memory)
    
    The model processes one timestep at a time:
      1. Receive state_t
      2. context_module(state_t) → context_t
      3. em_module.query(state_t, context_t) → predicted_next_state
      4. Observe true state_{t+1} → compute loss
      5. Backpropagate through context_module
      6. Write (state_t, context_t, state_{t+1}) to EM
    """
    
    def __init__(
        self,
        state_d=10,
        hidden_d=10,
        context_d=4,
        temperature=0.2,
        persistence=1.0,
        weighted_retrieval=True,
    ):
        super().__init__()
        self.state_d = state_d
        self.context_d = context_d
        
        self.context_module = RecurrentContextModule(state_d, hidden_d, context_d)
        self.em_module = EMModule(
            temperature=temperature,
            weighted_retrieval=weighted_retrieval,
        )
        
        self._init_weights(persistence)
    
    def _init_weights(self, persistence):
        """Match prep_recurrent_network from EmergentIntelligentControl/experiment3.py."""
        with torch.no_grad():
            cm = self.context_module
            # state_to_hidden: identity-like
            cm.state_to_hidden.weight.copy_(torch.eye(self.state_d, dtype=torch.float))
            cm.state_to_hidden.bias.zero_()
            # hidden_to_hidden: zeros
            cm.hidden_to_hidden.weight.zero_()
            cm.hidden_to_hidden.bias.zero_()
            # state_to_hidden_wt: zero weights, persistence bias
            cm.state_to_hidden_wt.weight.zero_()
            cm.state_to_hidden_wt.bias.copy_(
                torch.ones(len(cm.state_to_hidden_wt.bias), dtype=torch.float) * persistence
            )
            # hidden_to_hidden_wt: zeros
            cm.hidden_to_hidden_wt.weight.zero_()
            cm.hidden_to_hidden_wt.bias.zero_()
            # hidden_to_context: identity if dimensions match, random otherwise
            if self.state_d == self.context_d:
                cm.hidden_to_context.weight.copy_(torch.eye(self.state_d, dtype=torch.float))
            cm.hidden_to_context.bias.zero_()
        
        # Freeze all recurrent weights except hidden_to_context
        for p in cm.parameters():
            p.requires_grad = False
        cm.hidden_to_context.weight.requires_grad = True
    
    def trainable_parameters(self):
        """Return only parameters that should be updated during training."""
        return [p for p in self.context_module.parameters() if p.requires_grad]
    
    def forward_step(self, state, next_state, train=True):
        """
        Single timestep forward pass.
        
        Args:
            state: (1, state_d) one-hot current state
            next_state: (1, state_d) one-hot true next state
            train: if True, return loss and context; if False, return prediction only
        
        Returns:
            If train: (prediction, loss, context)
            If not train: prediction
        """
        # 1. Get context from context module (updates hidden state internally)
        context = self.context_module(state)  # (1, context_d)
        
        # 2. Query episodic memory
        if self.em_module.values is not None and len(self.em_module.values) > 0:
            prediction = self.em_module(state, context)  # (1, state_d)
        else:
            prediction = torch.zeros(1, self.state_d, device=state.device)
        
        if not train:
            return prediction
        
        # 3. Compute loss
        loss = nn.functional.mse_loss(prediction, next_state)
        
        return prediction, loss, context.detach()  # detach context for EM storage
    
    def write_to_memory(self, state, next_state, context):
        """Write the current experience to episodic memory."""
        self.em_module.write(state.detach(), context, next_state.detach())
    
    def reset(self):
        """Reset hidden state and episodic memory for a new seed/run."""
        self.context_module.reset_hidden()
        self.em_module.reset()
