"""
Continual Learning Module with Pseudo-Metaplasticity
Implements PR #6 features: EWC, Adaptive Synaptic Gates, Path Integral Importance

This module enables BDH to learn new tasks without catastrophic forgetting.
Key mechanisms:
1. Elastic Weight Consolidation (EWC) - protects important weights
2. Adaptive Synaptic Gates - regulates plasticity at neuron level
3. Online Importance Estimation - tracks weight significance during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
from copy import deepcopy


class FisherInformationEstimator:
    """
    Estimates Fisher Information for EWC.
    Fisher Information measures how important a parameter is for the current task.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
    
    def compute_fisher(
        self,
        dataloader,
        num_samples: int = 1000,
        empirical: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix diagonal.
        
        Args:
            dataloader: DataLoader with training data
            num_samples: Number of samples to use for estimation
            empirical: If True, use empirical Fisher (gradients from data)
        
        Returns:
            Dictionary mapping parameter names to Fisher information tensors
        """
        fisher = {}
        
        # Initialize Fisher info
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        self.model.eval()
        samples_processed = 0
        
        for batch in dataloader:
            if samples_processed >= num_samples:
                break
            
            # Move batch to device
            if isinstance(batch, dict):
                inputs = batch.get('input_ids', batch.get('narrative'))
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
            else:
                inputs = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            
            self.model.zero_grad()
            
            # Forward pass
            try:
                outputs = self.model(inputs, output_hidden_states=True)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use hidden state for representation-based Fisher
                    logits = outputs.last_hidden_state.mean(dim=1)
                else:
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                
                # Compute log-likelihood gradient
                if empirical:
                    # Use actual outputs as targets (empirical Fisher)
                    if logits.dim() > 2:
                        logits = logits.view(-1, logits.size(-1))
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    sampled = torch.multinomial(probs, 1).squeeze(-1)
                    loss = F.nll_loss(log_probs, sampled)
                else:
                    # Use representation norm as proxy objective
                    loss = logits.pow(2).mean()
                
                loss.backward()
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data.pow(2)
                
                samples_processed += inputs.size(0)
            except Exception as e:
                # Skip problematic batches
                continue
        
        # Average Fisher information
        num_samples_actual = max(samples_processed, 1)
        for name in fisher:
            fisher[name] /= num_samples_actual
        
        self.fisher_info = fisher
        
        # Store optimal parameters
        self.optimal_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        return fisher
    
    def get_ewc_loss(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            lambda_ewc: Regularization strength
            
        Returns:
            EWC loss tensor
        """
        if not self.fisher_info or not self.optimal_params:
            return torch.tensor(0.0)
        
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                # Penalize deviation from optimal params weighted by Fisher info
                loss += (fisher * (param - optimal).pow(2)).sum()
        
        return lambda_ewc * loss / 2.0


class AdaptiveSynapticGate(nn.Module):
    """
    Adaptive gates that regulate plasticity at the neuron level.
    Inspired by biological metaplasticity where synapses become more or less
    plastic based on their recent activity history.
    """
    
    def __init__(self, hidden_dim: int, num_gates: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gates = num_gates
        
        # Gate parameters - learnable
        self.gate_weights = nn.Parameter(torch.zeros(num_gates, hidden_dim))
        self.gate_bias = nn.Parameter(torch.zeros(num_gates, 1))
        
        # Activity history (running statistics)
        self.register_buffer('activity_ema', torch.zeros(hidden_dim))
        self.register_buffer('activity_var', torch.ones(hidden_dim))
        self.momentum = 0.1
        
        # Initialize gates to be mostly open
        nn.init.constant_(self.gate_bias, 2.0)  # sigmoid(2) ≈ 0.88
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply adaptive gating to input.
        
        Args:
            x: Input tensor [batch, ..., hidden_dim]
            
        Returns:
            Tuple of (gated output, gate values)
        """
        # Compute current activity
        if self.training:
            activity = x.detach().abs().mean(dim=tuple(range(x.dim() - 1)))
            # Update running statistics
            self.activity_ema = (1 - self.momentum) * self.activity_ema + self.momentum * activity
            self.activity_var = (1 - self.momentum) * self.activity_var + self.momentum * (activity - self.activity_ema).pow(2)
        
        # Compute normalized activity
        normalized_activity = (x.abs().mean(dim=-1, keepdim=True) - self.activity_ema) / (self.activity_var.sqrt() + 1e-8)
        
        # Compute gate values using learned weights and activity
        # Gates reduce plasticity for stable features, increase for novel ones
        gate_logits = torch.einsum('...d,gd->...g', x, self.gate_weights) + self.gate_bias.transpose(0, 1)
        
        # Activity-modulated gating
        gate_values = torch.sigmoid(gate_logits - normalized_activity)
        
        # Average gates if multiple
        if self.num_gates > 1:
            gate_values = gate_values.mean(dim=-1, keepdim=True)
        
        return x * gate_values, gate_values.squeeze(-1)


class OnlineImportanceEstimator:
    """
    Path Integral based online importance estimation.
    Tracks weight significance during training without requiring a separate Fisher computation pass.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.importance: Dict[str, torch.Tensor] = {}
        self.param_cache: Dict[str, torch.Tensor] = {}
        
        # Initialize
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.importance[name] = torch.zeros_like(param.data)
                self.param_cache[name] = param.data.clone()
    
    def update(self, learning_rate: float = 1e-4):
        """
        Update importance estimates based on parameter changes and gradients.
        Uses path integral approximation: importance ≈ |∇L * Δθ|
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Path integral: accumulate gradient * parameter change
                param_change = param.data - self.param_cache[name]
                importance_update = (param.grad.data.abs() * param_change.abs()).clamp(min=0)
                
                # Exponential moving average of importance
                self.importance[name] = 0.9 * self.importance[name] + 0.1 * importance_update
                
                # Update cache
                self.param_cache[name] = param.data.clone()
    
    def get_importance_loss(self, optimal_params: Dict[str, torch.Tensor], lambda_pi: float = 100.0) -> torch.Tensor:
        """
        Compute path integral regularization loss.
        """
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.importance and name in optimal_params:
                imp = self.importance[name]
                optimal = optimal_params[name]
                loss += (imp * (param - optimal).pow(2)).sum()
        
        return lambda_pi * loss / 2.0


class ContinualLearningWrapper(nn.Module):
    """
    Wraps a BDH model with continual learning capabilities.
    Combines EWC, Synaptic Gates, and Online Importance for robust lifelong learning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        ewc_lambda: float = 1000.0,
        use_synaptic_gates: bool = True,
        use_online_importance: bool = True
    ):
        super().__init__()
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        
        # Initialize components
        self.fisher_estimator = FisherInformationEstimator(model, device)
        
        if use_online_importance:
            self.importance_estimator = OnlineImportanceEstimator(model)
        else:
            self.importance_estimator = None
        
        # Add synaptic gates to hidden dimensions
        if use_synaptic_gates and hasattr(model, 'config'):
            hidden_dim = getattr(model.config, 'hidden_size', 512)
            self.synaptic_gate = AdaptiveSynapticGate(hidden_dim)
        else:
            self.synaptic_gate = None
        
        # Task counter
        self.num_tasks = 0
        self.task_fisher_info: List[Dict[str, torch.Tensor]] = []
        self.task_optimal_params: List[Dict[str, torch.Tensor]] = []
    
    def forward(self, *args, **kwargs):
        """Forward pass through wrapped model."""
        outputs = self.model(*args, **kwargs)
        
        # Apply synaptic gates to hidden states if available
        if self.synaptic_gate is not None and hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            gated_hidden, gate_values = self.synaptic_gate(outputs.hidden_states[-1])
            # Store gate values for analysis
            outputs.gate_values = gate_values
        
        return outputs
    
    def consolidate_task(self, dataloader, num_samples: int = 1000):
        """
        Called after training on a task to consolidate memory.
        Computes Fisher information and stores optimal parameters.
        """
        # Compute Fisher for this task
        fisher = self.fisher_estimator.compute_fisher(dataloader, num_samples)
        
        # Store for multi-task EWC
        self.task_fisher_info.append(deepcopy(fisher))
        self.task_optimal_params.append(deepcopy(self.fisher_estimator.optimal_params))
        
        self.num_tasks += 1
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get combined regularization loss from all continual learning components.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Multi-task EWC loss
        if self.task_fisher_info:
            for task_idx, (fisher, optimal) in enumerate(zip(self.task_fisher_info, self.task_optimal_params)):
                for name, param in self.model.named_parameters():
                    if name in fisher and name in optimal:
                        f = fisher[name].to(self.device)
                        o = optimal[name].to(self.device)
                        total_loss += (f * (param - o).pow(2)).sum()
            
            total_loss = self.ewc_lambda * total_loss / (2.0 * max(len(self.task_fisher_info), 1))
        
        # Online importance loss (if enabled)
        if self.importance_estimator is not None and self.task_optimal_params:
            total_loss += self.importance_estimator.get_importance_loss(
                self.task_optimal_params[-1],
                lambda_pi=self.ewc_lambda / 10
            )
        
        return total_loss
    
    def training_step(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Augment training loss with continual learning regularization.
        """
        # Add regularization
        reg_loss = self.get_regularization_loss()
        total_loss = loss + reg_loss
        
        return total_loss
    
    def on_optimizer_step(self, learning_rate: float = 1e-4):
        """
        Called after optimizer.step() to update online importance.
        """
        if self.importance_estimator is not None:
            self.importance_estimator.update(learning_rate)
