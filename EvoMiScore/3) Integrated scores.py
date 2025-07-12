"""
SVM Integration for Multi-class Decision Fusion

This module implements a decision fusion system that integrates multiple binary SVM classifiers
for a three-class classification problem (Healthy vs HRL vs Cancer).

Expected Input Format:
---------------------
The input should be a pandas DataFrame containing decision values from three binary SVM models:

decisions_df = pd.DataFrame({
    'con_hrl': [decision_values_from_control_vs_hrl_model],
    'hrl_can': [decision_values_from_hrl_vs_cancer_model], 
    'con_can': [decision_values_from_control_vs_cancer_model]
})

true_labels = [0, 1, 2, ...]  # 0: Healthy/Control, 1: HRL, 2: Cancer

Decision values are typically obtained from SVM's decision_function() method,
which returns the distance from the separating hyperplane.

Example Usage:
-------------
# Initialize the integrator
integrator = SVMIntegration()

# Load your decision values and true labels
decisions_df = pd.DataFrame({
    'con_hrl': svm_model_1.decision_function(X_test),
    'hrl_can': svm_model_2.decision_function(X_test),
    'con_can': svm_model_3.decision_function(X_test)
})
true_labels = y_test  # Ground truth labels

# Optimize weights for best integration
best_weights, best_fitness = integrator.optimize_weights(decisions_df, true_labels)

# Calculate integrated scores
integrated_scores = integrator.calculate_integrated_score(decisions_df)

# Visualize results
fig = integrator.visualize_results(decisions_df, true_labels)
"""

from scipy.special import expit
from scipy.stats import spearmanr, kendalltau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SVMIntegration:
    """
    SVM Decision Integration for Multi-class Classification
    
    This class implements a decision fusion system that combines decision values 
    from multiple binary SVM classifiers to create an integrated score for 
    multi-class classification problems.
    
    The system is designed for a three-class problem:
    - Class 0: Healthy/Control
    - Class 1: HRL (High-Risk Lesion) 
    - Class 2: Cancer
    
    It uses three binary classifiers:
    - con_hrl: Control vs HRL
    - hrl_can: HRL vs Cancer  
    - con_can: Control vs Cancer
    """
    
    def __init__(self, weights=None):
        """
        Initialize the SVM integration system
        
        Parameters:
        -----------
        weights : dict, optional
            Weights for each binary classifier. If None, uses default values.
            Expected format: {'con_hrl': float, 'hrl_can': float, 'con_can': float}
            Weights should sum to 1.0
        """
        # Set default weights if not provided
        self.weights = weights if weights is not None else {
            'con_hrl': 0.3,     # control vs HRL weight
            'hrl_can': 0.3,     # HRL vs Cancer weight
            'con_can': 0.4      # control vs Cancer weight
        }
    
    def normalize_decision_values(self, decision_values):
        """
        Normalize decision values to [0,1] range using sigmoid function
        
        Parameters:
        -----------
        decision_values : array-like
            Raw decision values from SVM decision_function()
            
        Returns:
        --------
        array-like
            Normalized values between 0 and 1
        """
        return expit(decision_values)
    
    def calculate_integrated_score(self, decisions_df):
        """
        Calculate integrated scores by combining normalized decision values
        
        Parameters:
        -----------
        decisions_df : pd.DataFrame
            DataFrame containing decision values with columns:
            - 'con_hrl': Control vs HRL decision values
            - 'hrl_can': HRL vs Cancer decision values  
            - 'con_can': Control vs Cancer decision values
            
        Returns:
        --------
        pd.Series
            Integrated scores normalized to [0, 100] range
        """
        # Normalize all decision values using sigmoid
        normalized_decisions = pd.DataFrame({
            'con_hrl': self.normalize_decision_values(decisions_df['con_hrl']),
            'hrl_can': self.normalize_decision_values(decisions_df['hrl_can']),
            'con_can': self.normalize_decision_values(decisions_df['con_can'])
        })
        
        # Calculate weighted combination
        integrated_scores = (
            normalized_decisions['con_hrl'] * self.weights['con_hrl'] +
            normalized_decisions['hrl_can'] * self.weights['hrl_can'] +
            normalized_decisions['con_can'] * self.weights['con_can']
        )
        
        # Normalize to [0, 100] range for interpretability
        score_range = integrated_scores.max() - integrated_scores.min()
        if score_range == 0:
            return integrated_scores * 100
        
        return 100 * (integrated_scores - integrated_scores.min()) / score_range

    def calculate_fitness(self, decisions_df, true_labels):
        """
        Calculate fitness of current weight combination using Spearman correlation
        
        Parameters:
        -----------
        decisions_df : pd.DataFrame
            DataFrame containing decision values from binary classifiers
        true_labels : array-like
            Ground truth labels (0: Healthy, 1: HRL, 2: Cancer)
            
        Returns:
        --------
        float
            Fitness score (Spearman correlation coefficient) between -1 and 1
            Higher values indicate better performance
        """
        # Calculate integrated scores
        integrated_scores = self.calculate_integrated_score(decisions_df)
        
        # Calculate Spearman rank correlation with true labels
        correlation, _ = spearmanr(integrated_scores, true_labels)
        
        # Return very low fitness if correlation is NaN
        if np.isnan(correlation):
            return -1.0
            
        return correlation

    def optimize_weights(self, decisions_df, true_labels, n_iterations=1000, random_state=42):
        """
        Optimize weights using random search to maximize Spearman correlation
        
        Parameters:
        -----------
        decisions_df : pd.DataFrame
            DataFrame containing decision values from binary classifiers
        true_labels : array-like
            Ground truth labels
        n_iterations : int, default=1000
            Number of random weight combinations to try
        random_state : int, default=42
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (best_weights, best_fitness)
            - best_weights: dict with optimized weights
            - best_fitness: float with best achieved fitness score
        """
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        best_fitness = -np.inf
        best_weights = self.weights.copy()
        
        print(f"Optimizing weights over {n_iterations} iterations...")
        
        for iteration in range(n_iterations):
            # Generate random weights using Dirichlet distribution (ensures sum = 1)
            random_weights = np.random.dirichlet(np.ones(3))
            random_weights = np.round(random_weights, decimals=4)
            
            # Ensure weights sum to exactly 1.0 (adjust for rounding errors)
            random_weights = random_weights / np.sum(random_weights)
            random_weights = np.round(random_weights, decimals=4)
            
            # Final adjustment to ensure exact sum of 1.0
            random_weights[2] = np.round(1 - random_weights[0] - random_weights[1], decimals=4)
            
            test_weights = {
                'con_hrl': random_weights[0],
                'hrl_can': random_weights[1],
                'con_can': random_weights[2]
            }
            
            # Temporarily update weights
            self.weights = test_weights
            
            # Calculate fitness for this weight combination
            fitness = self.calculate_fitness(decisions_df, true_labels)
            
            # Update best weights if this combination is better
            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = test_weights.copy()
                print(f"Iteration {iteration}: New best fitness = {best_fitness:.4f}")
                print(f"Weights: {best_weights}")
        
        # Set the best weights as current weights
        self.weights = best_weights
        
        print(f"\nOptimization completed!")
        print(f"Best fitness (Spearman correlation): {best_fitness:.4f}")
        print(f"Optimal weights: {best_weights}")
        
        return best_weights, best_fitness
    
    def visualize_results(self, decisions_df, true_labels, figsize=(12, 6)):
        """
        Visualize the distribution of integrated scores for each class
        
        Parameters:
        -----------
        decisions_df : pd.DataFrame
            DataFrame containing decision values from binary classifiers
        true_labels : array-like
            Ground truth labels
        figsize : tuple, default=(12, 6)
            Figure size for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure object
        """
        # Calculate integrated scores
        integrated_scores = self.calculate_integrated_score(decisions_df)
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Define colors for each class
        colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green, Blue, Orange
        class_names = ['Healthy', 'HRL', 'Cancer']
        
        # Plot distribution for each class
        for label in np.unique(true_labels):
            mask = true_labels == label
            class_scores = integrated_scores[mask]
            
            # Plot density curve
            sns.kdeplot(data=class_scores, 
                       label=class_names[label],
                       color=colors[label],
                       fill=True,
                       alpha=0.3)
            
            # Plot individual points
            sns.scatterplot(x=class_scores,
                          y=np.zeros_like(class_scores) + 0.03,
                          color=colors[label],
                          alpha=0.6,
                          s=50)
            
            # Add median line
            median_score = np.median(class_scores)
            plt.axvline(x=median_score, color=colors[label], 
                       linestyle='--', alpha=0.7, linewidth=2)
        
        # Add correlation information to title
        correlation = self.calculate_fitness(decisions_df, true_labels)
        plt.title(f'Integrated Score Distribution\n(Spearman Correlation: {correlation:.3f})', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Integrated Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    
    def get_summary(self, decisions_df, true_labels):
        """
        Get a summary of the integration performance
        
        Parameters:
        -----------
        decisions_df : pd.DataFrame
            DataFrame containing decision values
        true_labels : array-like
            Ground truth labels
            
        Returns:
        --------
        dict
            Summary statistics and performance metrics
        """
        integrated_scores = self.calculate_integrated_score(decisions_df)
        correlation = self.calculate_fitness(decisions_df, true_labels)
        
        summary = {
            'weights': self.weights,
            'spearman_correlation': correlation,
            'score_range': [integrated_scores.min(), integrated_scores.max()],
            'score_mean': integrated_scores.mean(),
            'score_std': integrated_scores.std(),
            'class_medians': {
                'Healthy': np.median(integrated_scores[true_labels == 0]),
                'HRL': np.median(integrated_scores[true_labels == 1]),
                'Cancer': np.median(integrated_scores[true_labels == 2])
            }
        }
        
        return summary
