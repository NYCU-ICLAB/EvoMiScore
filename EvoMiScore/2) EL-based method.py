import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import random
from typing import Tuple, Dict, List

class Individual:
    """Individual representing a solution (feature subset + SVM parameters)"""
    
    def __init__(self, n_features: int, min_features: int = 5, max_features: int = 50,
                 C_min: float = 0.01, C_max: float = 100.0, 
                 gamma_min: float = 0.001, gamma_max: float = 10.0):
        self.n_features = n_features
        self.min_features = min_features
        self.max_features = max_features
        self.C_min = C_min
        self.C_max = C_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        
        # Initialize feature selection and SVM parameters
        self.features = self._initialize_features()
        self.C = self._random_log_uniform(C_min, C_max)
        self.gamma = self._random_log_uniform(gamma_min, gamma_max)
        self.fitness = None
        
    def _initialize_features(self) -> np.ndarray:
        """Initialize feature selection randomly"""
        features = np.zeros(self.n_features, dtype=int)
        n_selected = random.randint(self.min_features, 
                                  min(self.max_features, self.n_features))
        selected_indices = random.sample(range(self.n_features), n_selected)
        features[selected_indices] = 1
        return features
    
    def _random_log_uniform(self, min_val: float, max_val: float) -> float:
        """Generate random value in log-uniform distribution"""
        return np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features"""
        return np.where(self.features == 1)[0]
    
    def n_selected_features(self) -> int:
        """Get number of selected features"""
        return np.sum(self.features)
    
    def copy(self):
        """Create a copy of the individual"""
        new_individual = Individual(self.n_features, self.min_features, self.max_features,
                                  self.C_min, self.C_max, self.gamma_min, self.gamma_max)
        new_individual.features = self.features.copy()
        new_individual.C = self.C
        new_individual.gamma = self.gamma
        new_individual.fitness = self.fitness
        return new_individual

class GASVMOptimizer:
    """Simple Genetic Algorithm for feature selection and SVM parameter optimization"""
    
    def __init__(self, population_size: int = 50, n_generations: int = 300,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.1,
                 elite_size: int = 5, tournament_size: int = 3,
                 min_features: int = 5, max_features: int = 50,
                 C_min: float = 0.01, C_max: float = 100.0,
                 gamma_min: float = 0.001, gamma_max: float = 10.0,
                 cv_folds: int = 5, random_state: int = 42, scoring: str = 'accuracy'):
        
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.min_features = min_features
        self.max_features = max_features
        self.C_min = C_min
        self.C_max = C_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scoring = scoring
        
        self.population = []
        self.best_individual = None



    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GASVMOptimizer':
        """
        Optimize feature selection and SVM parameters using GA
        
        Parameters:
        -----------
        X : ndarray
            Feature matrix
        y : ndarray  
            Target labels
            
        Returns:
        --------
        self : GASVMOptimizer
        """
        self.X = X
        self.y = y
        
        # Set random seeds
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        print(f"GA optimization: {X.shape[1]} features, {len(y)} samples")
        print(f"Population: {self.population_size}, Generations: {self.n_generations}")
        
        # Initialize population
        self._initialize_population()
        
        # Evolution
        for generation in range(self.n_generations):
            # Evaluate population
            self._evaluate_population()
            
            # Update best individual
            current_best = max(self.population, key=lambda x: x.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = current_best.copy()
            
            # Print progress
            if generation % 20 == 0 or generation == self.n_generations - 1:
                print(f"Gen {generation:3d}: Best={self.best_individual.fitness:.4f}, "
                      f"Features={self.best_individual.n_selected_features()}, "
                      f"C={self.best_individual.C:.3f}, gamma={self.best_individual.gamma:.3f}")
            
            # Create next generation
            if generation < self.n_generations - 1:
                self.population = self._create_next_generation()
        
        print(f"Optimization completed! Best fitness: {self.best_individual.fitness:.4f}")
        return self
    
    def _initialize_population(self):
        """Initialize the population"""
        self.population = []
        for _ in range(self.population_size):
            individual = Individual(
                self.X.shape[1], self.min_features, self.max_features,
                self.C_min, self.C_max, self.gamma_min, self.gamma_max
            )
            self.population.append(individual)
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals"""
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self._evaluate_individual(individual)
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """Evaluate individual using cross-validation"""
        selected_features = individual.get_selected_features()
        
        if len(selected_features) == 0:
            return 0.0
        
        X_selected = self.X[:, selected_features]
        
        # Create SVM
        svm = SVC(
            C=individual.C,
            gamma=individual.gamma,
            kernel='rbf',
            random_state=self.random_state,
            probability=True if self.scoring == 'roc_auc' else False
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                           random_state=self.random_state)
        
        try:
            cv_scores = cross_val_score(svm, X_selected, self.y, cv=cv, 
                                      scoring=self.scoring, n_jobs=-1)
            fitness = np.mean(cv_scores)
            
            # Small penalty for too many features
            feature_penalty = len(selected_features) / self.X.shape[1] * 0.01
            fitness = fitness - feature_penalty
            
        except:
            fitness = 0.0
        
        return fitness
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Crossover operation"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if random.random() < self.crossover_rate:
            # Feature crossover
            for i in range(len(child1.features)):
                if random.random() < 0.5:
                    child1.features[i], child2.features[i] = child2.features[i], child1.features[i]
            
            # Parameter crossover
            alpha = random.random()
            new_C1 = alpha * parent1.C + (1 - alpha) * parent2.C
            new_C2 = alpha * parent2.C + (1 - alpha) * parent1.C
            new_gamma1 = alpha * parent1.gamma + (1 - alpha) * parent2.gamma
            new_gamma2 = alpha * parent2.gamma + (1 - alpha) * parent1.gamma
            
            child1.C, child1.gamma = new_C1, new_gamma1
            child2.C, child2.gamma = new_C2, new_gamma2
            
            child1.fitness = None
            child2.fitness = None
        
        return child1, child2
    
    def _mutate(self, individual: Individual):
        """Mutation operation"""
        if random.random() < self.mutation_rate:
            # Feature mutation
            for i in range(len(individual.features)):
                if random.random() < 0.1:
                    individual.features[i] = 1 - individual.features[i]
            
            # Ensure minimum features
            if individual.n_selected_features() < self.min_features:
                unselected = np.where(individual.features == 0)[0]
                n_to_add = self.min_features - individual.n_selected_features()
                if len(unselected) >= n_to_add:
                    to_select = random.sample(list(unselected), n_to_add)
                    individual.features[to_select] = 1
            
            # Parameter mutation
            if random.random() < 0.3:
                individual.C *= np.exp(np.random.normal(0, 0.1))
                individual.C = max(self.C_min, min(self.C_max, individual.C))
            
            if random.random() < 0.3:
                individual.gamma *= np.exp(np.random.normal(0, 0.1))
                individual.gamma = max(self.gamma_min, min(self.gamma_max, individual.gamma))
            
            individual.fitness = None
    
    def _create_next_generation(self) -> List[Individual]:
        """Create next generation"""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = []
        
        # Elitism
        for i in range(self.elite_size):
            new_population.append(self.population[i].copy())
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            child1, child2 = self._crossover(parent1, parent2)
            
            self._mutate(child1)
            self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def get_best_features(self) -> np.ndarray:
        """Get indices of best selected features"""
        if self.best_individual is None:
            return np.array([])
        return self.best_individual.get_selected_features()
    
    def get_best_params(self) -> Dict:
        """Get best parameters"""
        if self.best_individual is None:
            return {}
        
        return {
            'C': self.best_individual.C,
            'gamma': self.best_individual.gamma,
            'n_features': self.best_individual.n_selected_features(),
            'fitness': self.best_individual.fitness,
            'selected_features': self.get_best_features()
        }
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Make predictions using best model"""
        if self.best_individual is None:
            raise ValueError("Model not fitted!")
        
        selected_features = self.get_best_features()
        X_train_selected = self.X[:, selected_features]
        X_test_selected = X_test[:, selected_features]
        
        # Train final model
        svm = SVC(
            C=self.best_individual.C,
            gamma=self.best_individual.gamma,
            kernel='rbf',
            random_state=self.random_state
        )
        
        svm.fit(X_train_selected, self.y)
        return svm.predict(X_test_selected)

def optimize_features_and_svm(X_train, y_train, X_test=None, y_test=None, 
                             population_size=50, n_generations=100, 
                             max_features=30, random_state=42):
    """
    Simple function to optimize features and SVM parameters
    
    Parameters:
    -----------
    X_train : DataFrame or ndarray
        Training features  
    y_train : Series or ndarray
        Training labels
    X_test : DataFrame or ndarray, optional
        Test features
    y_test : Series or ndarray, optional  
        Test labels
    population_size : int
        GA population size
    n_generations : int
        Number of generations
    max_features : int
        Maximum number of features to select
    random_state : int
        Random seed
        
    Returns:
    --------
    dict : Results containing optimizer and metrics
    """
    
    # Convert to numpy arrays if needed
    X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
    
    # Initialize optimizer
    optimizer = GASVMOptimizer(
        population_size=population_size,
        n_generations=n_generations,
        max_features=max_features,
        random_state=random_state
    )
    
    # Fit optimizer
    optimizer.fit(X_train_array, y_train_array)
    
    results = {
        'optimizer': optimizer,
        'best_params': optimizer.get_best_params(),
        'cv_score': optimizer.best_individual.fitness
    }
    
    # Test if test data provided
    if X_test is not None and y_test is not None:
        X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
        y_test_array = y_test.values if hasattr(y_test, 'values') else y_test
        
        y_pred = optimizer.predict(X_test_array)
        test_accuracy = accuracy_score(y_test_array, y_pred)
        results['test_accuracy'] = test_accuracy
        
        print(f"Test accuracy: {test_accuracy:.4f}")
    
    return results


    '''
    Reference: Shinn-ying Ho et al. (2004). "Intelligent evolutionary algorithms for 
    large parameter optimization problems." IEEE Transactions on Evolutionary Computation.

    The Orthogonal Array (OA) method was published in 2004 by Shinn-ying Ho and provides
    a more systematic approach to crossover operations, particularly effective for large
    parameter optimization problems like feature selection with high-dimensional data.

    Algorithm Generate_OA(OA, N):
    {
        n := 2⌈log2(N+1)⌉;
        for i := 1 to n do
            for j := 1 to N do
                level := 0;
                k = j;
                mask := n/2;
                while k > 0 do
                    if (k mod 2) and (bitwise_AND(i-1, mask) ≠ 0) then
                        level := (level+1) mod 2;
                    k := ⌊k/2⌋;
                    mask := mask/2;
                OA[i][j] := level+1;
    }

    '''
