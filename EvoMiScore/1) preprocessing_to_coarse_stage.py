import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau, mannwhitneyu

def custom_clr(X):
    """
    Centered Log-Ratio (CLR) transformation for compositional data
    
    Parameters:
    -----------
    X : DataFrame
        Input compositional data
    
    Returns:
    --------
    numpy.ndarray
        CLR transformed data
    """
    # Ensure data is numpy array and float type
    X1 = np.array(X).astype(float)
    X_processed = X1.copy()
    
    # Replace zero values with minimum non-zero value
    min_nonzero = np.min(X[X > 0])
    X_processed[X_processed == 0] = min_nonzero
    
    # Calculate geometric mean and perform CLR transformation
    geomean = np.exp(np.mean(np.log(X_processed), axis=1))
    return np.log(X_processed / geomean[:, np.newaxis])

def stratified_train_test_split(X, y, test_size=0.3, random_state=20):
    """
    Stratified sampling for train/test split
    
    Parameters:
    -----------
    X : DataFrame
        Feature data
    y : Series
        Label data
    test_size : float, default=0.3
        Proportion of test set
    random_state : int, default=20
        Random seed
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        
    return X_train, X_test, y_train, y_test

def calculate_kendall_tau(data, target_col='label'):
    """
    Calculate Kendall tau correlation coefficient and feature importance ranking
    
    Parameters:
    -----------
    data : DataFrame
        Complete data including features and labels
    target_col : str, default='label'
        Target variable column name
    
    Returns:
    --------
    DataFrame
        Feature importance table sorted by absolute tau values
    """
    input_data = data.astype(float)
    
    # Separate positive and negative samples for Mann-Whitney U test
    pos_mask = input_data[target_col] == 1
    pos = input_data[pos_mask]
    neg_mask = input_data[target_col] == 0
    neg = input_data[neg_mask]
    
    tau = []
    p_values = []
    
    # Calculate for all features (excluding label column)
    feature_cols = [col for col in data.columns if col != target_col]
    
    for col in feature_cols:
        # Mann-Whitney U test to calculate tau
        try:
            U1, p1 = mannwhitneyu(pos[col], neg[col], method="exact", alternative='greater')
            U2, p2 = mannwhitneyu(neg[col], pos[col], method="exact", alternative='greater')
            tau_value = (U1 - U2) / (len(pos) * len(neg))
            tau.append(tau_value)
        except:
            tau.append(0)
        
        # Kendall tau test to calculate p-value
        try:
            _, p_value = kendalltau(data[col], data[target_col])
            p_values.append(p_value)
        except:
            p_values.append(1.0)
    
    # Create result DataFrame
    tau_abs = pd.DataFrame({
        'feature': feature_cols,
        'tau_value': tau,
        '|tau_value|': [abs(t) for t in tau],
        'p_value': p_values
    })
    
    # Sort by absolute tau values
    tau_abs = tau_abs.sort_values(by='|tau_value|', ascending=False).reset_index(drop=True)
    
    return tau_abs

def preprocess_crc_data(raw_data_path, label_path, top_features=40, 
                       test_size=0.3, random_state=20, include_clinical=True,
                       output_dir=None):
    """
    Complete CRC data preprocessing pipeline
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw 16S data file
    label_path : str
        Path to label data file
    top_features : int, default=40
        Number of top features to select
    test_size : float, default=0.3
        Proportion of test set
    random_state : int, default=20
        Random seed
    include_clinical : bool, default=True
        Whether to include clinical variables
    output_dir : str, optional
        Output directory; if None, files are not saved
    
    Returns:
    --------
    dict
        Dictionary containing all processing results
    """
    
    print("1. Loading raw data...")
    raw_data = pd.read_csv(raw_data_path)
    label_data = pd.read_csv(label_path)
    
    print("2. Performing relative abundance normalization...")
    # Relative abundance calculation
    data = raw_data.iloc[:, 1:]  # All columns except the first
    data = data.div(data.sum(axis=0), axis=1)
    data = pd.concat([raw_data.iloc[:, 0], data], axis=1)
    
    print("3. Transposing data and merging labels...")
    # Transpose data
    df = data.transpose()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.rename(columns={'index': 'ID'})
    
    # Merge labels
    df = df.merge(label_data, how='inner', on='ID')
    
    print("4. Feature filtering and CLR transformation...")
    # Extract features and labels
    if include_clinical:
        # Assume clinical variables are in the last few columns
        X = df.iloc[:, 1:-7]  # Adjust according to your data structure
        clinical_cols = ['age', 'sex', 'bmi']
    else:
        X = df.iloc[:, 1:-1]  # All columns except ID and label
        clinical_cols = []
    
    y = df['label']
    
    # Remove all-zero columns and columns with 90% zeros
    X = X.loc[:, (X != 0).any(axis=0)]
    X = X.loc[:, (X != 0).sum() > 0.05 * X.shape[0]]
    
    print("5. CLR transformation...")
    # CLR transformation
    X_clr = custom_clr(X)
    X_clr = pd.DataFrame(X_clr, columns=X.columns)
    
    # Add clinical variables back if included
    if include_clinical:
        for col in clinical_cols:
            if col in df.columns:
                X_clr[col] = df[col].values
        # Sex encoding
        if 'sex' in X_clr.columns:
            X_clr['sex'] = X_clr['sex'].replace({'female': 0, 'male': 1})
    
    print("6. Stratified sampling...")
    # Stratified sampling
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_clr, y, test_size=test_size, random_state=random_state
    )
    
    print("7. Calculating Kendall tau...")
    # Combine training data for tau calculation
    train_data = pd.concat([X_train, y_train], axis=1)
    tau_results = calculate_kendall_tau(train_data)
    
    print("8. Selecting top features...")
    # Select top features
    top_feature_names = tau_results['feature'].head(top_features).tolist()
    X_train_selected = X_train[top_feature_names]
    X_test_selected = X_test[top_feature_names]
    
    print("9. Standardization...")
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Convert back to DataFrame
    X_train_final = pd.DataFrame(X_train_scaled, columns=top_feature_names)
    X_test_final = pd.DataFrame(X_test_scaled, columns=top_feature_names)
    
    # Reset indices
    y_train_final = y_train.reset_index(drop=True)
    y_test_final = y_test.reset_index(drop=True)
    
    print("10. Combining final data...")
    # Combine final data
    train_final = pd.concat([X_train_final, y_train_final], axis=1)
    test_final = pd.concat([X_test_final, y_test_final], axis=1)
    
    # Save files (if output directory is specified)
    if output_dir:
        print("11. Saving processed data...")
        tau_results.to_csv(f"{output_dir}/tau_results.csv", index=False)
        train_final.to_csv(f"{output_dir}/train_processed.csv", index=False)
        test_final.to_csv(f"{output_dir}/test_processed.csv", index=False)
        
        print(f"Files saved to {output_dir}")
    
    # Return results
    results = {
        'train_data': train_final,
        'test_data': test_final,
        'tau_results': tau_results,
        'scaler': scaler,
        'top_features': top_feature_names,
        'train_shape': train_final.shape,
        'test_shape': test_final.shape
    }
    
    print("Preprocessing completed!")
    print(f"Training set shape: {train_final.shape}")
    print(f"Test set shape: {test_final.shape}")
    print(f"Top {top_features} features selected and saved")
    
    return results

