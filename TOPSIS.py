import numpy as np
import pandas as pd

def topsis(data, weights, impacts):
    """
    Perform TOPSIS analysis.

    Parameters:
    data (numpy.ndarray): Decision matrix (alternatives x criteria).
    weights (list or numpy.ndarray): Criteria weights.
    impacts (list): List of '+' for benefit criteria and '-' for cost criteria.

    Returns:
    numpy.ndarray: TOPSIS scores for each alternative.
    numpy.ndarray: Rankings of alternatives.
    """
    # Normalize the decision matrix
    norm_data = data / np.sqrt(np.sum(data**2, axis=0))

    # Weight the normalized matrix
    weighted_data = norm_data * weights

    # Determine the ideal and negative-ideal solutions
    ideal = np.where(np.array(impacts) == '+', weighted_data.max(axis=0), weighted_data.min(axis=0))
    negative_ideal = np.where(np.array(impacts) == '+', weighted_data.min(axis=0), weighted_data.max(axis=0))

    # Calculate distances from ideal and negative-ideal solutions
    dist_to_ideal = np.sqrt(np.sum((weighted_data - ideal)**2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_data - negative_ideal)**2, axis=1))

    # Calculate the TOPSIS score
    scores = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

    # Rank alternatives based on scores (higher score = better rank)
    rankings = scores.argsort()[::-1] + 1

    return scores, rankings

# Example Usage
if __name__ == "__main__":
    # Input decision matrix (rows = alternatives, columns = criteria)
    decision_matrix = np.array([
    [300, 0.111, 0.1, 7],
    [300, 0.111, 0.15, 9],
    [300, 0.111, 0.2, 11],
    [300, 0.122, 0.1, 7],
    [300, 0.122, 0.15, 10],
    [300, 0.122, 0.2, 12],
    [300, 0.134, 0.1, 8],
    [300, 0.134, 0.15, 14],
    [300, 0.134, 0.2, 15],
    [755, 0.111, 0.1, 7],
    [755, 0.111, 0.15, 10],
    [755, 0.111, 0.2, 11],
    [755, 0.122, 0.1, 7],
    [755, 0.122, 0.15, 11],
    [755, 0.122, 0.2, 10],
    [755, 0.134, 0.1, 7],
    [755, 0.134, 0.15, 11],
    [755, 0.134, 0.2, 13],
    [1255, 0.111, 0.1, 6],
    [1255, 0.111, 0.15, 7],
    [1255, 0.111, 0.2, 10],
    [1255, 0.122, 0.1, 5],
    [1255, 0.122, 0.15, 8],
    [1255, 0.122, 0.2, 10],
    [1255, 0.134, 0.1, 7],
    [1255, 0.134, 0.15, 8],
    [1255, 0.134, 0.2, 11],
])

    # Weights for each criterion
    weights = np.array([0.3, 0.25, 0.2, 0.25])  # Adjust weights as needed

    # Impacts for each criterion ('+' for benefit, '-' for cost)
    impacts = ['+', '+', '+', '-']

    # Perform TOPSIS
    scores, rankings = topsis(decision_matrix, weights, impacts)

    # Create a DataFrame for better display
    results_df = pd.DataFrame(
        decision_matrix,
        columns=["Speed", "Feed", "Depth of Cut", "Cutting Force"]
    )
    results_df["TOPSIS Score"] = scores
    results_df["Rank"] = rankings

    # Sort by rank
    results_df = results_df.sort_values(by="Rank")

    # Display results
    print(results_df)

