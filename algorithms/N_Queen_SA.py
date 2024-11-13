import numpy as np
from scipy.optimize import dual_annealing


def maxing_queen(position):
    """
    Calculate the number of non-attacking queens in the given board position.

    Args:
        position (array-like): An array representing the positions of queens on the board,
                               where index is the column and value is the row.

    Returns:
        int: Negative of the number of non-attacking queens. (Using negative for minimization)
    """
    queen_not_attacking = 0
    for i in range(len(position) - 1):
        no_attack_on_j = 0
        for j in range(i + 1, len(position)):
            # Check for no conflicts: same row, and both diagonals
            if (position[j] != position[i]) and (position[j] != position[i] + (j - i)) and (
                    position[j] != position[i] - (j - i)):
                no_attack_on_j += 1
                # If no queens are attacking each other in this iteration
                if no_attack_on_j == len(position) - 1 - i:
                    queen_not_attacking += 1
    # Special case for maximum non-attacking queens
    if queen_not_attacking == len(position) - 1:
        queen_not_attacking += 1
    # Return negative value because dual_annealing minimizes the objective
    return -queen_not_attacking


def main():
    # Set bounds for each queen's row position (0 to 7 for an 8x8 board)
    bounds = [(0, 7) for _ in range(8)]

    # Convert positions to integers within the bounds
    def rounded_maxing_queen(position):
        return maxing_queen(np.round(position).astype(int))

    # Apply dual annealing to solve the optimization problem
    result = dual_annealing(rounded_maxing_queen, bounds=bounds, maxiter=1000)

    # Output the result
    best_position = np.round(result.x).astype(int)
    best_objective = -result.fun
    print("The best position found is:", best_position)
    print("The number of queens not attacking each other is:", int(best_objective))


# Only run the main function if this script is executed directly
if __name__ == "__main__":
    main()
