// ==============================================================================
// Hungarian Algorithm (Munkres) Implementation
// ==============================================================================
// Based on: https://en.wikipedia.org/wiki/Hungarian_algorithm
//
// The Munkres algorithm finds the optimal assignment in a cost matrix.
// Steps:
//   1. Subtract row minimums
//   2. Star zeros (one per row/col)
//   3. Cover columns with starred zeros
//   4. Find uncovered zeros, prime them
//   5. Augmenting path to increase matching
//   6. Adjust cost matrix and repeat
//
// Guaranteed O(N^3) optimal solution.
// ==============================================================================

#include "hungarian.h"
#include <algorithm>
#include <cmath>

namespace fcw {

float HungarianAlgorithm::solve(const std::vector<std::vector<float>>& costMatrix,
                                 std::vector<int>& assignment) {
    if (costMatrix.empty()) {
        assignment.clear();
        return 0.0f;
    }

    int rows = static_cast<int>(costMatrix.size());
    int cols = static_cast<int>(costMatrix[0].size());
    int n = std::max(rows, cols);

    // Create padded square cost matrix
    std::vector<std::vector<float>> cost(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cost[i][j] = costMatrix[i][j];
        }
    }

    // mask: 0=nothing, 1=starred, 2=primed
    std::vector<std::vector<int>> mask(n, std::vector<int>(n, 0));
    std::vector<bool> rowCover(n, false);
    std::vector<bool> colCover(n, false);

    int pathRow0 = 0, pathCol0 = 0;

    // Step 1: Subtract row minimums
    step1(cost, n);

    // Step 2: Star zeros
    step2(cost, n, mask, rowCover, colCover);

    int step = 3;
    while (step != 7) {
        switch (step) {
            case 3:
                step = step3(mask, n, colCover);
                break;
            case 4:
                step = step4(cost, n, mask, rowCover, colCover, pathRow0, pathCol0);
                break;
            case 5:
                step5(mask, n, rowCover, colCover, pathRow0, pathCol0);
                step = 3;
                break;
            case 6:
                step6(cost, n, rowCover, colCover);
                step = 4;
                break;
        }
    }

    // Extract assignment
    assignment.resize(rows, -1);
    float totalCost = 0.0f;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (mask[i][j] == 1) {
                assignment[i] = j;
                totalCost += costMatrix[i][j];
                break;
            }
        }
    }

    return totalCost;
}

// Step 1: Subtract smallest element in each row
void HungarianAlgorithm::step1(std::vector<std::vector<float>>& cost, int n) {
    for (int i = 0; i < n; i++) {
        float minVal = *std::min_element(cost[i].begin(), cost[i].end());
        for (int j = 0; j < n; j++) {
            cost[i][j] -= minVal;
        }
    }
}

// Step 2: Find zeros and star them (one per row/col)
void HungarianAlgorithm::step2(const std::vector<std::vector<float>>& cost, int n,
                                std::vector<std::vector<int>>& mask,
                                std::vector<bool>& rowCover, std::vector<bool>& colCover) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(cost[i][j]) < 1e-6f && !rowCover[i] && !colCover[j]) {
                mask[i][j] = 1;  // Star
                rowCover[i] = true;
                colCover[j] = true;
            }
        }
    }
    std::fill(rowCover.begin(), rowCover.end(), false);
    std::fill(colCover.begin(), colCover.end(), false);
}

// Step 3: Cover columns with starred zeros, check if done
int HungarianAlgorithm::step3(const std::vector<std::vector<int>>& mask, int n,
                               std::vector<bool>& colCover) {
    int colCount = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mask[i][j] == 1) {
                colCover[j] = true;
            }
        }
    }
    for (int j = 0; j < n; j++) {
        if (colCover[j]) colCount++;
    }
    return (colCount >= n) ? 7 : 4;  // 7 = done
}

// Step 4: Find uncovered zero, prime it
int HungarianAlgorithm::step4(const std::vector<std::vector<float>>& cost, int n,
                               std::vector<std::vector<int>>& mask,
                               std::vector<bool>& rowCover, std::vector<bool>& colCover,
                               int& pathRow0, int& pathCol0) {
    while (true) {
        // Find uncovered zero
        int row = -1, col = -1;
        for (int i = 0; i < n && row == -1; i++) {
            if (rowCover[i]) continue;
            for (int j = 0; j < n; j++) {
                if (!colCover[j] && std::abs(cost[i][j]) < 1e-6f) {
                    row = i;
                    col = j;
                    break;
                }
            }
        }

        if (row == -1) {
            return 6;  // No uncovered zero → step 6
        }

        mask[row][col] = 2;  // Prime it

        int starCol = findStarInRow(mask, n, row);
        if (starCol != -1) {
            rowCover[row] = true;
            colCover[starCol] = false;
        } else {
            pathRow0 = row;
            pathCol0 = col;
            return 5;  // Augmenting path → step 5
        }
    }
}

// Step 5: Augmenting path - swap stars and primes along path
void HungarianAlgorithm::step5(std::vector<std::vector<int>>& mask, int n,
                                std::vector<bool>& rowCover, std::vector<bool>& colCover,
                                int pathRow0, int pathCol0) {
    // Build augmenting path
    std::vector<std::pair<int, int>> path;
    path.push_back({pathRow0, pathCol0});

    while (true) {
        int row = findStarInCol(mask, n, path.back().second);
        if (row == -1) break;
        path.push_back({row, path.back().second});

        int col = findPrimeInRow(mask, n, row);
        path.push_back({row, col});
    }

    // Augment path: unstar starred, star primed
    for (auto& [r, c] : path) {
        if (mask[r][c] == 1) mask[r][c] = 0;
        else if (mask[r][c] == 2) mask[r][c] = 1;
    }

    // Clear covers and primes
    std::fill(rowCover.begin(), rowCover.end(), false);
    std::fill(colCover.begin(), colCover.end(), false);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mask[i][j] == 2) mask[i][j] = 0;
        }
    }
}

// Step 6: Adjust matrix by smallest uncovered value
void HungarianAlgorithm::step6(std::vector<std::vector<float>>& cost, int n,
                                const std::vector<bool>& rowCover, const std::vector<bool>& colCover) {
    float minVal = findSmallest(cost, n, rowCover, colCover);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (rowCover[i]) cost[i][j] += minVal;
            if (!colCover[j]) cost[i][j] -= minVal;
        }
    }
}

// Helper: Find starred zero in row
int HungarianAlgorithm::findStarInRow(const std::vector<std::vector<int>>& mask, int n, int row) {
    for (int j = 0; j < n; j++) {
        if (mask[row][j] == 1) return j;
    }
    return -1;
}

// Helper: Find starred zero in column
int HungarianAlgorithm::findStarInCol(const std::vector<std::vector<int>>& mask, int n, int col) {
    for (int i = 0; i < n; i++) {
        if (mask[i][col] == 1) return i;
    }
    return -1;
}

// Helper: Find primed zero in row
int HungarianAlgorithm::findPrimeInRow(const std::vector<std::vector<int>>& mask, int n, int row) {
    for (int j = 0; j < n; j++) {
        if (mask[row][j] == 2) return j;
    }
    return -1;
}

// Helper: Find smallest uncovered value
float HungarianAlgorithm::findSmallest(const std::vector<std::vector<float>>& cost, int n,
                                         const std::vector<bool>& rowCover, const std::vector<bool>& colCover) {
    float minVal = std::numeric_limits<float>::max();
    for (int i = 0; i < n; i++) {
        if (rowCover[i]) continue;
        for (int j = 0; j < n; j++) {
            if (colCover[j]) continue;
            minVal = std::min(minVal, cost[i][j]);
        }
    }
    return minVal;
}

} // namespace fcw
