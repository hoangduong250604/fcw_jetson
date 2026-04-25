#pragma once
// ==============================================================================
// Hungarian Algorithm (Munkres) for optimal assignment
// ==============================================================================
// Solves the linear assignment problem: given a cost matrix, find the
// assignment of workers to jobs that minimizes the total cost.
//
// Used by SORT tracker to optimally match tracks to detections.
//
// Time complexity: O(N^3) where N = max(tracks, detections)
// For typical ADAS scenarios (<20 objects), this is fast enough.
// ==============================================================================

#include <vector>
#include <limits>

namespace fcw {

class HungarianAlgorithm {
public:
    /**
     * Solve the optimal assignment problem.
     *
     * @param costMatrix  N x M cost matrix (lower = better match)
     * @param assignment  Output: assignment[i] = j means row i matched to col j
     *                    assignment[i] = -1 means unmatched
     * @return Total minimum cost
     */
    static float solve(const std::vector<std::vector<float>>& costMatrix,
                       std::vector<int>& assignment);

private:
    // Munkres steps
    static void step1(std::vector<std::vector<float>>& cost, int n);
    static void step2(const std::vector<std::vector<float>>& cost, int n,
                      std::vector<std::vector<int>>& mask,
                      std::vector<bool>& rowCover, std::vector<bool>& colCover);
    static int step3(const std::vector<std::vector<int>>& mask, int n,
                     std::vector<bool>& colCover);
    static int step4(const std::vector<std::vector<float>>& cost, int n,
                     std::vector<std::vector<int>>& mask,
                     std::vector<bool>& rowCover, std::vector<bool>& colCover,
                     int& pathRow0, int& pathCol0);
    static void step5(std::vector<std::vector<int>>& mask, int n,
                      std::vector<bool>& rowCover, std::vector<bool>& colCover,
                      int pathRow0, int pathCol0);
    static void step6(std::vector<std::vector<float>>& cost, int n,
                      const std::vector<bool>& rowCover, const std::vector<bool>& colCover);

    static int findStarInRow(const std::vector<std::vector<int>>& mask, int n, int row);
    static int findStarInCol(const std::vector<std::vector<int>>& mask, int n, int col);
    static int findPrimeInRow(const std::vector<std::vector<int>>& mask, int n, int row);
    static float findSmallest(const std::vector<std::vector<float>>& cost, int n,
                               const std::vector<bool>& rowCover, const std::vector<bool>& colCover);
};

} // namespace fcw
