import random
import numpy as np

"""
# Part 1
"""
def main():
    def value_iteration(P_win, gamma):
        P_lose = 1 - P_win

        V = np.zeros(101, dtype=float)
        V[100] = 1

        while True:
            prev = V.copy()
            for s in range(1, 100):
                V[s] = max(
                    (P_win * ((1 if s + bet >= 100 else 0) + (gamma * prev[s + bet]))) \
                        + (P_lose * ((1 if s - bet >= 100 else 0) + (gamma * prev[s - bet])))
                    for bet in range(1, min(s, 100 - s) + 1)
                )

            if np.abs(V - prev).sum() < 1e-9:
                break

        return V

    V = value_iteration(0.5, 1)

    print(V[0])
    for i in range(1, 101):
        print(f"{V[i]:.2f}", end=" ")
        if i % 10 == 0 and i > 0:
            print()

if __name__ == "__main__":
    main()
