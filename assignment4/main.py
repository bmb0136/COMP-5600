import random
import numpy as np

"""
# Part 1
"""
def main():
    def go_gambling(money, policy, P_win, gamma):
        assert 1 <= money <= 99, "Invalid starting capital"

        while money > 0 and money < 100:
            bet = min(policy[money], min(money, 100 - money))

            if random.random() < P_win:
                money += bet
            else:
                money -= bet

        return 1 if money >= 100 else 0

if __name__ == "__main__":
    main()
