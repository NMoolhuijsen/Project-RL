import numpy as np
import matplotlib.pyplot as plt
# Load q_table from file
# q_table = np.load('q_table.npy', allow_pickle=True).item()
#
# price = []
# hour = []
# values = []
#
# class QLearningAgent():
#     def __init__(self):
#         self.action_space = [-1, 0, 1]
#         self.learning_rate = 0.1
#         self.discount_factor = 0.9
#         self.epsilon = 0.1
#         self.q_table = {}
#
#     def get_q_value(self, state, action):
#         state_action = tuple(state + [action])
#         return self.q_table.get(state_action, 0.0)
#
#     def update_q_value(self, state, action, new_q_value):
#         state_action = tuple(state + [action])
#         self.q_table[state_action] = new_q_value
#
#     def choose_action(self, state):
#         actions = self.action_space
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(actions)
#         else:
#             q_values = [self.get_q_value(state, a) for a in actions]
#             index = np.argmax(q_values)
#             return actions[index]
#
# ql_agent = QLearningAgent()
#
# def init_q_table(ql_agent):
#     # Initialize the q table with better values for some hours
#     q_table = ql_agent.q_table  # [hour_of_day, battery_level, price, action]
#
#     for action in [-1, 1]:
#
#         if action == -1:  # selling
#
#             # init negative value for low price bins for all hours
#             for price in range(0, 3):
#                 for hour in range(24):
#                     for battery_level in range(1, 4):
#                         state_action = (hour, battery_level, price, action)
#                         q_table[state_action] = -10
#
#             # init positive value for high price bins
#             for price in range(7, 10):
#                 for hour in range(24):
#                     for battery_level in range(1, 4):
#                         state_action = (hour, battery_level, price, action)
#                         q_table[state_action] = 10
#
#         elif action == 1:  # buying
#
#             # init positive value for low price bins
#             for price in range(0, 3):
#                 for hour in range(24):
#                     for battery_level in range(3):
#                         state_action = (hour, battery_level, price, action)
#                         q_table[state_action] = 10
#
#             # init negative value for high price bins
#             for price in range(7, 10):
#                 for hour in range(24):
#                     for battery_level in range(3):
#                         state_action = (hour, battery_level, price, action)
#                         q_table[state_action] = -10
#
#     return q_table
#
# q_table = init_q_table(ql_agent)
q_table = np.load('q_table_init_tuned.npy', allow_pickle=True).item()
price = []
hour = []
values = []

for i in range(24):
    for j in range(10):
        hour.append(i)
        price.append(j)
        # Retrieve the value from q_table. If the key doesn't exist, return 0.
        value = q_table.get((i, 2, j, 1), 0)
        values.append(value)

# Reshape values to match the dimensions of the meshgrid
values = np.array(values).reshape((24, 10))
def threeD_plot(hour, price, values):
    # Create a meshgrid for hours and prices
    X, Y = np.meshgrid(np.unique(hour), np.unique(price))

    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface using meshgrid
    surf = ax.plot_surface(X, Y, values.T, cmap='viridis')  # Transpose values

    # Customize the plot
    ax.set_xlabel('Hour', fontsize=14)
    ax.set_ylabel('Price', fontsize=14)
    ax.set_zlabel('Q-value', fontsize=14)
    ax.set_title('Q-values of state combination by action Buy and battery capacity 25-37,5 kwh', fontsize=16)

    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    # Show the plot
    plt.show()

threeD_plot(hour, price, values)