import argparse
import matplotlib.pyplot as plt
import numpy as np
import gym
import pandas as pd
from scipy.signal import find_peaks

training_path = 'train.xlsx'
validation_path = 'validate.xlsx'
class Electric_Car(gym.Env):

    def __init__(self, file_path):

        # Define a continuous action space, -1 to 1. (You can discretize this later!)
        self.continuous_action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.state = np.empty(7)

        # Define the test data
        self.test_data = pd.read_excel(file_path)
        self.price_values = self.test_data.iloc[:, 1:25].to_numpy()
        self.timestamps = self.test_data['PRICES']

        # Battery characteristics
        self.battery_capacity = 50                      # kWh
        self.max_power = 25/0.9                         # kW
        self.charge_efficiency = 0.9                    # -
        self.discharge_efficiency = 0.9                 # -
        self.battery_level = self.battery_capacity/2    # kWh (start at 50%)
        self.minimum_morning_level = 20                 # kWh
        self.car_use_consumption = 20                   # kWh

        # Time Tracking
        self.counter = 0
        self.hour = 1
        self.day = 1
        self.car_is_available = True

    def step(self, action):

        action = np.squeeze(action)      # Remove the extra dimension # NOTE: wtf zit dit erin??

        # Calculate if, at 7am and after the chosen action, the battery level will be below the minimum morning level:
        if self.hour == 7:
            if action > 0 and (self.battery_level < self.minimum_morning_level):
                if (self.battery_level + action*self.max_power*self.charge_efficiency) < self.minimum_morning_level:     # If the chosen action will not charge the battery to 20kWh
                    action = (self.minimum_morning_level - self.battery_level)/(self.max_power*self.charge_efficiency)  # Charge until 20kWh
            elif action < 0:
                if (self.battery_level + action*self.max_power) < self.minimum_morning_level:
                    if self.battery_level < self.minimum_morning_level:                                                    # If the level was lower than 20kWh, charge until 20kWh
                        action = (self.minimum_morning_level - self.battery_level)/(self.max_power*self.charge_efficiency) # Charge until 20kWh
                    elif self.battery_level >= self.minimum_morning_level:                                                 # If the level was higher than 20kWh, discharge until 20kWh
                        action = (self.minimum_morning_level - self.battery_level)/(self.max_power)                        # Discharge until 20kWh
            elif action == 0:
                if self.battery_level < self.minimum_morning_level:
                    action = (self.minimum_morning_level - self.battery_level)/(self.max_power*self.charge_efficiency)

        # There is a 50% chance that the car is unavailable from 8am to 6pm
        if self.hour == 8:
            self.car_is_available = np.random.choice([True, False])
            if not self.car_is_available:
                self.battery_level -= self.car_use_consumption
        if self.hour == 18:
            self.car_is_available = True
        if not self.car_is_available:
            action = 0

        # Calculate the costs and battery level when charging (action >0)
        if (action >0) and (self.battery_level <= self.battery_capacity):
            if (self.battery_level + action*self.max_power*self.charge_efficiency) > self.battery_capacity:
                action = (self.battery_capacity - self.battery_level)/(self.max_power*self.charge_efficiency)
            charged_electricity_kW = action * self.max_power
            charged_electricity_costs = charged_electricity_kW * self.price_values[self.day-1][self.hour-1] * 2 * 1e-3
            reward = -charged_electricity_costs
            self.battery_level += charged_electricity_kW*self.charge_efficiency

        # Calculate the profits and battery level when discharging (action <0)
        elif (action < 0) and (self.battery_level >= 0):
            if (self.battery_level + action*self.max_power) < 0:
                action = -self.battery_level/(self.max_power)
            discharged_electricity_kWh = action * self.max_power           # Negative discharge value
            discharged_electricity_profits = abs(discharged_electricity_kWh) * self.discharge_efficiency * self.price_values[self.day-1][self.hour-1] * 1e-3
            reward = discharged_electricity_profits
            self.battery_level += discharged_electricity_kWh
            # Some small numerical errors causing the battery level to be 1e-14 to 1e-17 under 0 :
            if self.battery_level < 0:
                self.battery_level = 0

        else:
            reward = 0

        self.counter += 1    # Increase the counter
        self.hour += 1       # Increase the hour

        if self.counter % 24 == 0:  # If the counter is a multiple of 24, increase the day, reset hour to first hour
            self.day += 1
            self.hour = 1

        terminated = self.counter == len(self.price_values.flatten()) - 1   # If the counter is equal to the number of hours in the test data, terminate the episode
        truncated = False

        info = action                                                 # The final action taken after all constraints!
        self.state = self.observation()                               # Update the state

        return self.state, reward, terminated, truncated, info

    def observation(self):  # Returns the current state
        battery_level = self.battery_level
        price = self.price_values[self.day -1][self.hour-1]
        hour = self.hour
        day_of_week = self.timestamps[self.day -1].dayofweek  # Monday = 0, Sunday = 6
        day_of_year = self.timestamps[self.day -1].dayofyear  # January 1st = 1, December 31st = 365
        month = self.timestamps[self.day -1].month            # January = 1, December = 12
        year = self.timestamps[self.day -1].year
        self.state = np.array([battery_level, price, int(hour), int(day_of_week), int(day_of_year), int(month), int(year)])

        return self.state

def price_peaks(prices_data, hour, day):
    peak = False
    trough = False

    if day > 2:

        # prices of last week and today
        prices_ = prices_data[day-2:day, :]

        # remove today's prices after the current hour
        prices_[-1, hour:] = np.nan

        # remove hours before current hour of first day
        prices_[0, :hour] = np.nan

        # drop nan values
        prices_ = prices_[~np.isnan(prices_)]

        # duplicate prices
        prices_ = np.concatenate((prices_, prices_))


        # get index of peaks and troughs
        peaks, _ = find_peaks(prices_, distance=12)
        troughs, _ = find_peaks(-prices_, distance=12)

        if hour in peaks:
            peak = True

        if hour in troughs:
            trough = True

    return peak, trough

env = Electric_Car(training_path)
prices = env.price_values
prices = prices[prices < 300]
quantiles = np.arange(0, 1, 0.1)
quantile_values = np.quantile(prices, quantiles)
num_bins = len(quantile_values)
num_battery_bins = 4
num_seasons = 4
num_weekend = 2
print("Number of bins for price:", num_bins)

def bin_price(price, prices, quantile_values = quantile_values):

  # bin price into based on quantiles
  if price > quantile_values[-1]:
      price = len(quantile_values)

  else:
      for i in range(len(quantile_values)):
          if price < quantile_values[i]:
              price = i
              break
  return price

def get_state(observation, price_values_):
    battery_level, price, hour_of_day, day_of_week, day_of_year, month_of_year, year = observation
    hour_of_day = hour_of_day - 1 # Hour of day is between 1 and 24, but the Q-table is between 0 and 23
    month_of_year = month_of_year - 1 # Month of year is between 1 and 12, but the Q-table is between 0 and 11

    if day_of_week == 5 or day_of_week == 6:
        weekend = 0 # True
    else:
        weekend = 1 # False

    month_of_year, weekend, hour_of_day= int(month_of_year), int(weekend), int(hour_of_day)

    season = 0
    if month_of_year == 11 or month_of_year == 0 or month_of_year == 1:
        season = 0
    elif month_of_year == 2 or month_of_year == 3 or month_of_year == 4:
        season = 1
    elif month_of_year == 5 or month_of_year == 6 or month_of_year == 7:
        season = 2
    elif month_of_year == 8 or month_of_year == 9 or month_of_year == 10:
        season = 3

    # bin battery level into 4 bins
    if battery_level < 12.5:
        battery_level = 0
    elif battery_level < 25:
        battery_level = 1
    elif battery_level < 37.5:
        battery_level = 2
    else:
        battery_level = 3

    price = bin_price(price, price_values_)

    current_state = [hour_of_day, weekend, season, battery_level, price]

    return current_state

class QLearningAgent():
    def __init__(self):
        self.action_space_size = 3
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = {}

    def get_q_value(self, state, action):
        state_action = tuple(state + [action])
        return self.q_table.get(state_action, 0.0)

    def update_q_value(self, state, action, new_q_value):
        state_action = tuple(state + [action])
        self.q_table[state_action] = new_q_value

    def choose_action(self, state):
        actions = [-1, 0, 1]
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)
        else:
            q_values = [self.get_q_value(state, a) for a in actions]
            index = np.argmax(q_values)
            return actions[index]

def run_simulation(n_simulations = 10, lr=0.01, discount=0.9, e=0.4, train = True, print_ = True, decay = True, end_e = 0.01, number_of_bins_price = num_bins, number_of_battery_bins =num_battery_bins, initial_q_value_buy = 1,initial_q_value_sell = -1 ):

    ql_agent = QLearningAgent()

    if train == True:
        file_path = training_path

    # if validation data, use the Q-learning agent you trained in the previous step
    else:
        file_path = validation_path

        # load the Q-table from the previous step
        ql_agent.q_table = np.load('q_table.npy', allow_pickle=True).item()

    # Use (hyper)parameters of your choice
    ql_agent.learning_rate = lr
    ql_agent.discount_factor = discount
    ql_agent.epsilon = e
    epsilon_decay = (e - end_e) / n_simulations
    bankaccounts = []

    for i in range(number_of_bins_price):
        for j in range(number_of_battery_bins):
            for k in range(num_seasons):
                for q in range(num_weekend):
                    state_action = (4, q, k, j, i, 1)
                    ql_agent.q_table[state_action] = initial_q_value_buy

    # for i in range(number_of_bins_price):
    #     for j in range(number_of_battery_bins):
    #         for k in range(num_seasons):
    #             for q in range(num_weekend):
    #                 state_action = (16, q, k, j, i, -1)
    #                 ql_agent.q_table[state_action] = initial_q_value_sell

    for simulation in range(n_simulations):

        if print_:
            print('Simulation: ', simulation)
        if decay:
          ql_agent.epsilon = max(end_e, ql_agent.epsilon - epsilon_decay)

        env = Electric_Car(file_path)
        bankaccount = 0
        observation = env.observation()

        # get the current states
        current_state = get_state(observation, env.price_values)

        for i in range(730 * 24 - 1):  # Loop through 2 years -> 730 days * 24 hours

            # Choose an action based on the observation using your Q-learning agent
            action = int(ql_agent.choose_action(current_state))
            next_observation, reward, terminated, truncated, info = env.step(action)

            # save reward before shaping
            reward_ = reward

            # get peaks and troughs
            peak, trough = price_peaks(env.price_values, env.hour-1, env.day-1)

            # reward factor
            if peak and action < 0: # if peak and discharge (getting a positive reward)
                reward *= 2 # multiply by 1.5

            elif trough and action < 0: # if trough and discharge (getting a positive reward)
                reward *= -1 # flip to negative reward

            if trough and action > 0: # if trough and charge (getting a negative reward)
                reward *= -1 # flip to positive reward

            elif peak and action > 0: # if peak and charge (getting a negative reward)
                reward *= 2 # make more negative

            # Get next states
            next_state = get_state(next_observation, env.price_values)

            # Update Q-value using Q-learning update rule
            next_max_q_value = max([ql_agent.get_q_value(next_state, a) for a in [-1, 0, 1]])
            current_q_value = ql_agent.get_q_value(current_state, action)
            new_q_value = current_q_value + ql_agent.learning_rate * (reward + ql_agent.discount_factor * next_max_q_value - current_q_value)
            ql_agent.update_q_value(current_state, action, new_q_value)

            bankaccount += reward_
            observation = next_observation
            current_state = next_state

            # If the episode is terminated, reset the environment and break the loop
            if i == (729*24 - 2):
                bankaccounts.append(bankaccount) # Save the cumulative reward of the last day of the simulation
                break
    print(ql_agent.q_table)
    if train:
        plt.plot(bankaccounts) # Plot the cumulative reward per simulation
        plt.xlabel('Simulation')
        plt.ylabel('Cumulative reward (€)')
        plt.show()

        # save the Q-table
        np.save('q_table.npy', ql_agent.q_table)

        # Return the bankaccount of the last simulation
        return bankaccount
    # If validation data, return the average cumulative reward of all simulations
    # for stochasticity reasons
    else:
        return np.mean(bankaccounts)


run_simulation(n_simulations = 100, lr=0.01, discount=0.9, e=0.4, train = True)