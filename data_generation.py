"""
Generates the needed data of J to train the neural network model, saves a dataframe with states and rewards as a parquet file
"""
import numpy as np
from pyworld3 import World3
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

state_variables = ["p1", "p2", "p3", "p4", "ic", "sc", "al", "pal", "uil", "lfert", "ppol", "nr", "time"]

# Standard run used for randomizing initial state
world_standard = World3(year_max=2100)
world_standard.set_world3_control()
world_standard.init_world3_constants()
world_standard.init_world3_variables()
world_standard.set_world3_table_functions()
world_standard.set_world3_delay_functions()
world_standard.run_world3(fast=False)


def J_func(reward):
    """ 
    In:
        reward - numpy array: rewards for the simlation
    Out: 
        Array of J function values
    
    Computes the cumulative reward for each step onwards
    """
    iterations = reward.shape[0]
    J = np.zeros((iterations,1))
    J[iterations-1] = reward[iterations-1]
    for k in range(2,iterations+1):
        # J[n] is the reward at step n plus J[n+1]
        J[iterations-k] = reward[iterations-k] + J[iterations-k+1] 
    return J

def reward_pop(world):
    # reward function, trying simple with population
    return world.pop

def reward_le(world):
    return world.le

def reward_pop_stable(world):
    reward = np.zeros(world.n)
    reward[1:-1] = world.pop[1:-1] - world.pop[0:-2]
    return reward


def reward_HDI(world):

    # le: life expactancy [years], want a high value
    # j/pop: determine unemployment, a high value is önskvärt and would simule a low global unemployment
    # d1: deaths per year, ages 0-14 [persons/year], should simulate infants deaths, wants a low value therefore using a minustecken 

    # Collect max-values from standard run for le, j/pop, -d1 (minustecken pga tvinga att vi vill att den är låg)
    max_le_standard = np.max(world_standard.le)
    max_jpop_standard = np.max(world_standard.j / world_standard.pop)
    max_d1_standard = np.max(world_standard.d1)

    # Create HDI
    jpop = world.j/world.pop
    reward = ((world.le/max_le_standard) + (jpop/max_jpop_standard) - (world.d1/max_d1_standard)) / 3
    return reward

#print(reward_HDI(world_standard))
#plt.plot(reward_HDI(world_standard))
#plt.show()
    
def reward_le_50(world):
    return - (world.le - 50) ** 2


def get_mu_sigma(world, variable):
    """
    Gets mean and standard deviation of all state variables
    """
    data = getattr(world, variable)
    mean = np.mean(data) 
    std = np.std(data)
    return mean, std

def generate_initial(total_runs, variables):
    """ 
    In: 
        total_runs - int: total number of simulations to generate initial data for
        variables  - list[String]: the variables that will be initialised randomly using the standard run
    Out:
        initial_variables - list[dictionary<String,float>]: Dictionary with the initial variables 
    
    Generates initial values taken from a gaussian distribution with mean and varianve decided by the standard run 
    """
    array = []
    for _ in range(total_runs):
        dict = {}
        for variable in variables:
            mu, sigma = get_mu_sigma(world_standard, variable)
            value = np.random.normal(mu, sigma)
            while value < 0:
                value = np.random.normal(mu, sigma)
            dict[variable+"i"]=value 
        array.append(dict)
    return array


def main_loop(reward_func, runs=100):
    """ 
    In: 
        reward_func function: function that takes a world3 object as indata and returns an array of rewards
        runs: how many runs to do
    Returns:
        dataframe with states and reward for that state
    
    Simulates randomized runs of the world3 model without any control. Randomizing input based on the standard run
    50% of the runs will also start at a random year chosen uniformly between 1900 and 2100
    """

    variables = state_variables
    not_time_variables = [var for var in state_variables if var != 'time']
    initial_values = generate_initial(runs, not_time_variables)
    
    df_list = []

    for run in tqdm(range(runs)):
        if run > 0.5 * runs:
            min_year = np.random.randint(1925, 2100)
        else:
            min_year = 1900
        
        world3 = World3(year_max=2100, year_min=min_year)
        world3.set_world3_control()
        world3.init_world3_constants(**initial_values[run])
        world3.init_world3_variables()
        world3.set_world3_table_functions()
        world3.set_world3_delay_functions()
        world3.run_world3(fast=True) # no controls fast is safe here

        # temporary dataframe
        run_df = pd.DataFrame({var: getattr(world3, var) for var in variables})
        run_df["J"] = J_func(reward_func(world3))
        df_list.append(run_df)
    
    df = pd.concat(df_list, ignore_index=True)
    return df


def main():
    chosen_reward = reward_HDI
    df = main_loop(chosen_reward, 500)
    reward_func_name = chosen_reward.__name__
    df.to_parquet(f"data_{reward_func_name}.parquet", index=False)

main()