"""
Generates the needed data of J to train the neural network model, saves a dataframe with states and rewards as a parquet file
"""
import numpy as np
from sympy import N
from pyworld3 import World3
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

show_progress = True
state_variables = ["p1", "p2", "p3", "p4", "ic", "sc", "al", "pal", "uil", "lfert", "ppol", "nr", "time"]

# Standard run used for randomizing initial state
world_standard = World3(year_max=2100)
world_standard.set_world3_control()
world_standard.init_world3_constants()
world_standard.init_world3_variables()
world_standard.set_world3_table_functions()
world_standard.set_world3_delay_functions()
world_standard.run_world3(fast=False)

def write_out(message):
    if show_progress:
        print(message)

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

def reward_pop(world, k=None):
    # reward function, trying simple with population
    if k is not None:
        return world.pop[k]
    return world.pop

def reward_le(world, k=None):
    if k is not None:
        return world.le[k]
    return world.le

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

def run_random_runs(reward_func, runs=100):
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

    for run in tqdm(range(runs), disable=not show_progress):
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

class neuralNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        # 2 hidden layers
        super(neuralNet, self).__init__()
        self.input_layer = nn.Linear(in_features=in_dim+1, out_features=64)
        self.hidden_layer1 = nn.Linear(in_features=64, out_features=64) 
        self.hidden_layer2 = nn.Linear(in_features=64, out_features=32)
        self.outLayer = nn.Linear(in_features=32, out_features=out_dim)

    def forward(self, x):
        # forward pass with relu activation function
        ones = torch.ones((x.shape[0], 1))
        x = torch.cat((x, ones), dim=1)
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.outLayer(x)
        return x
    
def train_network(df):
    X = df.drop(columns=["J"]).to_numpy()
    J = df["J"].to_numpy().reshape(-1,1)

    X_train, X_test, J_train, J_test = train_test_split(X, J, test_size=0.2, random_state=42)

    # normalizing
    X_normalizer = StandardScaler()
    X_train = X_normalizer.fit_transform(X_train)
    X_test = X_normalizer.transform(X_test)
    J_normalizer = StandardScaler()
    J_train = J_normalizer.fit_transform(J_train)
    J_test = J_normalizer.transform(J_test)

    # turning into pytorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    J_train = torch.tensor(J_train, dtype=torch.float32)
    J_test = torch.tensor(J_test, dtype=torch.float32)

    model = neuralNet(X_train.shape[1], 1)
    loss_func = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Optimizing weights

    epochs = 300
    losses = np.zeros((epochs,1))
    model.train()

    for epoch in tqdm(range(epochs), disable=not show_progress):
        J_pred = model.forward(X_train)
        loss = loss_func(J_pred, J_train)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        losses[epoch] = loss.item()

    model.eval()

    with torch.no_grad():
        J_pred = model.forward(X_test)
        loss = loss_func(J_pred, J_test)
    
    write_out("Training complete error on test set:" + str(loss.item()))
    return model, X_normalizer

def nn_first_func(model, world, k, normalizer):
    """ 
    model: neural network model
    world: World3 object
    k: current iteration

    Returns
        J_hat
    """
    model.eval()
    state = np.array([getattr(world, var)[k] for var in state_variables])
    state = normalizer.transform(state.reshape(1, -1))
    state = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        J_ = model.forward(state)
    return J_.item()

def loop0(world):
    world.redo_loop = True
    while world.redo_loop:  # unsorted updates at initialization only
        world.redo_loop = False
        world.loop0_population()
        world.loop0_capital()
        world.loop0_agriculture()
        world.loop0_pollution()
        world.loop0_resource()
    
def generate_fioac_control_values():
    return np.linspace(0,1,50)

def get_fioac_control(world3, k, steps, J_hat, reward_func):
    """ 
    In:
        world3: pyworld3 simulation
        k: current iteration
        steps: how many steps to look ahead
        J_hat: Approximation of J function
        reward_func: the reward J is based on
    Returns:
        control: control value
    """
    # _self._loopk_world3_fast(k_ - 1, k_, k_ - 1, k_)  # sorted updates
    n = world3.n
    steps = min(steps,n-k)
    control = 0.43
    reward = 0
    world3.fioac_control = lambda _: control
    for k_new in range(k,k+steps):
        world3._loopk_world3_fast(k_new-1, k_new, k_new-1, k_new)
        if k_new != k+steps-1:
            reward += reward_func(world3, k_new)
        else:
            reward += J_hat(world3, k_new)
    best_J = reward
    fioac_controls = generate_fioac_control_values()

    for val in fioac_controls:
        reward = 0
        world3.fioac_control = lambda _: val
        for k_new in range(k,k+steps):
            world3._loopk_world3_fast(k_new-1, k_new, k_new-1, k_new)
            if k_new != k+steps-1:
                reward += reward_func(world3, k_new)
            else:
                reward += J_hat(world3, k_new)
        if reward > best_J:
            best_J = reward
            control = val
    return control

def optimize_run(model, normalizer, reward, initial_values):
    world_control = World3(year_max=2100)
    world_control.set_world3_control()
    world_control.init_world3_constants(**initial_values)
    world_control.init_world3_variables()
    world_control.set_world3_table_functions()
    world_control.set_world3_delay_functions()
    loop0(world_control)
    for k in range(1,world_control.n):
        if k % 10 == 0:
            J_hat = lambda world, k: nn_first_func(model, world, k, normalizer)
            control_val = get_fioac_control(world_control, k, 10, J_hat, reward)
            world_control.fioac_control = lambda _: control_val
            world_control._loopk_world3_fast(k -1, k, k-1, k)
        else:
            world_control._loopk_world3_fast(k -1, k, k-1, k)
    return world_control

def random_optimized(model, normalizer, chosen_reward, runs=100):
    """
    In:
        model: model for J function
        normalizer: normalizer that was used to 
    """
    df_list = []
    not_time_variables = [var for var in state_variables if var != 'time']
    initial_values = generate_initial(runs, not_time_variables)
    for run in tqdm(range(runs), disable=(not show_progress)):
        world = optimize_run(model, normalizer, chosen_reward, initial_values[run])
        run_df = pd.DataFrame({var: getattr(world, var) for var in state_variables})
        run_df["J"] = J_func(chosen_reward(world))
        df_list.append(run_df)
    df = pd.concat(df_list, ignore_index=True)
    return df

def main():
    chosen_reward = reward_le
    write_out("Generating first cycle training data")
    first_cycle_runs = 1000
    df = run_random_runs(chosen_reward, first_cycle_runs)
    write_out("Training neural network: ")
    model, normalizer = train_network(df)
    write_out("Generating second data: ")
    second_cycle_runs = 100
    df = random_optimized(model, normalizer, chosen_reward, second_cycle_runs)
    reward_func_name = chosen_reward.__name__
    df.to_parquet(f"datasets/data_{reward_func_name}_v2.parquet", index=False)
    write_out("Done! :)")

main()
