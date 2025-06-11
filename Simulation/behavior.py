import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

NUM_USERS = 10
def generate_attack_intensity(time_steps, baseline, fluctuation, spike_prob, spike_intensity, 
                                             persistence_coeffs, decay_factor, drop_prob, drop_intensity):
    # Initialize intensity array with the first three values set to baseline
    intensity = np.full(time_steps, baseline)
    
    # Generate time series with AR(3) process with decay and drop probability
    for t in range(3, time_steps):
        # AR(3) process with decay factor
        intensity[t] = (
            persistence_coeffs[0] * intensity[t - 1] +
            persistence_coeffs[1] * intensity[t - 2] +
            persistence_coeffs[2] * intensity[t - 3] +
            (1 - sum(persistence_coeffs)) * baseline +
            np.random.normal(0, fluctuation)
        )
        
        # Apply decay factor to gradually reduce intensity toward baseline
        # intensity[t] *= (1 - decay_factor)
        
        # Add random spikes
        if np.random.rand() < spike_prob:
            intensity[t] += spike_intensity
        
        # Randomly drop the intensity with a certain probability
        if np.random.rand() < drop_prob:
            intensity[t] -= drop_intensity
            # Ensure the intensity does not go below zero
            intensity[t] = max(intensity[t], baseline)
        intensity[t] = max(0, intensity[t])
        intensity[t] = min(2000, intensity[t])
    div_timesteps = np.ceil(time_steps/30).astype(int)
    intensity = np.repeat(intensity[:div_timesteps], 30)

    return intensity

def user_behavior_from_csv(streamer, run_for, plot=False, num_streamers = NUM_USERS):
    if streamer == "biggo":
        order, seasonal_order = (2,0,1), (1, 0, 1, 24) #Biggo and YT
    elif streamer == "youtube":
        order, seasonal_order = (2,0,1), (1, 0, 1, 24) #Biggo and YT
    elif streamer == "steamtv":
        order, seasonal_order = (1,0,1), (0, 0, 0, 24) #SteamTV
    ts_df = pd.read_csv("./webscrape_data/channels/" + streamer + ".csv", index_col="Hour")
    flattened_df = ts_df.melt(ignore_index=False, var_name="Day", value_name="Channels")
    
    # Create a timeline index (e.g., Hour_1 to Hour_168)
    flattened_df["Timeline"] = range(1, len(flattened_df) + 1)
    
    # Fit Sarima
    flattened_df = flattened_df.set_index("Timeline")
    # Fit the SARIMA model
    model = SARIMAX(flattened_df["Channels"], order=order, seasonal_order=seasonal_order,trend='n', enforce_stationarity=False, enforce_invertibility=False)
    sarima_model = model.fit(disp=False)
# def generate_values(sarima_model, plot=False):
    simulated_values = sarima_model.simulate(
        nsimulations=120, 
        # initial_state=sarima_model.predicted_state[:, -1], 
        initial_state=sarima_model.predicted_state[:, -1], 
        # repetitions= max(1,run_for//((120-10)*30) )
        repetitions= 5
    ).values    
    
    min_value = np.min(simulated_values)
    simulated_values -= min_value
    if min_value < 0: #! Not Needed?
        simulated_values -= min_value
    simulated_values = np.array(simulated_values)
    simulated_values = simulated_values / np.max(simulated_values) * num_streamers + 1
    simulated_values = np.round(simulated_values)    
    simulated_values = np.clip(simulated_values, 0,10)
    
    # min_value = np.min(simulated_values)
    # if min_value < 0:
    #     simulated_values -= min_value
    # simulated_values = np.array(simulated_values)
    # simulated_values = simulated_values / np.max(simulated_values) * 6
    # simulated_values = np.round(simulated_values)    
    
    final_values = np.repeat(simulated_values.T.flatten(), 30)

    if plot:
        # Plot the adjusted series
        plt.figure(figsize=(15, 5))
        # plt.plot(flat_youtube["Timeline"], flat_steamtv["Channels"], label="Actual Data")
        plt.plot(final_values)
        plt.title("SARIMA Simulated Time Series (First Sample)")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()    
    
    
    return final_values
    


def generate_hsmm(n_steps, states, duration_means, transition_matrix, plot=False):
    def simulate_duration(mean_duration):
        return np.random.poisson(mean_duration)
    
    def select_next_state(current_state):
        return np.random.choice(states, p=transition_matrix[states.index(current_state)])

    # Initialize variables
    current_state = random.choice(states)  # Randomly choose initial state
    state_sequence = [current_state]       # Track states over time
    duration_in_current_state = simulate_duration(duration_means[states.index(current_state)])
    time_in_state = 0  # Initialize time spent in the current state

    # Simulate HSMM process with "No Attack" state
    for t in range(1, n_steps):
        if time_in_state < duration_in_current_state:
            # Stay in the current state
            state_sequence.append(current_state)
            time_in_state += 1
        else:
            # Transition to the next state
            current_state = select_next_state(current_state)
            state_sequence.append(current_state)
            duration_in_current_state = simulate_duration(duration_means[states.index(current_state)])
            time_in_state = 1  # Reset time in the new state
    
    if plot:
        # Map state names to integers for plotting
        state_to_int = {state: idx for idx, state in enumerate(states)}
        int_sequence = [state_to_int[state] for state in state_sequence]

        # Plot the sequence of states over time
        plt.figure(figsize=(20, 3))
        plt.plot(range(n_steps), int_sequence, marker='o', linestyle='-', markersize=4)
        plt.yticks(range(len(states)), states)  # Map integers back to state names
        plt.title('Hidden Semi-Markov Model')
        plt.xlabel('Time Step')
        plt.ylabel('Attack Variant')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    return state_sequence


