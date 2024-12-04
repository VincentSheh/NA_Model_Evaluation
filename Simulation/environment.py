from streamer import Streamer
from attack import Attacker
from edgearea import EdgeArea, IDS, VideoServer
import random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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

    return intensity



# atk_states = ["NoAtk", "DDoS", "SYN Flood", "HTTP Flood"]
atk_states = ["NoAtk", "bonesi", "goldeneye", "hulk"]
# atk_states = ["NoAtk", "bonesi", "bonesi", "bonesi"]
# atk_states = ["NoAtk", "hulk", "hulk", "hulk"]
atk_duration_means = [5, 3, 2, 4]  # mean duration for each state (Poisson distribution)
atk_transition_matrix = np.array(  [[0.6, 0.1, 0.15, 0.15],  # No Attack
                                    [0.1, 0.7, 0.1, 0.1],  # DDoS
                                    [0.1, 0.1, 0.7, 0.1],  # SYN Flood
                                    [0.1, 0.1, 0.2, 0.6]])  # HTTP Flood   

stream_states = [0, 1] #0: Idle, #1: Streaming
stream_duration_means = [20, 30]  # mean duration for each state (Poisson distribution)
stream_transition_matrix = np.array([[0.3, 0.7],  # Idle
                                     [0.8, 0.2]])  # Streaming
# stream_transition_matrix = np.array([[0.0, 1.0],  # Idle
#                                      [0.0, 1.0]])  # Streaming

# Attack Intensity Parameters
baseline_intensity = 700
fluctuation_intensity = 60
spike_prob = 0.0
spike_intensity = 200
persistence_coeffs = [0.85, 0.15, 0.0]  # Weights for AR(3)
decay_factor = 0.00 # Decay factor to bring intensity down gradually
drop_prob = 0.00  # Probability of a sudden drop in intensity
drop_intensity = 100  # Amount by which intensity drops if drop event occurs

def generate_hsmm(n_steps, states, duration_means, transition_matrix, plot=False):
    def simulate_duration(mean_duration):
        return np.random.poisson(mean_duration)    
    def select_next_state(current_state):
        return np.random.choice(states, p=transition_matrix[states.index(current_state)])

     
    current_state = random.choice(states)  # randomly choose initial state
    state_sequence = [current_state]  # keep track of the states over time
    duration_in_current_state = simulate_duration(duration_means[states.index(current_state)])
    time_in_state = 0  # initialize time spent in the current state
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
        # Plot the sequence of states over time
        plt.figure(figsize=(20, 3))
        plt.plot(range(n_steps), state_sequence, marker='o')
        plt.yticks(range(len(states)), states)
        plt.title('Hidden Semi-Markov Model')
        plt.xlabel('Time Step')
        plt.ylabel('Attack Variant')
        plt.grid(True)
        plt.show()        

    return state_sequence



class Environment:
    def __init__(self, run_for=100, cpu_capacity = 3.0, seed=4022):
        
        self.global_qoe_list = []   #Average QOE of all Edge Areas !
        self.user_count_list = np.zeros(run_for)
        self.cpu_decision_list = []
        self.atk_intensity_list = [] #TODO: Change to Global
        self.area_dict = {}
        #TODO Model Streamer Interactions with server (Arrival And Departure)
        self.seed = seed  # Seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.cpu_allocated = cpu_capacity
        self.current_timestep = 0  # Initialize timestep counter
        self.run_for = run_for  # Total number of timesteps to simulate
    def initialize_agent(self, num_area, num_streamers, num_attackers):
        # num_area=3
        num_streamers = [6]*num_area #Depending on num_server
        num_attackers = [1]*num_area #Depending on num_attackers
        # num_streamers = [num_streamers]
        # num_attackers = [num_attackers]
        for i in range(num_area): #Initialize Edge Area Servers
            video_cpu = self.cpu_allocated
            streamers = []
            attackers = []
            aggregate_state_sequence = np.zeros(self.run_for)  
            for j in range(num_streamers[i]): #Initialize streamers
                hsmm_states = generate_hsmm(self.run_for, stream_states, stream_duration_means, stream_transition_matrix)

                streamers.append(Streamer(j, hsmm_states))     
                aggregate_state_sequence += hsmm_states
            self.user_count_list += aggregate_state_sequence     
                  
            for j in range(num_attackers[i]): #Initialize Attackers
                atk_hsmm_states = generate_hsmm(self.run_for, atk_states, atk_duration_means, atk_transition_matrix, plot=False)
                intensity_sequence = generate_attack_intensity(
                        self.run_for, baseline_intensity, fluctuation_intensity, spike_prob, spike_intensity,
                        persistence_coeffs, decay_factor, drop_prob, drop_intensity
                )   
                
                attackers.append(Attacker(j, intensity_sequence, atk_hsmm_states))
            # Define the Streamer/Attacker-Server Relation on Environment for Task Offloading
            self.area_dict[i] = {
                                          "area": EdgeArea(video_cpu),
                                        #   "server": VideoServer(video_cpu), 
                                          "streamers": streamers, #TODO: When Task Offload, put streamers on seperate list
                                          "attackers": attackers, #Attack Workers
                                          #For Plotting
                                          "qoe_list": [],
                                          "user_count_list": aggregate_state_sequence,
                                          "attack_states": atk_hsmm_states,
                                          "intensity_sequence": intensity_sequence,
                                          }
    def calculate_qoe(self): # Run One Timestep and Calculate QoE
        total_qoe = 0
        num_users = 0
        area_qoe = []
        for _, edge_area in self.area_dict.items():
            #! Moved to start_new_timestep()
            cur_server, streamers, attackers = edge_area['area'].server, edge_area['streamers'], edge_area['attackers']
            # for streamer in streamers: #Initiate the Streaming Task
            #     if streamer.state == 1:
            #         streamer.start_stream(cur_server) 
            # for attack in attackers: #Initiate the Attack Task
            #     attack.start(cur_server) #IDS is Called Here
                
            #Update QoE Records
            cur_qoe = cur_server.calculate_qoe() 
            edge_area["qoe_list"].append(cur_qoe)
            total_qoe += cur_qoe
            area_qoe.append(cur_qoe)
        average_qoe = total_qoe / len(self.area_dict) 
        self.global_qoe_list.append(average_qoe)
        return area_qoe

    def resource_decision(self): #TODO: Change to Control Node
        for _, edge_area in self.area_dict.items():
            server = edge_area['area'].server
            # server.query_resource_decision()
            
                    

    def start_new_timestep(self):
                    
        #? Update Agent
        #TODO: Move to Orchestrator Class
        global_states = np.zeros((len(self.area_dict), 4)) 
        
        #Variables to for Task Offloading
        streamer_counts = []
        remaining_intensity = []
        remaining_quotas = []
        # Assign Users and Attackers to Nearest Edge Server
        for i, edge_area in self.area_dict.items():
            atk_intensity = active_streamer = 0    
            server = edge_area['area'].server
            ids = edge_area['area'].ids
            edge_area['area'].forward()
            
            for attack in edge_area['attackers']: #Initiate the Attack Task
                attack.forward() #IDS is Called Here     
                atk_state = attack.atk_type
                atk_intensity = attack.intensity
                if atk_intensity > 0:                    
                    atk_intensity = atk_intensity * ids.accuracy[atk_state.value["name"]]
                remaining_atks = max(0,atk_intensity - ids.cur_quota) #Current Quota before Utilized by detection
                    
                #! Moved from calculated_qoe()    
                attack.start(server) #IDS is Called Here
                                                
            for streamer in edge_area['streamers']:
                streamer.forward()
                active_streamer += streamer.state
                
                #! Moved from calculated_qoe()
                if streamer.state == 1:
                    streamer.start_stream(server)               
            global_states[i][0] = atk_intensity / 2000 #! Max Intensity shouldn't be a constant
            global_states[i][1] = active_streamer / 6
            
            streamer_counts.append(active_streamer)
            # remaining_quotas.append((ids.cur_quota, ids.accuracy)) #Get the Remaining Defense Quota of each edge area
            remaining_quotas.append((ids.cur_quota, ids.accuracy)) #Get the Remaining Defense Quota of each edge area
            remaining_intensity.append(remaining_atks)

                        
        #? Perform Task Offloading for Streamers
        #TODO: Get the area that is in reasonable range
        #TODO: Normalize according to the CPU Capacity
        max_idx = np.argmax(streamer_counts)
        min_idx = np.argmin(streamer_counts)        
        while(len(streamer_counts)>1 and streamer_counts[max_idx] > streamer_counts[min_idx] + 1):
            # print(streamer_counts)

            if streamer_counts[max_idx] > 0:
                try:
                    self.area_dict[max_idx]['area'].server.active_streamers.pop() #TODO: Remove the lowest workload
                    self.area_dict[min_idx]['area'].server.active_streamers.append(-1)
                    streamer_counts[max_idx]-=1
                    streamer_counts[min_idx]+=1
                    
                except IndexError as e:
                    print(f"Error during task offloading: {e}")
                    break
            max_idx = np.argmax(streamer_counts)
            min_idx = np.argmin(streamer_counts)
        # print("FINAL", streamer_counts)
        
        #? Perform Task Offloading for Defense Task 
        # print("Previously: ", remaining_intensity, remaining_quotas)
        #Method 1 - Offload Aiming Balance
        total_quota = sum([quota[0] for quota in remaining_quotas]) #remaining_quotas is tuple of (quota, acc)
        total_atk = sum(remaining_intensity)
        # Get z = D/n'
        z_prime = (total_atk - total_quota)/len(remaining_intensity)
        a_prime = np.array(remaining_intensity)
        while True:
            defense_allocation = np.zeros_like(remaining_intensity)
            # Find indices where remaining intensity exceeds z_prime
            idx_to_reduce = np.where((a_prime >= z_prime) & (a_prime > 0))[0]
            if len(idx_to_reduce) == 0:
                break  # No areas left to reduce
            z_prime = max(0,(sum(a_prime[idx_to_reduce]) - total_quota) / len(idx_to_reduce))# Update z' considering only these indices
            defense_allocation[idx_to_reduce] = a_prime[idx_to_reduce] - z_prime# Update defense allocation for these indices
            # Check termination condition
            if np.all(defense_allocation >= 0):
                break
            
        total_allocation=0
        new_atk_list = []
        # print("After: ", a_prime, defense_allocation)
        # for cur_quota, accuracy in self.area_dict.items():
        #     ids = edge_area['area'].ids
        for i, ids_info in enumerate(remaining_quotas): #For different ids
            cur_quota, accuracy = ids_info
            # print("Area:", i)
            for idx in idx_to_reduce: #For different Atks
                if cur_quota <= 0:
                    break
                allocation = min(cur_quota, defense_allocation[idx])
                cur_quota -= allocation
                defense_allocation[idx] -= allocation
                total_allocation+=allocation
                server = self.area_dict[idx]['area'].server
                atk_config = server.attack_config_list[0]
                atk_config["new_intensity"] -= allocation*accuracy[atk_config["name"]]
            new_atk_list.append(server.attack_config_list[0])
        # print(total_allocation, defense_allocation)
        # print(new_atk_list)
        #TODO: Method 2 - Offload to nearest Edge Area
        self.current_timestep += 1
        return global_states, atk_state

    def update_timestep(self):
        self.start_new_timestep()
        self.calculate_qoe()
        # self.resource_decision()
        # self.start_new_timestep_controled()
    
    def plot_qoe(self, ma_window_size = 10):
        def calculate_moving_average(data, window_size=10):
            """
            Calculates the moving average of the given data over a specified window size.
            """
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')              
        for _, edge_area in self.area_dict.items():
            qoe_list = edge_area["qoe_list"]
            user_count_list = edge_area["user_count_list"]
            intensity_sequence = edge_area["intensity_sequence"] #? Not Used
            attack_states = edge_area["attack_states"]
            qoe_moving_avg = calculate_moving_average(np.array(qoe_list), ma_window_size)
            print("Mean QoE", np.mean(qoe_list[:]))
            # Plot QoE and number of users over time
            fig, ax1 = plt.subplots(figsize=(20, 6))

            # Plot QoE
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("QoE", color="tab:blue")
            ax1.plot(qoe_list, label="QoE", color="tab:blue", linewidth=1.5, marker="o")
            # ax1.plot(range(ma_window_size-1,len(self.global_qoe_list)), qoe_moving_avg, label=f"QoE MA {ma_window_size}", color="tab:green", linewidth=3)
            ax1.set_ylim(-0.05,1.05)
            ax1.tick_params(axis="y", labelcolor="tab:green")
            ax1.legend()

            # Create a second y-axis to plot number of users
            ax2 = ax1.twinx()
            ax2.set_ylabel("Number of Active Users", color="tab:orange")
            ax2.plot(user_count_list, label="Active Users", color="tab:orange", linewidth=2)
            ax2.tick_params(axis="y", labelcolor="tab:orange")
            ax2.legend()
            
            # Overlay state sequence if provided
            try:
                state_sequence = attack_states
                ax3 = ax1.twinx()  # Create a third y-axis for state sequence
                ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
                ax3.set_yticks(range(len(set(state_sequence))))  # Set y-ticks based on unique states
                ax3.set_yticklabels(set(state_sequence), rotation=90)  # Label the states
                
                ax3.plot(state_sequence, label="HSMM States", color="tab:red", marker='o', linestyle='None')
                ax3.set_yticks(range(len(atk_states)))  # Set y-ticks based on the number of states
                ax3.set_yticklabels(atk_states, rotation=90)  # Set the attack state labels
                ax3.set_ylabel("Attack State", color="tab:red")
                ax3.tick_params(axis="y", labelcolor="tab:red")
                ax3.legend(loc="upper center")        

                ax3.set_xlim(max(0, len(qoe_list) - 2000), len(qoe_list))      

            except:
                pass
            ax1.set_xlim(max(0, len(qoe_list) - 2000), len(qoe_list))      
            ax2.set_xlim(max(0, len(qoe_list) - 2000), len(qoe_list))      
            # Add legends and title
            fig.suptitle("QoE and Number of Active Users Over Time")
            fig.tight_layout()
            plt.legend()
            plt.show()        
            print(len(qoe_list))

    def run(self, intensity=0):
        for _ in range(self.run_for):
            self.update_timestep()            
            # self.calculate_qoe()
            # self.start_new_timestep_controled(intensity)            
        print("Simulation Complete.")
        return self.global_qoe_list