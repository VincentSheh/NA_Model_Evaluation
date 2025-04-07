from streamer import Streamer
from attack import Attacker
from edgearea import EdgeArea, IDS, VideoServer
import random
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from behavior import generate_attack_intensity, generate_hsmm
from behavior import user_behavior_from_csv
from itertools import permutations


# Attack Intensity Parameters
BASELINE_INTENSITY = 2500 #1500 # Previously 700
fluctuation_intensity = 60
spike_prob = 0.0
spike_intensity = 200
persistence_coeffs = [0.85, 0.15, 0.0]  # Weights for AR(3)
decay_factor = 0.00 # Decay factor to bring intensity down gradually
drop_prob = 0.00  # Probability of a sudden drop in intensity
drop_intensity = 100  # Amount by which intensity drops if drop event occurs


atk_states = ["NoAtk", "bonesi", "goldeneye", "hulk", "bonesi_x_ge"]
# atk_states = ["NoAtk", "bonesi", "bonesi", "bonesi"]
# atk_states = ["NoAtk", "hulk", "hulk", "hulk"]
# atk_duration_means = [5, 3, 2, 4]  # mean duration for each state (Poisson distribution)
# atk_transition_matrix = np.array(  [[0.6, 0.1, 0.15, 0.15],  # No Attack
#                                     [0.1, 0.7, 0.1, 0.1],  # DDoS
#                                     [0.1, 0.1, 0.7, 0.1],  # SYN Flood
#                                     [0.1, 0.1, 0.2, 0.6]])  # HTTP Flood   

atk_duration_means = np.array([10, 3, 6, 8, 5])*30  # mean duration for each state (Poisson distribution)
#Original
atk_transition_matrix = np.array(  [[0.5, 0.2, 0.15, 0.15],  # No Attack
                                    [0.8, 0.2, 0.0, 0.0],  # DDoS
                                    [0.8, 0.0, 0.2, 0.0],  # SYN Flood
                                    [0.8, 0.0, 0.0, 0.2]])  # HTTP Flood   run_for = 10_000

atk_transition_matrix = np.array(  [[0.25, 0.25, 0.25, 0.25],  # No Attack
                                    [0.75, 0.25, 0.0, 0.0],  # DDoS
                                    [0.75, 0.0, 0.25, 0.0],  # SYN Flood
                                    [0.75, 0.0, 0.0, 0.25]])  # HTTP Flood   run_for = 10_000

atk_transition_matrix = np.array([
    [0.1, 0.225, 0.225, 0.225, 0.225],  # No Attack
    [0.7, 0.3, 0.0, 0.0, 0.0],  # bonesi
    [0.7, 0.0, 0.3, 0.0, 0.0],  # bonesi_x_ge
    [0.7, 0.0, 0.0, 0.3, 0.0],  # goldeneye
    [0.7, 0.0, 0.0, 0.0, 0.3],  # hulk
])
# atk_transition_matrix = np.array([
#     [0.2, 0.2, 0.2, 0.2, 0.2],  # No Attack
#     [0.5, 0.5, 0.0, 0.0, 0.0],  # bonesi
#     [0.5, 0.0, 0.5, 0.0, 0.0],  # bonesi_x_ge
#     [0.5, 0.0, 0.0, 0.5, 0.0],  # goldeneye
#     [0.5, 0.0, 0.0, 0.0, 0.5],  # hulk
# ])
# atk_transition_matrix = np.array([
#     [0.2, 0.2, 0.2, 0.2, 0.2],  # No Attack
#     [0.2, 0.8, 0.0, 0.0, 0.0],  # bonesi
#     [0.2, 0.0, 0.8, 0.0, 0.0],  # bonesi_x_ge
#     [0.2, 0.0, 0.0, 0.8, 0.0],  # goldeneye
#     [0.2, 0.0, 0.0, 0.0, 0.8],  # hulk
# ])


# stream_states = [0, 1] #0: Idle, #1: Streaming
# stream_duration_means = [20, 30]  # mean duration for each state (Poisson distribution)
# stream_transition_matrix = np.array([[0.3, 0.7],  # Idle
#                                      [0.8, 0.2]])  # Streaming
# # stream_transition_matrix = np.array([[0.0, 1.0],  # Idle
# #                                      [0.0, 1.0]])  # Streaming


class Environment:
    def __init__(self, run_for=100, cpu_capacity = 3.0, seed=4022):
        
        self.global_qoe_list = []   #Average QOE of all Edge Areas !
        self.user_count_list = np.zeros(run_for)
        self.area_dict = {}
        #TODO Model Streamer Interactions with server (Arrival And Departure)
        self.seed = seed  # Seed for reproducibility
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.cpu_allocated = cpu_capacity
        self.current_timestep = 0  # Initialize timestep counter
        self.run_for = run_for  # Total number of timesteps to simulate
        self.streamer_type = ["biggo", "youtube", "steamtv"]

        
    def initialize_agent(self, num_area, num_streamers, num_attackers, baseline_intensity = BASELINE_INTENSITY):
        # num_area=3
        num_streamers = [num_streamers]*num_area #Depending on num_server
        num_attackers = [1]*num_area #Depending on num_attackers
        # num_streamers = [num_streamers]
        # num_attackers = [num_attackers]
        for i in range(num_area): #Initialize Edge Area Servers
            video_cpu = self.cpu_allocated
            streamers = []
            attackers = []
            aggregate_state_sequence = np.zeros(self.run_for)  
            active_streamers = user_behavior_from_csv(self.streamer_type[i], self.run_for, num_streamers = num_streamers)
            for j in range(num_streamers[i]): #Initialize streamers
                # hsmm_states = generate_hsmm(self.run_for, stream_states, stream_duration_means, stream_transition_matrix)
                # streamers.append(Streamer(j, hsmm_states))     
                # aggregate_state_sequence += hsmm_states
                
                #* Updated
                streamer_state = np.ones_like(active_streamers)
                active_streamers = active_streamers - 1
                streamer_state[active_streamers < 0] = 0
                streamers.append(Streamer(j, streamer_state))     
                aggregate_state_sequence += streamer_state[:self.run_for]
                
            self.user_count_list += aggregate_state_sequence     
                  
            for j in range(num_attackers[i]): #Initialize Attackers
                atk_hsmm_states = generate_hsmm(self.run_for, atk_states, atk_duration_means, atk_transition_matrix, plot=False)
                intensity_sequence = generate_attack_intensity(
                        self.run_for, baseline_intensity, fluctuation_intensity, spike_prob, spike_intensity,
                        persistence_coeffs, decay_factor, drop_prob, drop_intensity
                )   
                
                attackers.append(Attacker(j, intensity_sequence, atk_hsmm_states, edge_area=i))
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
        # global_states = np.zeros((len(self.area_dict), 7)) 
        tb_log = {} # {Edge_Area1: {cpu_allocation, total_user (with offloaded), total_defense}}
        #Variables to for Task Offloading
        streamer_counts = []
        cpu_list = []
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
                atk_config = attack.atk_config
                atk_intensity = attack.intensity
                if atk_intensity > 0:                    
                    atk_intensity = atk_intensity * ids.accuracy[attack.state] #[attack.value["name"]]
                remaining_atks = max(0,atk_intensity - ids.cur_quota) #Current Quota before Utilized by detection
                #* For Logging To Global States
                atk_L = atk_config["L_k"]
                atk_beta_0 = atk_config["beta_0_k"]
                atk_beta_CPU = atk_config["beta_CPU_k"]
                
                    
                #! Moved from calculated_qoe()    
                attack.start(server) #IDS is Called Here
                                                
            for streamer in edge_area['streamers']:
                streamer.forward()
                active_streamer += streamer.state
                
                #! Moved from calculated_qoe()
                if streamer.state == 1:
                    streamer.start_stream(server)               
            global_states[i][0] = active_streamer / 10
            global_states[i][1] = atk_intensity / 2000 #! Max Intensity shouldn't be a constant
            # global_states[i][2] = atk_L
            # global_states[i][3] = atk_beta_0
            # global_states[i][4] = atk_beta_CPU
            streamer_counts.append(active_streamer)
            cpu_list.append(server.cpu_allocated)
            # remaining_quotas.append((ids.cur_quota, ids.accuracy)) #Get the Remaining Defense Quota of each edge area
            remaining_quotas.append((ids.cur_quota, ids.accuracy)) #Get the Remaining Defense Quota of each edge area
            remaining_intensity.append(remaining_atks)

                        
        #* Perform Task Offloading for Streamers
        #TODO: Get the area that is in reasonable range
        max_idx = np.argmax(np.array(streamer_counts)/np.array(cpu_list))    
        min_idx = np.argmin(np.array(streamer_counts)/np.array(cpu_list))        
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
                    print(streamer_counts)
                    print(self.area_dict[max_idx]['area'].server.active_streamers)                    
                    break
            max_idx = np.argmax(streamer_counts)
            min_idx = np.argmin(streamer_counts)
        
        #* Perform Task Offloading for Defense Task 
        
        #* Method 1 - Offload Aiming Balance
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
                    
        for i, ids_info in enumerate(remaining_quotas): #For different ids
            cur_quota, accuracy = ids_info #Current IDS Defense Quota
            # print("Area:", i)
            for idx in idx_to_reduce: #For different Atks
                if cur_quota <= 0:
                    break
                server = self.area_dict[idx]['area'].server
                atk_config = server.attack_config_list[0]     
                allocation = min(min(cur_quota, defense_allocation[idx]), atk_config["new_intensity"])
                cur_quota -= allocation
                defense_allocation[idx] -= allocation
                # total_allocation+=allocation

                atk_config["new_intensity"] -= allocation*accuracy[atk_config["name"]]
                if atk_config["new_intensity"] < 0: 
                    print("WARNING!: Over allocation of Defense Resource", atk_config["new_intensity"])                       
            # new_atk_list.append(server.attack_config_list[0])
        # print(total_allocation, defense_allocation)
        # print(new_atk_list)
        #TODO: Method 2 - Offload to nearest Edge Area
        
        #* Log into Tensor Board
        for i, edge_area in self.area_dict.items():
            server = edge_area['area'].server
            ids = edge_area['area'].ids    
            atk_config = server.attack_config_list[0]

            tb_log[f"Edge_Area_{i}/video_cpu"] = server.cpu_allocated
            # tb_log[f"Edge_Area_{i}/total_users"] = streamer_counts[i]
            tb_log[f"Edge_Area_{i}/total_users"] = len(server.active_streamers)
            tb_log[f"Edge_Area_{i}/final_atk_intensity"] = atk_config["new_intensity"]
            # break
        
        return tb_log, global_states, atk_config

    def update_timestep(self):
        self.start_new_timestep()
        self.calculate_qoe()
        # self.resource_decision()
        # self.start_new_timestep_controled()
    def start_new_timestep_controlled(self):
        max_qoe = 0
        best_cpu = [0,0,0]
        for cpu_list in permutations(np.arange(0.5, self.cpu_allocated, 0.5), 3): # Try all different CPU Configuration
            for i, edge_area in self.area_dict.items(): # Find the best area
                edge_area["area"].server.cpu_allocated = cpu_list[i]
                edge_area["area"].ids.cpu_allocated = 6.0 - cpu_list[i]
            # Start Timestep
            self.start_new_timestep()
            # Obtain QoE and Update Max Configurations
            for i, edge_area in self.area_dict.items():
                server = edge_area['area'].server            
            mean_qoe = np.mean([edge_area["area"].server.calculate_qoe() for i, edge_area in self.area_dict.items()])
            if max_qoe < mean_qoe:
                max_qoe = mean_qoe
                best_cpu = cpu_list
            # Reset the Timestep and Repeat
            for i, edge_area in self.area_dict.items():
                server = edge_area['area'].server
                server.forward()
                ids = edge_area['area'].ids    
                ids.forward()

                for attack in edge_area['attackers']: #Initiate the Attack Task
                    attack.time_elapsed-=1
                for streamer in edge_area['streamers']:
                    streamer.time_elapsed-=1
                    
        #? Start New Timestep with the Best CPU Configuration
        for i, edge_area in self.area_dict.items():
            edge_area["area"].server.cpu_allocated = best_cpu[i]
            edge_area["area"].ids.cpu_allocated = 6.0 - best_cpu[i]
        #* Return back to the step function and run normally
        return self.start_new_timestep()
        
        
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
                ax3.set_xlim(max(0), len(qoe_list))      

            except:
                pass
            ax1.set_xlim(0, len(qoe_list))     
            ax2.set_xlim(0, len(qoe_list))
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