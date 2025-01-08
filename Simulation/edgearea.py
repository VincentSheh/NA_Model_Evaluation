import json
import numpy as np
import pandas as pd
from resource_allocation import QLearningAgent
max_cpu = 6.0
with open("DARA.json", "r") as file:
    best_approx_decision_dict = json.load(file)
    
    
def logistic(x, impact_config={}, L_base=0.97, beta_0=-2.27, beta_CPU_base=4.8802):
    # Use default logistic function if no impact configuration is provided
    if impact_config == {}:
        return L_base / (1 + np.exp(-(beta_0 + beta_CPU_base * x)))

    # Define exponential saturation adjustment for intensity effect
    #! intensity_effect = 1 - np.exp(-impact_config["alpha"] * impact_config["new_intensity"])
    intensity_effect = impact_config["alpha"] * impact_config["new_intensity"]/2000

    # Adjust parameters based on intensity effect
    L = L_base * (1 + impact_config["L_k"] * intensity_effect)  # Decrease L with intensity effect
    beta_0 = beta_0 * (1 + impact_config["beta_0_k"] * intensity_effect)  # Adjust beta_0 similarly
    beta_CPU = beta_CPU_base * (1 + impact_config["beta_CPU_k"] * intensity_effect)  # Adjust beta_CPU similarly

    # Return adjusted logistic function value
    return L / (1 + np.exp(-(beta_0 + beta_CPU * x)))    


def discrete_approx_decision(intensity, n_streamer):
    intensity_interval = 200
    closest_intensity = round(intensity/intensity_interval)*intensity_interval
    return best_approx_decision_dict[str((closest_intensity, n_streamer))]
    
class EdgeArea:
    def __init__(self, video_cpu):
     
        self.vim = VIM(self)
        self.ids = IDS(self, max_cpu-video_cpu)
        self.server = VideoServer(self, video_cpu)
        self.current_timestep = 0
    def forward(self):
        self.current_timestep+=1
        # self.vim.forward(self.server.cur_info)
        self.server.forward()
        self.ids.forward()
        
        
        
class IDS(object):
    def __init__(self,area,cpu):
        self.area = area
        self.cpu_allocated = cpu  # Initial CPU allocation for the IDS
        # self.processing_speed = {}  # Processing speed for each attack type or variant
        #! self.processing_speed
        self.processing_speed = lambda x: 300*x - 100  # Minimum CPU = 0.5 #TODO: Add Noise and Accelerate for Different Types 
        # self.accuracy = {"bonesi": 0.95, "goldeneye": 0.8, "hulk": 1.0}  # Accuracy for different attack variants
        self.accuracy = {"bonesi": 1.0, "goldeneye": 1.0, "hulk": 1.0}  # Accuracy for different attack variants
        self.cur_quota = self.processing_speed(self.cpu_allocated)

    def detect(self, attack):
        reduced_intensity = self.cur_quota * self.accuracy[attack["name"]]
        attack_intensity = attack["old_intensity"]
        if reduced_intensity < attack_intensity: #If can't detect all of it
            attack_intensity -= reduced_intensity 
            self.cur_quota = 0
        else:
            attack_intensity = 0
            self.cur_quota -= attack_intensity / self.accuracy[attack["name"]]
        return attack_intensity
        # else:
        #     print(f"Unknown attack variant '{attack}' - detection failed.") #TODO: No Detection fail
        #     return False

    def train(self, attack_type):
        # Check if attack type is already in the accuracy dictionary
        if attack_type not in self.accuracy:
            self.accuracy[attack_type] = 0.5  # Start with base accuracy for new types
        # Increase accuracy for the given attack type
        self.accuracy[attack_type] = min(1.0, self.accuracy[attack_type] + 0.1)
        print(f"Trained on '{attack_type}'. New accuracy: {self.accuracy[attack_type]:.2f}")

    def forward(self):
        #Update CPU and Processing Speed
        self.cur_quota = self.processing_speed(self.cpu_allocated)
        
            
    def update_defense_factor(self, attack_type, factor):
        self.defense_factor[attack_type] = factor
        print(f"Updated defense factor for '{attack_type}' to {factor:.2f}")            

    def add_attack_variant(self, variant_name):
        if variant_name not in self.accuracy:
            self.accuracy[variant_name] = 0.5  # Initial accuracy for the new variant
            self.processing_speed[variant_name] = self.cpu_allocated  # Initial processing speed
            self.defense_factor[variant_name] = 1.0  # Default defense factor
            print(f"Added new attack variant '{variant_name}' with default settings.")
        else:
            print(f"Attack variant '{variant_name}' already exists.")


class VideoServer(object): # Edge Area Server
    def __init__(self, area, cpu_allocated):
        self.area = area
        self.cpu_allocated = cpu_allocated  # Total CPU capacity of the server        
        # self.ids = [IDS(6.0 - self.cpu_allocated)]
        self.current_cpu_usage = 0  # Current CPU usage
        self.active_streamers = []  # ! Use Priority Queue to sort the load
        self.active_attackers = []
        # self.qoe_vs_n_user_params = {
        #     1: [0.97, -2.2782716980482065, 4.8802368476587255],
        #     2: [0.97, -3.8873078667967476, 5.175670954212056],
        #     3: [0.97, -4.4959769724864636, 4.081147830542201],
        #     4: [0.97, -3.3377765392197962, 2.5431386672697025],
        #     5: [0.97, -4.186919060489949, 2.3893219975556157],
        #     6: [0.97, -5.935630098896784, 2.56309746149014],
        # }
        self.qoe_vs_n_user_params = {
            1:  [0.9991619042014769,-2.8026108493117943,5.668093271506809],
            2:  [0.9999999996309884,-2.0961944923922555,1.998039343427015],
            3:  [0.9999999999739316,-2.2265292326900163,2.002895418416976],
            4:  [0.9965690849479469,-2.8152311397077043,2.004385398593321],
            5:  [0.9945007296615602,-3.147645008537425,2.038873125719032],
            6:  [0.9800000000000001,-2.717441897354403,1.5195963820348175],
            7:  [0.9800000000000034,-2.6831785348682815,1.3106707068703483],
            8:  [0.9800000000000001,-2.7938552902013924,1.2722296076571054],
            9:  [0.9967234163504595,-3.19640794164371,1.3140589747660822],
            10: [0.9854243346750624,-3.224343461342699,1.2811557634774904],
        }
        self.attack_config_list = []
        self.std_matrix = [ # [User] [CPU * 2]
                    [0.0902, 0.1282, 0.0794, 0.0151, 0.0227, 0.0241, 0.0221, 0.0205, 0.0205, 0.0205, 0.0205],
                    [0.0050, 0.2217, 0.1078, 0.0451, 0.1111, 0.0493, 0.0493, 0.0493, 0.0493, 0.0493, 0.0493 ],
                    [0.0050, 0.1800, 0.2824, 0.2959, 0.0377, 0.1578, 0.1578, 0.0964, 0.0364, 0.0364, 0.0348 ],
                    [0.0050, 0.0000, 0.2355, 0.3006, 0.2886, 0.1364, 0.0975, 0.0629, 0.0325, 0.0325, 0.0325 ],
                    [0.0050, 0.0478, 0.2645, 0.2917, 0.2898, 0.1224, 0.1113, 0.1113, 0.1113, 0.1113, 0.1113 ],
                    [0.0050, 0.0000, 0.2680, 0.2680, 0.3473, 0.3473, 0.0956, 0.0956, 0.1527, 0.1527, 0.0387 ],
                    ]       
        self.cur_qoe = 0 

    # def allocate_resources(self, cpu_needed):
    #     if self.current_cpu_usage + cpu_needed <= self.cpu_allocated:
    #         self.current_cpu_usage += cpu_needed
    #         return True
    #     else:
    #         print("Not enough CPU resources!")
    #         return False
    def query_resource_decision(self):
        # self.area.vim.resource_decision_QLearning(self.cur_info)
        self.area.vim.resource_decision_baseline(self.cur_info)
          
    def add_streamer(self, streamer):
        self.active_streamers.append(streamer)

    def add_attack(self, atk_config):
        # TODO: Round Robin on the atack reduction     
        ids = self.area.ids
        # print(ids.processing_speed)
        # print("Before", atk_config["old_intensity"])
        if atk_config["old_intensity"] <= 0 or atk_config["name"] == "normal":
            atk_config["new_intensity"] = 0
        else:         
            atk_config["new_intensity"] = ids.detect(atk_config)
        # print("After", atk_config["new_intensity"])
        self.attack_config_list.append(atk_config) #TODO: Change this formula, Ranges from 0.0 to 1.0
        
    def get_qoe_video(self):
        n_user = len(self.active_streamers)
        params = self.qoe_vs_n_user_params[n_user]
        base_qoe = logistic(self.cpu_allocated, {}, *params)
        # std_dev = np.random.normal(0,self.std_matrix[n_user-1][int(self.cpu_allocated*2-1)], n_user)
        # qoe = np.tile(base_qoe, n_user) + 1*std_dev
        qoe = np.tile(base_qoe, n_user) 
        # print(self.cpu_allocated, self.active_streamers, qoe)
        return qoe
      
    def get_atk_impact(self): #TODO: Allow handling of multiple attack
        n_user = len(self.active_streamers)
        params = self.qoe_vs_n_user_params[n_user]
        base_qoe = logistic(self.cpu_allocated, {}, *params)
        if len(self.attack_config_list) == 0:
            return 0
        for impact_config in self.attack_config_list:
            atk_impact = 1 - logistic(self.cpu_allocated, impact_config, *params) / base_qoe
            pass #TODO: A function that calculates the total impacts of all attack to QoE
        # total_impact = np.mean(self.attack_config_list)
        return atk_impact

    def calculate_qoe(self):
        if len(self.active_streamers) == 0:
          return 1
        qoe_video = self.get_qoe_video().mean() 
        aggregate_attack_impact = self.get_atk_impact()   
        final_qoe = max(0,qoe_video*(1 - aggregate_attack_impact)) 
        
        # Record Current Condition and Performance
 
        self.cur_info = {
            "qoe": final_qoe,
            "video_cpu": self.cpu_allocated,
            "n_streamers": len(self.active_streamers),
            "ori_intensity": self.attack_config_list[0]["old_intensity"] if len(self.attack_config_list) else 0,
            "red_intensity": self.attack_config_list[0]["new_intensity"] if len(self.attack_config_list) else 0,                             
            } 
        # final_qoe = aggregate_attack_impact
        return min(1,final_qoe)
        
    def forward(self): #TODO: Resource Allocation from Control Node        
        self.active_streamers = []
        self.active_attackers = []
        self.attack_config_list = []


'''
~~~~~~ Control Node ~~~~~~
'''
class VIM:
    def __init__(self, area):
        self.area = area
        self.previous_info = pd.DataFrame(columns = ["timestep", "qoe", "n_streamers","original_intensity", "reduced_intensity", "video_cpu", "ids_cpu"]) 
        self.agent = QLearningAgent(self)
        
    def resource_decision_baseline(self, cur_info): 
        server = self.area.server
        ids = self.area.ids
        # Retrieve previous intensity and user count
        previous_intensity = cur_info["ori_intensity"]
        previous_user = cur_info["n_streamers"]
        
        # Get the updated CPU allocation
        updated_cpu = discrete_approx_decision(previous_intensity, previous_user)
        server.cpu_allocated = updated_cpu['best_cpu']
        ids.cpu_allocated = max_cpu - updated_cpu['best_cpu']
        # Print all relevant information
        print(f"Previous Intensity: {previous_intensity}, Previous User Count: {previous_user}")
        print(f"Updated CPU Allocation: {updated_cpu['best_cpu']}")
    
    def resource_decision_QLearning(self, cur_info):
        """ Update Q-table based on the new observation"""
        state = self.agent.get_state(cur_info)  # Get the current state
        action = cur_info.get("video_cpu", 3.0)  # Last video CPU allocation (default 3.0 if not present)
        reward = cur_info["qoe"]  # Use the calculated QoE as the reward
        next_state = self.agent.get_state(cur_info)
        print(state, action, reward, next_state)
        
        self.agent.update_q_table(state, action, reward, next_state)

        """Use the Q-learning agent to decide on resource allocation."""
        video_cpu = self.agent.take_action(cur_info)  # Get new CPU allocation for video
        # video_cpu = round(video_cpu/0.5)*0.5
        # video_cpu = round(video_cpu/0.5)*0.5
        ids_cpu = max_cpu - video_cpu  # IDS CPU based on remaining CPU capacity
        
        server = self.area.server
        ids = self.area.ids
        server.cpu_allocated = video_cpu
        ids.cpu_allocated = ids_cpu
        # Print the decision details
        print(f'Updated CPU Allocation: Video CPU={cur_info["video_cpu"]}, IDS CPU={(max_cpu - cur_info["video_cpu"])}, QoE={cur_info["qoe"]}')    
    
    def forward(self, cur_info):
        # Append the information to previous_info DataFrame
        self.previous_info = pd.concat([self.previous_info, pd.DataFrame({
            "timestep": [0],
            "qoe": [cur_info["qoe"]],
            "n_streamers": [cur_info["n_streamers"]],
            "original_intensity": [cur_info['ori_intensity']],
            "reduced_intensity": [cur_info["red_intensity"]],
            "video_cpu": [cur_info["video_cpu"]],
        })], ignore_index=True)
        
class LMM:
    pass
    #TODO:
    
class Orchestrator:
    def __init__(self):
        pass
    

        