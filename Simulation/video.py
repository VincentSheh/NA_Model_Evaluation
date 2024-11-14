import numpy as np
from ids import IDS
from control import EdgeArea, VIM
def logistic(x, impact_config={}, L_base=0.97, beta_0=-2.27, beta_CPU_base=4.8802):
    # Use default logistic function if no impact configuration is provided
    if impact_config == {}:
        return L_base / (1 + np.exp(-(beta_0 + beta_CPU_base * x)))

    # Define exponential saturation adjustment for intensity effect
    intensity_effect = 1 - np.exp(-impact_config["alpha"] * impact_config["new_intensity"])

    # Adjust parameters based on intensity effect
    L = L_base * (1 + impact_config["L_k"] * intensity_effect)  # Decrease L with intensity effect
    beta_0 = beta_0 * (1 + impact_config["beta_0_k"] * intensity_effect)  # Adjust beta_0 similarly
    beta_CPU = beta_CPU_base * (1 + impact_config["beta_CPU_k"] * intensity_effect)  # Adjust beta_CPU similarly

    # Return adjusted logistic function value
    return L / (1 + np.exp(-(beta_0 + beta_CPU * x)))

class Streamer(object): # Task
    def __init__(self, streamer_id, hsmm_states):

        self.streamer_id = streamer_id  # Unique identifier for the streamer
        self.hsmm_states = hsmm_states  
        self.time_elapsed = 0
        self.state = self.hsmm_states[self.time_elapsed]
        self.current_server = None 

    def start_stream(self, video_server):
        video_server.add_streamer(self.streamer_id)
        self.current_server = video_server
    def stop_stream(self):
        pass #TODO: Disconnect by removing it from the active_streamer list

    def add_viewer(self, user):
        if len(self.current_viewers) < self.max_viewers:
            self.current_viewers.append(user)
        else:
            print(f"Streamer {self.streamer_id} is at full capacity!")

    def remove_viewer(self, user):
        if user in self.current_viewers:
            self.current_viewers.remove(user)
    def forward(self):
        self.time_elapsed += 1  
        if self.time_elapsed < len(self.hsmm_states):
          self.state = self.hsmm_states[self.time_elapsed]      
        else:
          self.state = None


class VideoServer(object): # Edge Area Server
    def __init__(self, cpu_capacity):
        self.cpu_capacity = cpu_capacity  # Total CPU capacity of the server
        # self.ids = [IDS(6.0 - self.cpu_capacity)]
        self.current_cpu_usage = 0  # Current CPU usage
        self.active_streamers = []  # List of active Streamer objects
        self.active_attackers = []  
        self.qoe_vs_n_user_params = {
            1: [0.97, -2.2782716980482065, 4.8802368476587255],
            2: [0.97, -3.8873078667967476, 5.175670954212056],
            3: [0.97, -4.4959769724864636, 4.081147830542201],
            4: [0.97, -3.3377765392197962, 2.5431386672697025],
            5: [0.97, -4.186919060489949, 2.3893219975556157],
            6: [0.97, -5.935630098896784, 2.56309746149014],
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

    def allocate_resources(self, cpu_needed):
        if self.current_cpu_usage + cpu_needed <= self.cpu_capacity:
            self.current_cpu_usage += cpu_needed
            return True
        else:
            print("Not enough CPU resources!")
            return False
          
    def add_streamer(self, streamer):
        self.active_streamers.append(streamer)

    def add_attack(self, atk_config):
        # TODO: Round Robin on the atack reduction     
        for ids in self.ids:
            # print(ids.processing_speed)
            print("Before", atk_config["old_intensity"])
            if atk_config["old_intensity"] <= 0:
              atk_config["new_intensity"] = 0
              break     
            atk_config["new_intensity"] = ids.detect(atk_config)
            print("After", atk_config["new_intensity"])
        self.attack_config_list.append(atk_config) #TODO: Change this formula, Ranges from 0.0 to 1.0
        
    def get_qoe_video(self):
        n_user = len(self.active_streamers)
        params = self.qoe_vs_n_user_params[n_user]
        base_qoe = logistic(self.cpu_capacity, {}, *params)
        std_dev = np.random.normal(0,self.std_matrix[n_user-1][int(self.cpu_capacity*2-1)], n_user)
        qoe = np.tile(base_qoe, n_user) + 0*std_dev
        # print(self.cpu_capacity, self.active_streamers, qoe)
        return qoe
      
    def get_atk_impact(self): #TODO: Allow handling of multiple attack
        n_user = len(self.active_streamers)
        params = self.qoe_vs_n_user_params[n_user]
        base_qoe = logistic(self.cpu_capacity, {}, *params)
        if len(self.attack_config_list) == 0:
            return 0
        for impact_config in self.attack_config_list:
            atk_impact = 1 - logistic(self.cpu_capacity, impact_config, *params) / base_qoe
            pass #TODO: A function that calculates the total impacts of all attack to QoE
        # total_impact = np.mean(self.attack_config_list)
        return atk_impact

    def calculate_qoe(self):
        if len(self.active_streamers) == 0:
          return 1
        qoe_video = self.get_qoe_video().mean() 
        aggregate_attack_impact = self.get_atk_impact()        
        final_qoe = max(0,qoe_video*(1 - aggregate_attack_impact)) #TODO: Reduce attack impact using IDS
        # final_qoe = aggregate_attack_impact
        return min(1,final_qoe)
        
    def forward(self): #TODO: Resource Allocation from Control Node
        self.active_streamers = []
        self.active_attackers = []
        self.attack_config_list = []
        for ids in self.ids:
          ids.forward(6.0-self.cpu_capacity)
        # self.ids = {"A": [Quota, Accuracy],
        #             "B": [Quota, Accuracy], ...}
        # self.cpu, self.ids.cpu
        # self.ids = {"A": []}

        
