import numpy as np
from enum import Enum


class AttackConfig(Enum):
    NORMAL = {
        "name": "normal",
        "alpha": 0,
        "L_k": -1.0,
        "beta_0_k": 1.0,
        "beta_CPU_k":-1.0,
    }
    BONESI = {
        "name": "bonesi",
        "alpha": 0.005,
        "L_k": -0.2,
        "beta_0_k": 0.2,
        "beta_CPU_k":-0.6,
    }
    GOLDENEYE = {
        "name": "goldeneye",
        "alpha": 0.01,
        "L_k": -0.02,
        "beta_0_k": 0.5,
        "beta_CPU_k":-0.4,
    }
    HULK = {
        "name": "hulk",
        "alpha": 0.015,
        "L_k": -0.6,
        "beta_0_k": 0.2,
        "beta_CPU_k":-0.3,
    }
    def __init__(self, properties):
        self.properties = properties
        # Optionally
        self.alpha = properties["alpha"]
        self.L_k = properties["L_k"]
        self.beta_0_k = properties["beta_0_k"]
        self.beta_CPU_k = properties["beta_CPU_k"]        
attack_type_dict = {
    "NoAtk": AttackConfig.NORMAL,
    "bonesi": AttackConfig.BONESI,
    "goldeneye": AttackConfig.GOLDENEYE,
    "hulk": AttackConfig.HULK,
}



class Attacker:
    def __init__(self, atk_id, intensity_sequence, hsmm_state, atk_type="bonesi"):
        self.atk_id = atk_id
        self.hsmm_state = hsmm_state  # Attack state change over timesteps
        self.intensity_sequence = intensity_sequence
        self.time_elapsed = 0  # Track how long the attack has been active
        self.state = hsmm_state[self.time_elapsed]  # Initial state
        self.atk_type = attack_type_dict[self.state]
        self.intensity = 0  # e.g., requests per second

    def update_intensity(self):
        # Simulate intensity decay based on time elapsed
        # decay_factor = np.exp(-0.1 * self.time_elapsed)      
        if self.state == "NoAtk":
            self.intensity = 0
        else: 
            self.intensity = self.intensity_sequence[self.time_elapsed] 
            # self.intensity = 1000
            
            
    def get_config(self):
        # Calculate impact based on current intensity and impact function
        impact_config = self.atk_type.properties
        impact_config["old_intensity"] = self.intensity
        return impact_config
    
    def start(self, video_server):
        impact_config = self.get_config()
        video_server.add_attack(impact_config)
        
    def forward(self):
        # Move forward in time, updating state based on the hsmm_state sequence
        self.time_elapsed += 1
        if self.time_elapsed < len(self.hsmm_state):
            self.state = self.hsmm_state[self.time_elapsed]
            self.atk_type = attack_type_dict[self.state]
            self.update_intensity()
        else:
            self.state = None  # Indicates that the attack sequence has ended
    def forward_controlled(self, new_state, intensity):
        self.time_elapsed += 1
        if self.time_elapsed < len(self.hsmm_state):
            self.state = new_state
            self.atk_type = attack_type_dict[new_state]
            self.intensity = intensity
        else:
            self.state = None  # Indicates that the attack sequence has ended        
        