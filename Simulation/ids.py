from control import EdgeArea
class IDS(object):
    def __init__(self,cpu):
        self.cpu = cpu  # Initial CPU allocation for the IDS
        # self.processing_speed = {}  # Processing speed for each attack type or variant
        #! self.processing_speed
        self.processing_speed = lambda x: 450*x - 100  # Minimum CPU = 0.5 #TODO: Add Noise 
        self.accuracy = {"bonesi": 0.9, "goldeneye": 0.5, "hulk": 0.5}  # Accuracy for different attack variants
        self.cur_quota = self.processing_speed(self.cpu)

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

    def forward(self, cpu_allocation):
        #Update CPU and Processing Speed
        self.cpu = cpu_allocation
        self.cur_quota = self.processing_speed(self.cpu)
        
            
    def update_defense_factor(self, attack_type, factor):
        self.defense_factor[attack_type] = factor
        print(f"Updated defense factor for '{attack_type}' to {factor:.2f}")            

    def add_attack_variant(self, variant_name):
        if variant_name not in self.accuracy:
            self.accuracy[variant_name] = 0.5  # Initial accuracy for the new variant
            self.processing_speed[variant_name] = self.cpu  # Initial processing speed
            self.defense_factor[variant_name] = 1.0  # Default defense factor
            print(f"Added new attack variant '{variant_name}' with default settings.")
        else:
            print(f"Attack variant '{variant_name}' already exists.")
