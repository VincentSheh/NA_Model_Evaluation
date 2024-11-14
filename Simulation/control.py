import json
from ids import IDS
from video import VideoServer
with open("DARA.json", "r") as file:
    best_approx_decision_dict = json.load(file)
    
    
class EdgeArea:
    def __init__(self):
        self.vim = VIM(self)
        self.ids = IDS(self)
        self.server = VideoServer(self)

def discrete_approx_decision(intensity, n_streamer):
    intensity_interval = 200

    closest_intensity = int(intensity/intensity_interval)*intensity_interval
    return best_approx_decision_dict[str((closest_intensity, n_streamer))]
    


class VIM:
    def __init__(self, area):
        self.area = area
    def resource_decision(self): 
        server = self.edge_area
        # Retrieve previous intensity and user count
        previous_intensity = server.attack_config_list[0]["old_intensity"]
        previous_user = len(server.active_streamers)
        
        # Get the updated CPU allocation
        updated_cpu = discrete_approx_decision(previous_intensity, previous_user)
        server.cpu_capacity = updated_cpu['best_cpu']
        
        # Print all relevant information
        print(f"Previous Intensity: {previous_intensity}, Previous User Count: {previous_user}")
        print(f"Updated CPU Allocation: {updated_cpu['best_cpu']}")
        
class LMM:
    pass
    #TODO:
    

        