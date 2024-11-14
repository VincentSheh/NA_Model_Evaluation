import json


    
def discrete_approx_decision(intensity, n_streamer):
    intensity_interval = 200
    with open("DARA.json", "r") as file:
        best_approx_decision_dict = json.load(file)
    closest_intensity = int(intensity/intensity_interval)*intensity_interval
    return best_approx_decision_dict[str((closest_intensity, n_streamer))]
    

