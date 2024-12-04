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



        
