from collections import deque
import random
class User:
    def __init__(self, user_id, video, initial_bandwidth=1000):
        self.user_id = user_id
        self.video = video
        self.buffer = deque()  # list of chunk durations
        self.estimated_bandwidth = initial_bandwidth  # in kbps
        self.playback_time = 0                                                                                                   
        self.playback_started = False
        self.current_chunk = 0
        self.res_history = []

    def abr_decision(self):
        buffer_sec = sum(self.buffer)
        safety_margin = 0.85
        safe_bw = self.estimated_bandwidth * safety_margin

        if buffer_sec < 10:
            for res in [480, 720, 1080]:
                chunk_size = (res / 480) * 300
                if chunk_size * 8 / self.video.chunk_duration <= safe_bw:
                    return res
            return 480
        elif buffer_sec > 30:
            for res in reversed([480, 720, 1080]):
                chunk_size = (res / 480) * 300
                if chunk_size * 8 / self.video.chunk_duration <= self.estimated_bandwidth:
                    return res
            return 480
        else:
            for res in reversed([480, 720, 1080]):
                chunk_size = (res / 480) * 300
                if chunk_size * 8 / self.video.chunk_duration <= safe_bw:
                    return res
            return 480

    def download_chunk(self):
        selected_res = self.abr_decision()
        chunk = self.video.serve_chunk(selected_res)
        if chunk:
            download_time = (chunk.size_kb * 8) / self.estimated_bandwidth  # in seconds
            # simulate bandwidth fluctuation
            self.estimated_bandwidth *= random.uniform(0.9, 1.1)
            self.buffer.append(chunk.duration)
            self.res_history.append(selected_res)
            return True
        return False

    def playback_tick(self):
        if self.playback_started:
            if self.buffer:
                self.buffer[0] -= 1
                if self.buffer[0] <= 0:
                    self.buffer.popleft()
                self.playback_time += 1
            else:
                # stall
                pass
        elif sum(self.buffer) >= 8:
            self.playback_started = True