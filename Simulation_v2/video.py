# Re-import required libraries after reset
import random
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from utils import plot_streaming_overview

# Constants
CHUNK_DURATION = 4  # seconds
TOTAL_TIMESTEPS = 100
B_RAN = 100e6  # 100 MHz
ETA = 5        # bps/Hz
CHANNEL_BANDWIDTH_HZ = 100e6
BITRATE_SELECTION = np.array([1000, 1800, 4500])

'''
Current Implementation assume allocated bandwidth is fair for every user
TODO: If no bitrate selection available,   do stalling
'''
class User:
    def __init__(self, user_id, snr, bw):
        self.user_id = user_id
        self.snr = snr  # linear SNR
        self.allocated_bandwidth = bw
        self.selected_bitrate = None

    def download_chunk(self, current_time):
        valid = BITRATE_SELECTION[BITRATE_SELECTION <= self.estimate_capacity()]
        if len(valid) > 0:
            self.selected_bitrate = max(valid)
        else:
            self.selected_bitrate = BITRATE_SELECTION[0]  # fallback

    def estimate_capacity(self):
        # Return maximum achievable throughput in kbps for full channel use
        if self.allocated_bandwidth==None:
            return 0
        return ETA * math.log2(1 + self.snr) * self.allocated_bandwidth / 1000


class LiveStream:
    def __init__(self, stream_id, duration, current_time):
        self.stream_id = stream_id
        self.duration = duration  # in seconds
        self.start_time = None
        self.users = []
        self.finished = False
        self.start(current_time)

    def start(self, current_time):
        self.start_time = current_time
        self.users = self.initialize_users()
        self.transcoding_resource_impact()

    def initialize_users(self):
        user_count = random.randint(100, 200)
        return [User(user_id=i, snr=random.uniform(20, 40), bw=None) for i in range(user_count)]

    def transcoding_resource_impact(self):
        # TODO: Apply CPU and memory cost per stream
        pass

    def update(self, current_time):
        if current_time - self.start_time >= self.duration:
            self.finished = True
        else:
            for user in self.users:
                user.download_chunk(current_time)


'''
TODO: Effect of CPU and Mem Usage
'''
class VideoServer:
    def __init__(self, server_id, channel_bandwidth_hz=CHANNEL_BANDWIDTH_HZ):
        self.server_id = server_id
        self.channel_bandwidth_hz = channel_bandwidth_hz
        self.active_streams = []
        self.time_elapsed = 0
        self.all_users = []
        self.average_qoe = None
        self.bandwidth_utilization = None
        self.cpu = 0
        self.memory = 0
        self.transcoding_speed = {
            "360": 0,
            "720": 0,
            "1080": 0,
        }
        self.available_cpu_cycle = 0

        
    def update(self):
        self.time_elapsed += CHUNK_DURATION
        all_users = [user for stream in self.active_streams for user in stream.users]
        num_users = len(all_users)

        if num_users == 0:
            return

        # ====== Allocate Bandwidth ====== #
        # Step 1: Initial allocation
        for user in all_users:
            user.allocated_bandwidth = B_RAN
            user.download_chunk(self.time_elapsed)  # choose bitrate based on fair estimate

        # Step 2: Compute actual required bandwidth for chosen bitrate
        user_bandwidth_demand = []
        total_bw = 0

        for user in all_users:
            spectral_eff = ETA * math.log2(1 + user.snr)
            if spectral_eff == 0:
                required_bw = float("inf")
            else:
                required_bw = (user.selected_bitrate * 1000) / spectral_eff

            user_bandwidth_demand.append((user, required_bw))
            total_bw += required_bw

        # Step 3: Sort users by demand (descending)
        user_bandwidth_demand.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Reduce until total fits
        while total_bw > self.channel_bandwidth_hz and user_bandwidth_demand:
            user, req_bw = user_bandwidth_demand.pop(0)  # highest demander

            # Downgrade bitrate if possible
            current_idx = np.where(BITRATE_SELECTION == user.selected_bitrate)[0][0]
            if current_idx > 0:
                downgraded = BITRATE_SELECTION[current_idx - 1]
                user.selected_bitrate = downgraded
                spectral_eff = ETA * math.log2(1 + user.snr)
                new_req_bw = (downgraded * 1000) / spectral_eff

                total_bw -= (req_bw - new_req_bw)
                user_bandwidth_demand.append((user, new_req_bw))
                user_bandwidth_demand.sort(key=lambda x: x[1], reverse=True)
            else:
                # Can't downgrade further, lock in lowest bitrate
                total_bw -= (req_bw - req_bw)
        # Step 5: Assign final bandwidths
        for user, final_bw in user_bandwidth_demand:
            user.allocated_bandwidth = final_bw
            user.download_chunk(self.time_elapsed)

        # Final stream update
        for stream in self.active_streams:
            stream.update(self.time_elapsed)
            self.all_users.extend(stream.users)

        self.active_streams = [s for s in self.active_streams if not s.finished]




    def start_live_stream(self, stream_id, duration, current_time):
        stream = LiveStream(stream_id, duration, current_time)
        self.active_streams.append(stream)


    def get_transcoding_speed(self):
        pass
    
    def allocate_transcoding(self):
        pass


if __name__ == '__main__':
    def run_simulation(total_timesteps=50, downlink_capacity_mbps=500):
        server = VideoServer(server_id=1)
        stream_id_counter = 1
        stream_start_probability = 0.5
        log_records = []

        for timestep in range(total_timesteps):
            current_time = timestep * CHUNK_DURATION

            # Possibly start a new live stream
            if random.random() < stream_start_probability:
                duration = random.randint(20, 40)
                server.start_live_stream(stream_id=stream_id_counter, duration=duration, current_time=current_time)
                stream_id_counter += 1

            # Run server update
            server.update()

            # Log each user's performance
            for stream in server.active_streams:
                for user in stream.users:
                    log_records.append({
                        "time": current_time,
                        "user_id": user.user_id,
                        "snr": user.snr,
                        "allocated_bandwidth_hz": user.allocated_bandwidth,
                        "selected_bitrate": user.selected_bitrate,
                        "n_stream": len(server.active_streams),
                    })
        return pd.DataFrame(log_records)

    # Run simulation
    df = run_simulation(total_timesteps=TOTAL_TIMESTEPS, downlink_capacity_mbps=CHANNEL_BANDWIDTH_HZ)
    plot_streaming_overview(df, CHANNEL_BANDWIDTH_HZ)



