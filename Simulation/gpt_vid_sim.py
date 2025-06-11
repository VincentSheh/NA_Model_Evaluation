import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque

class Chunk:
    def __init__(self, bitrate, duration):
        self.bitrate = bitrate  # Bitrate of the chunk in Kbps
        self.duration = duration  # Duration of the chunk in seconds
        self.size = bitrate * duration  # Size of the chunk in kilobits (Kb)
        self.downloaded = 0  # Track how much of the chunk has been downloaded

    def download(self, throughput, time_slot):
        """Simulate the download of a part of this chunk for a given time_slot."""
        download_amount = throughput * time_slot  # Throughput is in Kbps, time_slot is in seconds
        self.downloaded += download_amount
        # print(f"dowanload amount: {download_amount}, dowloaded: {self.downloaded}, size: {self.size}")
        if self.downloaded >= self.size:
            self.downloaded = self.size  # Cap the downloaded size to the chunk size
            return True  # Download complete
        return False  # Still downloading

    def is_complete(self):
        """Check if the chunk has been fully downloaded."""
        # print(f"chunk downloaded: {self.downloaded}, chunk size: {self.size}")
        return self.downloaded >= self.size
    
    def __str__(self) -> str:
        return f"({self.bitrate} Kbps, {self.duration} s)"

class VideoStreamingQoE:
    def __init__(self, duration = -1, buffer_capacity=8, chunk_duration=2, directory='figs'):
        # self.bitrates = sorted(bitrates)  # Available bitrates (in Kbps)
        self.buffer = 0  # Initial buffer in seconds
        self.max_buffer = buffer_capacity  # Maximum buffer size in seconds
        self.chunk_duration = chunk_duration  # Duration of each video chunk in seconds
        self.current_chunk = None  # Current chunk being downloaded
        self.resolution_history = []  # Track the bitrate for each time slot
        self.stall_indicator = []
        self.frame_history = []
        self.throughput_history = []

        self.buffer_history = []

        self.bitrates = [4, 7.5, 12]
        self.max_resolution = max(self.bitrates)

        self.downloaded_chunks = deque([])

        self.bitrate_0 = self.bitrates[0]

        self.current_time = 0
        self.directory = f"{directory}/figs"

    def select_bitrate(self, throughput):
        """Select the highest bitrate that can be sustained by the current throughput."""
        # suitable_bitrate = self.bitrates[0]
        # for bitrate in self.bitrates:
        #     if bitrate <= throughput:
        #         suitable_bitrate = bitrate
        #     else:
        #         break
        # return suitable_bitrate
        new_bitrate = 0.6*throughput + 0.4* self.bitrate_0
        # print(f'new_bitrate: {new_bitrate}')
        suitable_bitrate = self.bitrates[0]
        for bitrate in self.bitrates:
            if bitrate <= new_bitrate:
                suitable_bitrate = bitrate
            else:
                break
        self.bitrate_0 = suitable_bitrate
        # print(f'suitable bitrate: {suitable_bitrate}')
        return suitable_bitrate


    def create_new_chunk(self, throughput):
        """Create a new chunk based on the selected bitrate and start downloading."""
        selected_bitrate = self.select_bitrate(throughput)
        self.current_chunk = Chunk(selected_bitrate, self.chunk_duration)
        # print(f"Started downloading new chunk at {selected_bitrate} Kbps")

    def download_chunk(self, throughput, time_slot=1):
        """Download the current chunk or create a new one if needed."""
        if self.current_chunk is None or self.current_chunk.is_complete():
            # print('create new chunk')
            self.create_new_chunk(throughput)

        # Attempt to download the chunk
        if self.current_chunk.download(throughput, time_slot):
            # If the chunk download completes, add to buffer
            # self.buffer += self.chunk_duration
            # self.buffer = min(self.buffer, self.max_buffer)
            if self.buffer + self.chunk_duration > self.max_buffer:
                # print(f'buffer full')
                cut_chunk = (self.buffer + self.chunk_duration) - self.max_buffer
                self.current_chunk.downloaded -= cut_chunk
                self.buffer = self.max_buffer

            else:
                # print(f'buffer not full')
                self.buffer += self.chunk_duration
                
            self.downloaded_chunks.append(self.current_chunk)
            # print(f"Completed downloading chunk at {self.current_chunk.bitrate} Kbps. Buffer: {self.buffer:.2f}s.")

    def next_timestamp(self, throughput):

        # Download a chunk in every time step, regardless of buffer status
        self.download_chunk(throughput)
        self.throughput_history.append(throughput)
        if self.buffer > 0:
            # Buffer depletion simulation
            self.buffer -= 1  # Decrease buffer by 1 second for each time slot
            # print(f"Buffer remaining: {self.buffer:.2f}s")
            # Record the bitrate used for playback
            # if self.current_chunk and self.current_chunk.is_complete():
            #     self.resolution_history.append(self.current_chunk.bitrate)
            if len(self.downloaded_chunks) > 0:
                if self.downloaded_chunks[0].duration > 0:
                    self.downloaded_chunks[0].duration -= 1
                    self.stall_indicator.append(0)
                    self.frame_history.append(self.downloaded_chunks[0].bitrate)
                    self.resolution_history.append(self.downloaded_chunks[0].bitrate / self.max_resolution)
                    if self.downloaded_chunks[0].duration <=0:
                        self.downloaded_chunks.popleft()
                else:
                    raise ValueError("No remaining chunks to play.")
            else:
                print(f"Wrong")
        else:
            # If buffer is empty, stall and record 0 bitrate, but continue downloading
            # print("Buffer empty! Stalling, but still downloading data.")
            self.resolution_history.append(0)  # Stall, record 0 bitrate
            self.stall_indicator.append(1)
            self.frame_history.append(0)

        self.buffer_history.append(self.buffer)
        self.current_time += 1
        # print(f"Buffer remaining: {self.buffer:.2f}s")
        # print(f'Chunks: {" | ".join([str(chunk) for chunk in self.downloaded_chunks])}\n')


    def simulate(self, throughputs, steps=10):
        """Simulate adaptive bitrate streaming over a series of throughput measurements."""
        for step in range(steps):
            throughput = throughputs[step] if step < len(throughputs) else random.choice(throughputs)
            # print(f"Time slot {step + 1} | Throughput: {throughput} Kbps")
            self.next_timestamp(throughput)
            # self.next_timestamp(throughputs[step])


    def get_resolution_history(self):
        """Return the recorded bitrate history."""
        return self.resolution_history
    
    def buffer_execeed_limit(self):
        return self.buffer >= self.max_buffer
    
    def clear_buffer_chunk(self):
        self.buffer = 0
        self.downloaded_chunks = deque([])

    def calculate_VQ(self):
        # print(f'resolution history: ')
        # print(self.resolution_history)
        return np.mean(self.resolution_history)

    def calculate_SW(self):
        diffs = np.diff(self.resolution_history)
        return np.mean(diffs**2)

    def calculate_T_w(self):
        return self.t_B + self.D_buf / self.resolution_history[0]

    def calculate_T_st(self):
        return np.sum(self.stall_indicator) / len(self.stall_indicator)

    def calculate_QoE(self, alpha, beta, gamma, delta):
        VQ = self.calculate_VQ()
        SW = self.calculate_SW()
        # T_w = self.calculate_T_w()
        T_st = self.calculate_T_st()
        
        # QoE = alpha * VQ - beta * SW - gamma * T_w - delta * T_st
        QoE = alpha * VQ - beta * SW - delta * T_st
        # print(f"VQ: {VQ}")
        # print(f"SW: {SW}")
        # print(f"T_st: {T_st}")
        return QoE
    
    def plot_buffer_size(self, car_id = -1):
        plt.figure(figsize=(12, 6))
        plt.plot(range(self.current_time), self.buffer_history)
        plt.title('Buffer Size Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Buffer Size (frames)')
        plt.grid(True)
        plt.savefig(f"{self.directory}/Buffer_Size of car {car_id}")
        plt.close()

    # def plot_resolution(self, car_id = -1):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(range(self.current_time), self.resolution_history)
    #     plt.plot(range(self.current_time), self.throughput_history)
    #     plt.title('Resolution Over Time')
    #     plt.xlabel('Time (seconds)')
    #     plt.ylabel('Resolution')
    #     plt.ylim(-0.5, 1.5)
    #     plt.grid(True)
    #     plt.savefig(f"{self.directory}/Resolution of car {car_id}")
    #     plt.close()
    def plot_resolution(self, car_id = -1):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot for resolution_history on the left y-axis
        ax1.plot(range(self.current_time), self.resolution_history, color='b', label='Resolution')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Resolution', color='b')
        ax1.set_ylim(-0.5, 1.5)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Create a twin Axes sharing the x-axis for throughput_history on the right y-axis
        ax2 = ax1.twinx()
        ax2.plot(range(self.current_time), self.throughput_history, color='r', label='Throughput')
        ax2.set_ylabel('Throughput', color='r')
        ax2.set_ylim(0, max(self.throughput_history))
        ax2.tick_params(axis='y', labelcolor='r')

        # Title, grid, and save the plot
        plt.title(f'Resolution and Throughput Over Time for car {car_id}')
        ax1.grid(True)

        # Save the plot
        plt.savefig(f"{self.directory}/Resolution_and_Throughput_of_car_{car_id}.png")
        plt.close()


if __name__ == "__main__":
    # Example usage:
    # bitrates = [300, 750, 1200, 2400]  # Available bitrates in Kbps
    # throughputs = [400, 800, 1500, 1000, 600, 300, 2000, 2500]  # Simulated throughput values in Kbps
    bitrates = [4, 7.5, 12]
    throughputs = [8, 12, 19, 10, 7, 8, 4, 6]  # Simulated throughput values in Kbps
    # throughputs= np.random.normal(loc=100.0, scale=4.0, size=1000)
    # print(t)

    simulator = VideoStreamingQoE(directory='./')
    simulator.simulate(throughputs, steps=10)

    # Output the recorded bitrate history
    print("\nBitrate history during playback:")
    print(simulator.get_resolution_history())
    print(len(simulator.get_resolution_history()))
    print(simulator.stall_indicator)

    # Calculate QoE
    # alpha, beta, gamma, delta = 0.5, 0.2, 0.1, 0.2  # example weights
    alpha, beta, gamma, delta = 0.7, 0.2, 0.1, 0.2  # example weights
    qoe = simulator.calculate_QoE(alpha, beta, gamma, delta)
    simulator.plot_resolution()
    print(f"Calculated QoE: {qoe}")