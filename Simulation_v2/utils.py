import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
def plot_streaming_overview(df, CHANNEL_BANDWIDTH_HZ):

    # 1. Group: Users per bitrate
    bitrate_counts = df.groupby(["time", "selected_bitrate"]).size().unstack(fill_value=0)

    # 2. Group: Total users and bandwidth per time
    metrics = df.groupby("time").agg(
        total_users=("user_id", "nunique"),
        total_bandwidth=("allocated_bandwidth_hz", "sum"),
    ).reset_index()
    # Add number of active streams per time (already logged)
    stream_counts = df.groupby("time")["n_stream"].max().reset_index(name="n_stream")
    metrics = metrics.merge(stream_counts, on="time", how="left")

    # 3. Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Primary y-axis: Stacked area chart for bitrate usage ---
    bitrate_counts.plot.area(ax=ax1, stacked=True, alpha=0.6, cmap="viridis")
    ax1.set_ylabel("Number of Users (per Bitrate)")
    ax1.set_xlabel("Time (seconds)")
    ax1.set_title("Streaming Overview: Users per Bitrate, Stream Count, Bandwidth Utilization")
    ax1.grid(True)

    # --- Secondary y-axis: Line for total users and bandwidth ---
    ax2 = ax1.twinx()
    sns.lineplot(data=metrics, x="time", y="total_users", label="Total Streams", ax=ax2, color="black", linestyle="--")
    sns.lineplot(data=metrics, x="time", y="total_bandwidth", label="Bandwidth Used (Hz)", ax=ax2, color="red")
    sns.lineplot(data=metrics, x="time", y="n_stream", label="Active Streams", ax=ax2, color="blue")
    # Max capacity line
    ax2.axhline(y=CHANNEL_BANDWIDTH_HZ, color="red", linestyle=":", label="Max Bandwidth")

    ax2.set_ylabel("Streams / Bandwidth (Hz)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("bitrate_and_bandwidth_overview.png")
    df.to_csv("test.csv")