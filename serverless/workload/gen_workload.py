import os
import matplotlib.pyplot as plt
import numpy as np
import logging

def simulate_loads(seed=0, period=200, amplitude=200, offset=0, noise_range=(-200, 200), size=1000, verbose=False):
    """
    This function generates and plots two types of loads: sporadic and periodic.

    Parameters:
    seed (int): The seed for the random number generator.
    period (int): The period length for the periodic load.
    amplitude (int): The amplitude for the periodic load.
    offset (int): The offset for the periodic load.
    noise_range (tuple): The range of the random noise for the periodic load.
    size (int): The size of the loads to generate.
    verbose (bool): If True, prints the first 50 points of each load.

    Returns:
    tuple: A tuple containing the generated sporadic load and periodic load.
    """

    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer.")
    if not isinstance(amplitude, int) or amplitude <= 0:
        raise ValueError("Amplitude must be a positive integer.")
    if not isinstance(offset, int):
        raise ValueError("Offset must be an integer.")
    if not isinstance(noise_range, tuple) or len(noise_range) != 2 or noise_range[0] >= noise_range[1]:
        raise ValueError("Noise range must be a tuple of two different numbers.")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate a sporadic load
    sporadic_load = np.random.choice([0] * 970 + list(np.random.randint(low=0, high=100, size=30)), size)

    # Generate a periodic load
    random_noise = np.random.uniform(noise_range[0], noise_range[1], size)
    periodic_load = amplitude * np.sin(2 * np.pi * np.arange(size) / period) + offset + random_noise
    periodic_load = periodic_load.astype(int)
    for i in range(len(periodic_load)):
        if periodic_load[i] < 0:
            periodic_load[i] = 0

    # Print the first 50 points if verbose is True
    if verbose:
        logging.info("Sporadic load (first 50 points): {}".format(sporadic_load[:50]))
        logging.info("Periodic load (first 50 points): {}".format(periodic_load[:50]))

    return sporadic_load, periodic_load

def plot_and_save_loads(sporadic_load, periodic_load):
    """
    This function plots and saves the given sporadic and periodic loads.

    Parameters:
    sporadic_load (ndarray): The sporadic load to plot and save.
    periodic_load (ndarray): The periodic load to plot and save.
    """

    plt.figure(figsize=(15, 5))
    # Plot the sporadic load
    plt.subplot(1, 2, 1)
    plt.plot(sporadic_load, label="Sporadic load", color="blue")
    plt.ylabel('Number of Requests')
    plt.xlabel('Time Unit')
    plt.title('Sporadic Load')
    plt.grid(True)
    plt.legend()

    # Plot the periodic load
    plt.subplot(1, 2, 2)
    plt.plot(periodic_load, label="Periodic load", color="green")
    plt.ylabel('Number of Requests')
    plt.xlabel('Time Unit')
    plt.title('Periodic Load')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    root_path = os.getenv("STREAMBOX_ROOT")
    
    # Save the loads to files
    with open(f'{root_path}/serverless/workload/sporadic-load.txt', 'w') as f:
        for i in range(len(sporadic_load)):
            f.write(str(sporadic_load[i]) + ' ')

    with open(f'{root_path}/serverless/workload/periodic-load.txt', 'w') as f:
        for i in range(len(periodic_load)):
            f.write(str(periodic_load[i]) + ' ')

# Usage example
sporadic_load, periodic_load = simulate_loads(seed=0, verbose=True)
plot_and_save_loads(sporadic_load, periodic_load)
