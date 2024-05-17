"""
Utilities
---------
This file contains helper functions for various tasks, among others plotting.
"""
import random
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from moviepy.editor import ImageSequenceClip


def set_seed(seed=2023):
    """
    Seeds random functions of python, numpy and tensorflow.
    Args:
        seed: int
            Seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

    return None


def smoothing(lst, n=10):
    """
    Given a list of values, it smoothes the values by replacing a value with the mean of itself
    and the previous `n` values. If less than `n` values exist prior to it, then all prior values
    are included.

    Args:
        lst: list
            A list of values.

        n: int
            An integer determining the smoothing window.

    Returns:
        result: list
            The result after smoothing the list.

    """
    result = [np.mean(lst[max(0, i-n):i+1]) for i in range(len(lst))]
    return result


def process_stats(data):
    """
    Given collected data from one or more runs, produce stats of these runs.

    Parameters
    ----------
    data: list
        List of list(s), where each inner list contains data collected during
        a training run, such as episode returns or lengths.

    Returns
    -------
    means, upper, lower: list
        Lists of the means as well as means ± 1.96*standard deviations of the collected data across the runs.

    """
    stats = [(np.mean(el), np.std(el)) for el in zip(*data)]
    mean = [el[0] for el in stats]
    stds = [el[1] for el in stats]
    upper = [mean[i] + 1.96 * stds[i] for i in range(len(stats))]
    lower = [mean[i] - 1.96 * stds[i] for i in range(len(stats))]

    return smoothing(mean), smoothing(upper), smoothing(lower)


def plot_training(all_episode_returns, all_episode_lengths, all_critic_losses):
    """
    Plot the average return, together with shading with the stddev, for a number of runs.

    Parameters
    ----------
    all_episode_returns: list
        A list of lists, where each inner list contains the episode returns for a training run.
    all_episode_lengths: list
        A list of lists, where each inner list contains the episode lengths for a training run.
    all_critic_losses: list
        A list of lists, where each inner list contains the losses of the critic.
    figure_name: str
        Name to save figure to.


    Returns
    -------
    None
    """
    return_mean, return_upper, return_lower = process_stats(all_episode_returns)
    length_mean, length_upper, length_lower = process_stats(all_episode_lengths)
    loss_mean, loss_upper, loss_lower = process_stats(all_critic_losses)
    x = range(1, len(return_upper)+1)

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))
    ax1.set(xlabel='Episodes', ylabel='Return', title='Episode returns')
    ax1.plot(return_mean, color='tomato', label='return_mean')
    ax1.fill_between(x=x, y1=return_upper, y2=return_lower, color='grey',
                     label='mean ± 1.96*std', alpha=0.3, edgecolor='black')
    ax1.legend()
    ax2.set(xlabel='Episodes', ylabel='Length', title='Episode lengths')
    ax2.plot(length_mean, color='olive', label='length_mean')
    ax2.fill_between(x=x, y1=length_upper, y2=length_lower, color='grey',
                     label='mean ± 1.96*std', alpha=0.3, edgecolor='black')
    ax2.legend()

    ax3.set(xlabel='Steps', ylabel='Loss', title='Critic loss')
    ax3.plot(loss_mean, color='slateblue', label='loss_mean')
    ax3.fill_between(x=range(len(loss_upper)), y1=loss_upper, y2=loss_lower, color='grey',
                     label='mean ± 1.96*std', alpha=0.3, edgecolor='black')
    ax3.legend()
    # plt.savefig(f'media/{figure_name}.pdf')

    return None


# def record_gif(images, name):
#     """
#     Create a gif from a list of images.
#     Args:
#         images: list
#             Each image is a numpy.ndarray.
#         name: str
#             Name of gif.

#     Returns:
#         None

#     """
#     clip = ImageSequenceClip(sequence=images, fps=48)
#     clip.write_gif(f"videos/{name}.gif")

# =============== END OF FILE ===============
