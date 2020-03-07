# a few plotting functions for examining results

import seaborn
import matplotlib.pyplot as plt

def plotPosterior(T, xlabels, ylabels, title='Posterior Distribution'):
    fig, ax = plt.subplots(figsize=(16,15))
    ax = seaborn.heatmap(matrix, annot=True, xticklabels=labels, yticklabels=labels, cmap='Oranges', mask=np.ones_like(T))
    plt.title(title, fontsize=28, fontweight='bold')
    plt.xlabel('Bin Midpoint', fontsize=15, fontweight='bold')
    plt.ylabel('TMRCA of IBDs', fontsize=15, fontweight='bold')
    plt.savefig(f'{title}.png', dpi=300)