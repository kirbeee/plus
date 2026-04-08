'''
Implementation of the local and global unlinkability metrics for biometric template protection systems evaluation.
More details in:


[TIFS18] M. Gomez-Barrero, J. Galbally, C. Rathgeb, C. Busch, "General Framework to Evaluate Unlinkability
in Biometric Template Protection Systems", in IEEE Trans. on Informations Forensics and Security, vol. 3, no. 6, pp. 1406-1420, June 2018

Please remember to reference article [TIFS18] on any work made public, whatever the form,
based directly or indirectly on these metrics.
'''

__author__ = "Marta Gomez-Barrero"
__copyright__ = "Copyright (C) 2017 Hochschule Darmstadt"
__license__ = "License Agreement provided by Hochschule Darmstadt (https://github.com/dasec/unlinkability-metric/blob/master/hda-license.pdf)"
__version__ = "2.0"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab
import seaborn as sns

class UnlinkabilityMetric:
    def __init__(self, mated_scores, non_mated_scores, omega=1.0, n_bins=-1):
        self.mated_scores = np.array(mated_scores)
        self.non_mated_scores = np.array(non_mated_scores)
        self.omega = omega

        if n_bins == -1:
            self.n_bins = int(min(len(self.mated_scores) // 10, 100))
        else:
            self.n_bins = n_bins

        self.bin_edges = None
        self.bin_centers = None
        self.D = None
        self.Dsys = None

    def evaluate(self):
        # load scores
        matedScores = self.mated_scores
        nonMatedScores = self.non_mated_scores

        # define range of scores to compute D
        bin_edges = np.linspace(min([min(matedScores), min(nonMatedScores)]),
                                   max([max(matedScores), max(nonMatedScores)]), num=self.n_bins + 1, endpoint=True)
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2  # find bin centers

        # compute score distributions (normalised histogram)
        y1 = np.histogram(matedScores, bins=bin_edges, density=True)[0]
        y2 = np.histogram(nonMatedScores, bins=bin_edges, density=True)[0]

        # Compute LR and D
        LR = np.divide(y1, y2, out=np.ones_like(y1), where=y2 != 0)
        D = 2 * (self.omega * LR / (1 + self.omega * LR)) - 1
        D[self.omega * LR <= 1] = 0
        D[y2 == 0] = 1  # this is the definition of D, and at the same time takes care of inf / nan

        # Compute and print Dsys
        Dsys = np.trapz(x=bin_centers, y=D * y1)
        return Dsys

    def plot(self, figure_file, figure_title='Unlinkability analysis', legend_loc='upper right'):
        ### Plot final figure of D + score distributions
        plt.clf()

        sns.set_context("paper", font_scale=1.7, rc={"lines.linewidth": 2.5})
        sns.set_style("white")

        ax = sns.kdeplot(self.mated_scores, shade=False, label='Mated', color=sns.xkcd_rgb["medium green"])
        x1, y1 = ax.get_lines()[0].get_data()
        ax = sns.kdeplot(self.non_mated_scores, shade=False, label='Non-Mated', color=sns.xkcd_rgb["pale red"], linewidth=5,
                         linestyle='--')
        x2, y2 = ax.get_lines()[1].get_data()

        ax2 = ax.twinx()
        lns3, = ax2.plot(self.bin_centers, self.D, label='$\mathrm{D}_{\leftrightarrow}(s)$', color=sns.xkcd_rgb["denim blue"],
                         linewidth=5)

        # print omega * LR = 1 lines
        index = np.where(self.D <= 0)
        ax.axvline(self.bin_centers[index[0][0]], color='k', linestyle='--')

        # index = np.where(LR > 1)
        # ax.axvline(bin_centers[index[0][2]], color='k', linestyle='--')
        # ax.axvline(bin_centers[index[0][-1]], color='k', linestyle='--')

        # Figure formatting
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("Score")
        ax.set_title("%s, $\mathrm{D}_{\leftrightarrow}^{\mathit{sys}}$ = %.2f" % (figure_title, self.Dsys), y=1.02)

        labs = [ax.get_lines()[0].get_label(), ax.get_lines()[1].get_label(), ax2.get_lines()[0].get_label()]
        lns = [ax.get_lines()[0], ax.get_lines()[1], lns3]
        ax.legend(lns, labs, loc=legend_loc)

        ax.set_ylim([0, max(max(y1), max(y2)) * 1.05])
        ax.set_xlim([self.bin_edges[0] * 0.98, self.bin_edges[-1] * 1.02])
        ax2.set_ylim([0, 1.1])
        ax2.set_ylabel("$\mathrm{D}_{\leftrightarrow}(s)$")

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.gcf().subplots_adjust(left=0.15)
        plt.gcf().subplots_adjust(right=0.88)
        pylab.savefig(figure_file)
