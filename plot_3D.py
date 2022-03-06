import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import os
import pandas as pd


class Plot3DArray:
    PLOT_CONST = 13.24/19.4
    def __init__(self, output_dir=os.path.join(os.getcwd(), "imgfiles")):

        self.output_dir=output_dir
        self.max_digit = 4
        self.plotted_img_paths = []


    def _plot_map(self, alpha, mu, z, z_label, z_fn, figure_size, rival, cmap):
        """
        Resources:
            - plot_surface() example and old doc:
                https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html
            - cmap: the color set for meshcolor.
                https://matplotlib.org/stable/tutorials/colors/colormaps.html
        """
        plt.figure(figsize=(figure_size, figure_size*Plot3DArray.PLOT_CONST), dpi=150)
        ax = plt.axes(projection="3d")
        # ax.set_title(r"Rivalness ($\lambda$): "+"{:.1f}".format(rival))

        # plot surface
        surf = ax.plot_surface(alpha, mu, z, cmap=cmap,
                               rstride=1, cstride=1, vmin=-0.2, vmax=1.0,
                               linewidth=1, edgecolor="black")
        
        # set axis
        ax.set_zlabel(z_label)
        ax.set_zlim(0.0, 1.0)
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter('{x:.02f}')

        ax.set_xlabel(r"Cohensiveness ($\alpha$)")
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_locator(LinearLocator(11))
        ax.xaxis.set_major_formatter('{x:.01f}')
        ax.invert_xaxis()

        ax.set_ylabel(r"Incentive ($\mu$)")
        ax.set_ylim(0.0, 50.0)
        ax.yaxis.set_major_locator(LinearLocator(11))
        ax.yaxis.set_major_formatter('{x:0.0f}')

        # set box size
        x_scale=1
        y_scale=1
        z_scale=0.5

        scale = np.diag([x_scale, y_scale, z_scale, 1.0])
        scale = scale*(1.0/scale.max())
        scale[3,3] = 1.0

        def short_proj():
            return np.dot(Axes3D.get_proj(ax), scale)

        ax.get_proj=short_proj

        # adjust camera angle
        ax.view_init(elev=45.0, azim=335)
        
        # adjust padding
        plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=1.15, wspace=0, hspace=0)
        self._save_fig(self.output_dir, filename+"_{}.png".format(z_fn))
        plt.close()


    def plot_map(self, filename, cmap="gray", figure_size=13):
        filename_prefix = os.path.splitext(filename)[0]
        lambda_rival = float(filename_prefix.split("_")[1])
        
        # process data
        file_df = pd.read_csv(filename)
        file_df.sort_values(by=["alpha", "mu"], ascending=[True, True],
                            ignore_index=True, inplace=True)
        alpha = file_df["alpha"].to_numpy().reshape((50, 51))
        mu = file_df["mu"].to_numpy().reshape((50, 51))
        part = file_df["participation"].to_numpy().reshape((50, 51))
        pro = file_df["promote"].to_numpy().reshape((50, 51))
        opp = file_df["oppose"].to_numpy().reshape((50, 51))

        # draw part
        self._plot_map(alpha, mu, part, r"Participation", "part",
                       figure_size=figure_size, rival=lambda_rival, cmap=cmap)
        self._plot_map(alpha, mu, pro, r"Promoting", "pro",
                       figure_size=figure_size, rival=lambda_rival, cmap=cmap)
        self._plot_map(alpha, mu, opp, r"Opposing", "opp",
                       figure_size=figure_size, rival=lambda_rival, cmap=cmap)
        

    def _save_fig(self, output_dir, fn):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, fn)
        plt.savefig(file_path)
        print("figrue saved to {}".format(file_path))
        return file_path




if __name__ == "__main__":
    for rival in np.arange(0.0, 1.01, 0.1):
        filename = "lambda_{:.1f}_rndSeed_1025_nRepli_10.csv".format(rival)
        plotter = Plot3DArray()
        plotter.plot_map(filename)
    