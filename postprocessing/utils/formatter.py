import argparse
import sys
import matplotlib as mpl
import numpy as np
import string

class CustomArgparseFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """ Display default values in the helper message. """

    def _get_help_string(self, action):
        help_msg = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    if isinstance(action.default, type(sys.stdin)):
                        help_msg += ' [default: ' + str(action.default.name) + ']'
                    elif isinstance(action.default, bool):
                        help_msg += ' [default: ' + str(action.default) + ']'
                    elif isinstance(action.default, str):
                        help_msg += ' [default: ' + str(action.default) + ']'
                    elif action.default is not None:
                        help_msg += f' [default: {", ".join(map(str, action.default))}]'
        return help_msg


class Publication:
    """ Set the publication quality figure settings. """
    @staticmethod
    def set_basics():
        # font_path = r"C:\Users\cchan\Downloads\helvetica-255\Helvetica.ttf"  # Your font path goes here
        # mpl.font_manager.fontManager.addfont(font_path)
        # prop = mpl.font_manager.FontProperties(fname=font_path)

        default = 12
        larger_font = 14

        lines_width = 2
        spines_width = 0.5
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.family'] = 'Helvetica'
        mpl.rcParams['font.size'] = larger_font
        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['legend.handlelength'] = 1
        mpl.rcParams['legend.fontsize'] = default
        #change legend spacing
        mpl.rcParams['legend.labelspacing'] = 0.3
        mpl.rcParams['lines.linewidth'] = lines_width
        mpl.rcParams['axes.labelsize'] = larger_font
        mpl.rcParams['axes.titlesize'] = larger_font + 4
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['axes.grid.which'] = 'both'
        mpl.rcParams['grid.linewidth'] = 0.3
        mpl.rcParams['grid.color'] = 'D3D3D3'
        mpl.rcParams['axes.linewidth'] =spines_width
        mpl.rcParams['ytick.labelsize'] = larger_font
        mpl.rcParams['ytick.major.size'] = 1
        mpl.rcParams['ytick.minor.size'] = 1
        mpl.rcParams['ytick.major.width'] = spines_width
        mpl.rcParams['ytick.direction'] = "in"
        mpl.rcParams['xtick.major.width'] = spines_width
        mpl.rcParams['xtick.labelsize'] = larger_font
        mpl.rcParams['xtick.major.size'] = 1
        mpl.rcParams['xtick.minor.size'] = 1
        mpl.rcParams['xtick.direction'] = "in"
        mpl.rcParams['figure.subplot.hspace'] = 0.4  # Adjust the vertical spacing
        mpl.rcParams['figure.subplot.wspace'] = 0.4  # Adjust the horizontal spacing
        mpl.rcParams['savefig.dpi'] = 400
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['scatter.marker'] = 'x'
        mpl.rcParams['lines.markersize'] = 5

<<<<<<< Updated upstream
        # ax.ticklabel_format(useOffset=False, style='plain') in rcparams
        mpl.rcParams['axes.formatter.useoffset'] = False
=======
    @staticmethod
    def twin_x(ax):
        ax2 = ax.twinx()
        ax.spines['left'].set_color('C0')
        ax.spines['right'].set_color('C1')
        ax.yaxis.label.set_color('C0')
        ax.yaxis.label.set_color('C0')
        ax.tick_params(axis='y', colors='C0')

        ax2.spines['left'].set_color('C0')
        ax2.spines['right'].set_color('C1')
        ax2.yaxis.label.set_color('C1')
        ax2.yaxis.label.set_color('C1')
        ax2.tick_params(axis='y', colors='C1')
        return ax, ax2

>>>>>>> Stashed changes
        
    @staticmethod
    def twin_x(ax):
        ax2 = ax.twinx()
        ax.spines['left'].set_color('C0')
        ax.spines['right'].set_color('C1')
        ax.yaxis.label.set_color('C0')
        ax.yaxis.label.set_color('C0')
        ax.tick_params(axis='y', colors='C0')

        ax2.spines['left'].set_color('C0')
        ax2.spines['right'].set_color('C1')
        ax2.yaxis.label.set_color('C1')
        ax2.yaxis.label.set_color('C1')
        ax2.tick_params(axis='y', colors='C1')
        return ax, ax2

    @staticmethod
    def set_titles(axes, col: bool = False):
        """
        Set the title for each axes in the figure.

        Parameters
        ----------
        axes : list
            List of axes.
        up_down : bool, optional
            Set the titles by columns consecutively. The default is False.
        """
        lower = list(string.ascii_lowercase)
        if col:
            length = len(axes)
            arr1, arr2 = axes[:length//2], axes[length//2:]
            axes = [*sum(zip(arr1,arr2),())]
        for idx, ax in enumerate(axes):
            ax.set_title(f"({lower[idx]})")

    @staticmethod
    def get_one_col_figsize():
        return (9/2.54)
    
    @staticmethod
    def get_two_col_figsize():
        return (18/2.54)
    
    @staticmethod
    def cm2inch(val):
        return val/2.54
    
    @property
    def dpi(self):
        return 400