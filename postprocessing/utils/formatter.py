import argparse
import sys
import matplotlib as mpl

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
        mpl.font_manager.FontProperties(fname=r"C:\Users\tyc1g20\Downloads\Helvetica_Font_Family\Helvetica 400.ttf")
        legendsize = 6
        small_font = 6
        medium_font = 8
        large_font = 10

        lines_width = 0.4
        spines_width = 0.5
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        mpl.rcParams['font.family'] = 'Helvetica'
        mpl.rcParams['font.size'] = medium_font
        mpl.rcParams['legend.frameon'] = False
        mpl.rcParams['legend.fontsize'] = medium_font
        mpl.rcParams['lines.linewidth'] = lines_width
        mpl.rcParams['axes.labelsize'] = medium_font
        mpl.rcParams['axes.titlesize'] = large_font
        mpl.rcParams['axes.titleweight'] = "bold"
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.grid'] = False
        mpl.rcParams['axes.grid.which'] = 'both'
        mpl.rcParams['grid.linewidth'] = 0.3
        mpl.rcParams['grid.color'] = 'D3D3D3'
        mpl.rcParams['axes.linewidth'] =spines_width
        mpl.rcParams['ytick.labelsize'] = medium_font
        mpl.rcParams['ytick.major.size'] = 1
        mpl.rcParams['ytick.minor.size'] = 1
        mpl.rcParams['ytick.major.width'] = spines_width
        mpl.rcParams['ytick.direction'] = "in"
        mpl.rcParams['xtick.major.width'] = spines_width
        mpl.rcParams['xtick.labelsize'] = medium_font
        mpl.rcParams['xtick.major.size'] = 1
        mpl.rcParams['xtick.minor.size'] = 1
        mpl.rcParams['xtick.direction'] = "in"
        mpl.rcParams['figure.subplot.hspace'] = 0.4  # Adjust the vertical spacing
        mpl.rcParams['figure.subplot.wspace'] = 0.4  # Adjust the horizontal spacing
        mpl.rcParams['savefig.dpi'] = 400
        mpl.rcParams['savefig.bbox'] = 'tight'
        mpl.rcParams['scatter.marker'] = 'x'
        mpl.rcParams['lines.markersize'] = 5
        

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