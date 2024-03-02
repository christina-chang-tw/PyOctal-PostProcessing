import argparse
import sys


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
