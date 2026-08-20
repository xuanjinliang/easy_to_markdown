import argparse
import warnings


class CLIDeprecationWarning(DeprecationWarning):
    pass


class DeprecatedOptionAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        assert option_string
        warnings.warn(
            f"The option `{option_string}` has been deprecated and will be removed in the future. Please refer to the documentation for more details.",
            CLIDeprecationWarning,
        )
        setattr(namespace, self.dest, values)


def warn_deprecated_param(name, new_name=None):
    msg = (
        f"The parameter `{name}` has been deprecated and will be removed in the future."
    )
    if new_name is not None:
        msg += f" Please use `{new_name}` instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
