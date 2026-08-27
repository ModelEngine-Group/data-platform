"""histoqc.config

helpers for providing configuration ini files for histoqc

"""
import re
try:
    import importlib.resources as _resources
except ImportError:
    # fallback for Python < 3.7
    import importlib_resources as _resources


# note to developers:
#   to store a new ini file in the histoqc package add an ini file that
#   matches the regular expression below to this directory
CONFIG_TEMPLATE_RE = re.compile(r'config(_(?P<name>[A-Za-z][A-Za-z0-9.]+))?[.]ini')


def list_config_templates():
    """return a dictionary mapping config template names to filenames

    note: the default config is stored under name `None`
    """
    templates = {}
    for filename in _resources.contents('histoqc.config'):
        m = CONFIG_TEMPLATE_RE.match(filename)
        if m:
            templates[m.group('name') or 'default'] = filename
    return templates


def read_config_template(name=None):
    """return the contents of a configuration template"""
    templates = list_config_templates()
    if name not in templates:
        raise KeyError(f'no configuration template found under key {name!r}')
    return _resources.read_text('histoqc.config', templates[name])
