"""histoqc.data

helpers for providing data resources for histoqc modules

"""
import os
import re
from configparser import ConfigParser
from contextlib import ContextDecorator
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType

try:
    from importlib.resources import files as _files
except ImportError:
    from importlib_resources import files as _files


# --- dispatch config rewriting ---------------------------------------

# noinspection PyPep8Naming
class _ManagedPkgData(ContextDecorator):

    _section_re = re.compile(
        r'(?P<module>[A-Za-z_][A-Za-z0-9_]+[.][A-Za-z_][A-Za-z0-9_]+)'
        r'([:](?P<label>[A-Za-z_][A-Za-z0-9_-]))?'
    )

    def __init__(self):
        self._stack = None
        self._tmp_dir = None

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)
        self._tmp_dir = None
        self._stack = None

    def get_tmp_dir(self):
        if self._tmp_dir is None:
            self._tmp_dir = self._stack.enter_context(
                TemporaryDirectory(prefix=".histoqc_pkg_data_tmp", dir=os.getcwd())
            )
            for rsrc in {'models', 'pen', 'templates'}:
                package_resource_copytree('histoqc.data', rsrc, self._tmp_dir, None)
        return self._tmp_dir

    def inject_pkg_data_fallback(self, config: ConfigParser):
        """support template data packaged in module"""
        for section in config.sections():
            m = self._section_re.match(section)
            if not m:
                continue

            # dispatch
            mf = m.group('module')
            s = config[section]
            if mf == 'HistogramModule.compareToTemplates':
                self._inject_HistogramModule_compareToTemplates(s)

            elif mf == 'ClassificationModule.byExampleWithFeatures':
                self._inject_ClassificationModule_byExampleWithFeatures(s)

            else:
                pass

    def _inject_HistogramModule_compareToTemplates(self, section):
        # replace example files in a compareToTemplates config section
        # with the histoqc package data examples if available
        if 'templates' in section:
            _templates = []
            for template in map(str.strip, section['templates'].split('\n')):
                f_template = os.path.join(os.getcwd(), template)

                if not os.path.isfile(f_template):
                    tmp = self.get_tmp_dir()
                    f_template_pkg_data = os.path.join(tmp, template)
                    if os.path.isfile(f_template_pkg_data):
                        f_template = f_template_pkg_data

                _templates.append(f_template)
            section['templates'] = '\n'.join(_templates)

    def _inject_ClassificationModule_byExampleWithFeatures(self, section):
        # replace template files in a byExampleWithFeatures config section
        # with the histoqc package data templates if available
        if 'examples' in section:
            # mimic the code structure in ClassificationModule.byExampleWithFeatures,
            # which expects example templates to be specified as pairs.
            # each template in the pair is separated by a ':' and pairs are separated by a newline.
            _lines = []
            for ex in section["examples"].splitlines():
                ex = re.split(r'(?<!\W[A-Za-z]):(?!\\)', ex)  # workaround for windows: don't split on i.e. C:backslash
                _examples = []
                for example in ex:
                    f_example = os.path.join(os.getcwd(), example)

                    if not os.path.isfile(f_example):
                        tmp = self.get_tmp_dir()
                        f_example_pkg_data = os.path.join(tmp, example)
                        if os.path.isfile(f_example_pkg_data):
                            f_example = f_example_pkg_data

                    _examples.append(f_example)
                _line = ":".join(_examples)
                _lines.append(_line)
            section['examples'] = "\n".join(_lines)


managed_pkg_data = _ManagedPkgData()


# --- helper functions ------------------------------------------------

def package_resource_copytree(package, resource, dest, result):
    """helper for copying a package resource to a destination

    Parameters
    ----------
    package :
        a package as accepted by importlib.resources
    resource :
        a package resource (does support directories)
    dest :
        a destination directory

    """
    def _traverse_copy(traversable, root):
        # copy an importlib.resources Traversable structure
        # to a pathlib.Path directory recursively
        assert root.is_dir()

        if isinstance(traversable.name, MethodType):
            # workaround for https://bugs.python.org/issue43643
            name = traversable.name()
        else:
            name = traversable.name

        pth = root.joinpath(name)

        if traversable.is_file():
            pth.write_bytes(traversable.read_bytes())
            
            # add default result file name as global variable
            if result and name == 'global_config.js':
                with open(pth, 'a') as f:
                    f.write(f"// Default result file name \nvar DEFAULT_RESULT_FILE_NAME = \"{result}\";")
                     
        elif traversable.is_dir():
            pth.mkdir(exist_ok=True)
            for t in traversable.iterdir():
                _traverse_copy(t, pth)

    rsrc = _files(package) / resource
    out_path = Path(dest)
    _traverse_copy(rsrc, out_path)

