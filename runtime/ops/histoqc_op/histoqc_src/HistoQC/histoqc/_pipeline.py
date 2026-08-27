"""histoqc._pipeline

helper utilities for running the HistoQC pipelines

"""
import glob
import logging
import multiprocessing
import os
import platform
import shutil
import threading
import warnings
from contextlib import ExitStack
from contextlib import contextmanager
from contextlib import nullcontext
from importlib import import_module
from logging.config import dictConfig
from logging.handlers import QueueHandler


# --- logging helpers -------------------------------------------------

DEFAULT_LOG_FN = "error.log"


def setup_logging(*, capture_warnings, filter_warnings):
    """configure histoqc's logging instance

    Parameters
    ----------
    capture_warnings: `bool`
        flag if warnings should be captured by the logging system
    filter_warnings: `str`
        action for warnings.filterwarnings
    """
    dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'class': 'logging.Formatter',
                'format': '%(asctime)s - %(levelname)s - %(message)s',
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'default',
            },
            'logfile': {
                'class': 'logging.FileHandler',
                'level': 'WARNING',
                'filename': DEFAULT_LOG_FN,
                'mode': 'w',  # we initially start overwriting existing logs
                'formatter': 'default',
            },
        },
        'root': {
            'level': 'INFO',
            'handlers': ['console', 'logfile']
        }
    })

    # configure warnings too...
    warnings.filterwarnings(filter_warnings)
    logging.captureWarnings(capture_warnings)



def move_logging_file_handler(logger, destination, debug=False):
    """
    Redirect FileHandlers to a new destination directory, moving the log file.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance whose FileHandlers will be moved.
    destination : str
        Directory path for the new log files.
    debug : bool
        If True, set the log level of the new handlers to DEBUG. If False, the new handlers retain the original handler's log level.
    """
    for handler in reversed(logger.handlers):
        if not isinstance(handler, logging.FileHandler):
            continue
        if handler.baseFilename != os.path.join(os.getcwd(), DEFAULT_LOG_FN):
            continue
        if not os.path.exists(handler.baseFilename):
            logger.warning(f"Original log file {handler.baseFilename!r} does not exist, cannot move.")
            continue

        if not destination.endswith(handler.baseFilename):
            destination = os.path.join(destination, os.path.relpath(handler.baseFilename, os.getcwd()))
        logger.info(f'moving fileHandler {handler.baseFilename!r} to {destination!r}')

        # remove handler
        logger.removeHandler(handler)
        handler.close()

        new_filename = shutil.copy2(handler.baseFilename, destination)
        try:
            os.remove(handler.baseFilename)
        except OSError:
            # on WSL/Windows, file might still be locked briefly
            logger.warning(f"Could not remove original log file {handler.baseFilename!r}")

        # create and attach new handler
        new_handler = logging.FileHandler(new_filename, mode='a')
        new_handler.setLevel(logging.DEBUG if debug else handler.level)
        new_handler.setFormatter(handler.formatter)
        logger.addHandler(new_handler)


class MultiProcessingLogManager:

    def __init__(self, logger_name, *, manager):
        """create a MultiProcessingLogManager

        Note: this uses a multiprocessing Queue to correctly transfer
          logging information from worker processes to the main
          process logging instance

        Parameters
        ----------
        logger_name : str
            the name of the logger instance
        manager : multiprocessing.Manager
            the mp Manager instance used for creating sharable context
        """
        self._logger_name = logger_name
        self._log_queue = manager.Queue()
        self._log_thread_active = False

    @property
    def is_main_process(self):
        return multiprocessing.current_process().name == "MainProcess"

    @property
    def logger(self):
        """returns the logger instance"""
        if self.is_main_process:
            return logging.getLogger(self._logger_name)
        else:
            root = logging.getLogger()
            if not root.hasHandlers():
                qh = QueueHandler(self._log_queue)
                root.setLevel(logging.INFO)
                root.addHandler(qh)
                # note: this should be revisited and set by the main process
                warnings.filterwarnings('ignore')
                logging.captureWarnings(True)
            return root

    @contextmanager
    def logger_thread(self):
        """context manager for multiprocess logging

        Note: this starts the thread responsible for handing the log records
          emitted by child processes to the main logger instance
        """
        assert self.is_main_process
        assert not self._log_thread_active  # not reentrant...
        self._log_thread_active = True

        def process_queue(q, ln):
            main_logger = logging.getLogger(ln)
            while True:
                log_record = q.get()
                if log_record is None:
                    break
                main_logger.handle(log_record)

        lt = threading.Thread(target=process_queue, args=(self._log_queue, self._logger_name))
        lt.start()
        try:
            yield
        finally:
            self._log_queue.put(None)
            lt.join()
            self._log_thread_active = False


def log_pipeline(config, log_manager):
    """log the pipeline information

    Parameters
    ----------
    config : configparser.ConfigParser
    log_manager : MultiProcessingLogManager
    """
    assert log_manager.is_main_process
    steps = config.get(section='pipeline', option='steps').splitlines()

    log_manager.logger.info("the pipeline will use these steps:")
    for process in steps:
        mod_name, func_name = process.split('.')
        log_manager.logger.info(f"\t\t{mod_name}\t{func_name}")
    return steps


# --- worker process helpers ------------------------------------------

def setup_plotting_backend(logger=None, debug=False):
    """loads the correct matplotlib backend

    Parameters
    ----------
    logger :
        the logging.Logger instance
    """
    import matplotlib

    if debug and (platform.system() == "Windows" or os.environ.get('DISPLAY')):
        matplotlib.use('TkAgg')
        logger.info("Display found and debug mode enabled. Using TkAgg backend for matplotlib.")

    else:
        matplotlib.use('Agg')


class BatchedResultFile:
    """BatchedResultFile encapsulates the results writing

    Note: this is multiprocessing safe
    """
    FILENAME_GLOB = "results*.tsv"
    FILENAME_NO_BATCH = "results.tsv"
    FILENAME_BATCH = "results_{:d}.tsv"

    def __init__(self, dst, *, manager, batch_size=None, force_overwrite=False):
        """create a BatchedResultFile instance

        Parameters
        ----------
        dst : os.PathLike
            the output directory for the result files
        manager : multiprocessing.Manager
            the mp Manager instance used for creating sharable context
        batch_size : int or None
            after `batch_size` calls to increment_counter() the results
            file will be rotated
        force_overwrite : bool
            overwrite result files if they are already present. default
            is to append.
        """
        if not os.path.isdir(dst):
            raise ValueError(f"dst {dst!r} is not a directory or does not exist")
        if batch_size is not None:
            batch_size = int(batch_size)
            if batch_size < 1:
                raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self.dst = os.path.abspath(dst)
        self.batch_size = batch_size
        self.force_overwrite = bool(force_overwrite)

        # multiprocessing safety
        self._headers = manager.list()
        self._rlock = manager.RLock()

        # internal state
        self._batch = 0
        self._completed = 0
        self._first = True

        # contextmanager
        self._f = None
        self._stack = None

    def __enter__(self):
        self._stack = ExitStack()
        self._stack.callback(self.increment_counter)
        self._stack.enter_context(self._rlock)
        self._f = nullcontext(self._stack.enter_context(self._file()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.close()
        self._stack = None
        self._f = None

    def _file(self):
        if self._f is not None:
            return self._f  # we're in the context manager

        if self.batch_size is None:
            fn = self.FILENAME_NO_BATCH
        else:
            fn = self.FILENAME_BATCH.format(self._batch)
        pth = os.path.join(self.dst, fn)

        mode = "a"
        if self._first and os.path.isfile(pth):
            if self.force_overwrite:
                mode = "w"
            else:
                mode = "a"
        self._first = False
        return open(pth, mode=mode)

    def add_header(self, header):
        """add a new header to the results file

        Parameters
        ----------
        header :
            a string that can be written to the file by calling the
            write_headers method
        """
        self._headers.append(header)

    def is_empty_file(self):
        """return if the current file is empty

        Note: this is useful to determine if you want to write_headers
          ... technically the name is incorrect, but in this use case
              pos 0 is equivalent to an empty file
        """
        with self._rlock, self._file() as f:
            return f.tell() == 0

    def write_headers(self, *args):
        """write the internally collected headers to the current file

        Parameters
        ----------
        state : dict
            the current histoqc implementation writes the outputs to
            the header files, so *args supports `state` for now.
            overwrite in subclass to control header output behavior
        """
        with self._rlock:
            # write headers
            for line in self._headers:
                self.write_line(f"#{line}")
            # histoqc specific
            _state, = args
            _outputs = '\t'.join(_state['output'])
            line = f"#dataset:{_outputs}\twarnings"
            self.write_line(line)

    def write_line(self, text, end="\n"):
        """write text to the file

        Parameters
        ----------
        text : str
        end : str
            defaults to newline
        """
        with self._rlock, self._file() as f:
            f.write(text)
            if end:
                f.write(end)

    def increment_counter(self):
        """increment the completed counter

        moves to the next batch as determined by batch_size
        """
        # move on to the next batch if needed
        with self._rlock:
            self._completed += 1
            if self._completed and self.batch_size and self._completed % self.batch_size == 0:
                self._batch += 1
                self._first = True

    @classmethod
    def results_in_path(cls, dst):
        """return if a dst path contains results files

        Parameters
        ----------
        dst : os.PathLike
        """
        return bool(glob.glob(os.path.join(dst, cls.FILENAME_GLOB)))


def load_pipeline(config):
    """load functions and parameters from config

    Parameters
    ----------
    config : configparser.ConfigParser
    """
    steps = config.get(section='pipeline', option='steps').splitlines()

    process_queue = []
    for process in steps:
        mod_name, func_name = process.split('.')

        try:
            mod = import_module(f"histoqc.{mod_name}")
        except ImportError:
            raise NameError(f"Unknown module in pipeline from config file:\t {mod_name}")

        func_name = func_name.split(":")[0]  # take base of function name
        try:
            func = getattr(mod, func_name)
        except AttributeError:
            raise NameError(f"Unknown function from module in pipeline from config file:\t {mod_name}.{func_name}")

        if config.has_section(process):
            params = dict(config.items(section=process))
        else:
            params = {}

        process_queue.append((func, params))
    return process_queue
