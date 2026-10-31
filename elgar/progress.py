"""Batch progress and delayed status for long indivisible operations."""
from contextlib import contextmanager
from threading import Event, Thread
from time import perf_counter, time
import sys

from tqdm.auto import tqdm


def progress(iterable, *, desc, **kwargs):
    return tqdm(iterable, desc=desc, dynamic_ncols=True, disable=None, **kwargs)


@contextmanager
def stage(description):
    """Show status only if an operation lasts more than two seconds."""
    started = perf_counter()
    stopped = Event()
    bars = []
    shown = Event()

    def refresh():
        if stopped.wait(2):
            return
        shown.set()
        if sys.stderr.isatty():
            bar = tqdm(desc=description, total=None, dynamic_ncols=True,
                       bar_format='{desc} | elapsed {elapsed} {postfix}')
            bar.start_t = time() - (perf_counter() - started)
            bars.append(bar)
            while not stopped.wait(1):
                bar.refresh()
        else:
            print(f'{description}: running', flush=True)
            while not stopped.wait(30):
                print(f'{description}: elapsed {perf_counter()-started:.0f}s', flush=True)

    worker = Thread(target=refresh, daemon=True)
    worker.start()
    success = False
    try:
        yield
        success = True
    finally:
        stopped.set()
        worker.join()
        status = 'done' if success else 'failed'
        for bar in bars:
            bar.set_postfix_str(status)
            bar.close()
        if shown.is_set() and not bars:
            print(f'{description}: {status} in {perf_counter()-started:.1f}s', flush=True)
