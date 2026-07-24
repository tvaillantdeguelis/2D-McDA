#!/usr/bin/env python
# coding: utf8

from datetime import datetime

import numpy as np

def print_time(tic=None): 
    """Print current time. Optionally print elapsed time from a given start time.
    If tic=None: print current time and return current time
    If start time (tic) given: print current time and elapsed time"""

    if tic:
        toc = datetime.now()
        elapsed_time = toc - tic
        time_zone = tic.astimezone().tzname()
        print(toc.strftime('\n=> End time:   %a %F %T' + ' %s' % time_zone), 
              ' (Elapsed time: ', elapsed_time, ')\n', sep='')

        return

    else:
        tic = datetime.now()
        time_zone = tic.astimezone().tzname()
        print(tic.strftime('\n=> Start time: %a %F %T' + ' %s\n' % time_zone))

        return tic


def print_elapsed_time(tic, head_string=''):
    """Print elapsed time from tic time and return current time"""

    elapsed_time = datetime.now() - tic
    print(f'{head_string}(Elapsed time: {elapsed_time})')

    return datetime.now() # tic for next

