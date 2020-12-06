import unittest
from pprint import pprint
from skyfield import api, almanac
import pandas as pd
import threading
import simple_cache
import numpy as np
pd.set_option('display.max_columns', 999)
import warnings
import datetime
import isodate
from pytz import timezone

ts = api.load.timescale(builtin=True)
load = api.Loader('/var/data')
equinoxen = ['Vernal', 'Autumnal']


def threaded(f, daemon=False):
    import queue

    def wrapped_f(q, *args, **kwargs):
        '''this function calls the decorated function and puts the
        result in a queue'''
        ret = f(*args, **kwargs)
        q.put(ret)

    def wrap(*args, **kwargs):
        '''this is the function returned from the decorator. It fires off
        wrapped_f in a new thread and returns the thread object with
        the result queue attached'''

        q = queue.Queue()

        t = threading.Thread(target=wrapped_f, args=(q,)+args, kwargs=kwargs)
        t.daemon = daemon
        t.start()
        t.result_queue = q
        return t

    return wrap

utcnow = datetime.datetime.utcnow()


def sunsuperlatives(lat, lng, timezone_name, year=None):
    foo = _sunsuperlatives(round(lat, 2), round(lng, 2), timezone_name, year)
    return foo


@simple_cache.cache_it(filename=".sun_superlatives.cache", ttl=1200000)
def _sunsuperlatives(lat, lng, timezone_name, year):
    eph = load('de430t.bsp')
    if year is None:
        year=utcnow.year
    result = {}
    observer = api.Topos(lat,lng)
    localtz = timezone(timezone_name)

    t0 = ts.utc(year) # midnight Jan 1
    t1 = ts.utc(year+1)  # midnight Jan 1 following year
    try:
        t, y = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, observer))
    except Exception as e:
        pprint(observer)
        raise ValueError(f"error calculating sunrise_sunset for {observer}: {e}")

    # build list of 365 rises, sets and day length
    days=[]
    for ti, yi in zip(t, y):
        dt = ti.utc_datetime()
        ldt = dt.astimezone(localtz)
        key = ldt.strftime("%H:%M:%S")
        if yi:
            risedt=dt
        else:
            #store the datetime of sunrise, sunset, time the Sun is above the horizon, and how close that is to 12 hours
            #avoid pandas conversion datetime64 and its timezone complexities with iso format
            days.append({'risedt': dt.isoformat(), 'riset': risedt.astimezone(localtz).time(),
                         'setdt': dt.isoformat(), 'sett': dt.astimezone(localtz).time(),
                         'deltadt': dt.isoformat(), 'deltat': dt-risedt, # time the Sun is above the horizon
                         'deltafrom12hrsdt': dt.isoformat(), 'deltafrom12hrst': abs(pd.Timedelta('12 hours')-(dt-risedt))})

    df = pd.DataFrame(days)
    for k in ['rise', 'set', 'delta']:
        # convert ISO8601 datetime string to datetime in specified timezone
        mintimeiso = df[df[f'{k}t'] == df[f'{k}t'].min()][f'{k}dt'].values[0]
        maxtimeiso = df[df[f'{k}t'] == df[f'{k}t'].max()][f'{k}dt'].values[0]
        result[k] = {'min': isodate.parse_datetime(mintimeiso).astimezone(localtz),
                     'max': isodate.parse_datetime(maxtimeiso).astimezone(localtz)}
        if k in ['delta']:
            # convert pandas.timedelta back to datetime.timedelta retaining microsecond resolution
            mindelta = df[df[f'{k}t'] == df[f'{k}t'].min()][f'{k}t'].values[0] / np.timedelta64(1, 's')
            maxdelta = df[df[f'{k}t'] == df[f'{k}t'].max()][f'{k}t'].values[0] / np.timedelta64(1, 's')
            result[k]['value'] = {'min': datetime.timedelta(seconds=float(mindelta)),
                                  'max': datetime.timedelta(seconds=float(maxdelta))}
    result['equilux'] = {}
    # There are 2 days with close to 12 hours of daylight, around the equinoxes
    for i, equaluxdata in df.nsmallest(2, 'deltafrom12hrst').reset_index().iterrows():
        result['equilux'][equinoxen[i]]={'dt': isodate.parse_datetime(equaluxdata['deltafrom12hrsdt']).astimezone(localtz),
                                  'value': equaluxdata['deltafrom12hrst'] / np.timedelta64(1, 's')
                                  }
    eph.close()
    return result

if __name__ == '__main__':
    foo = sunsuperlatives(35.7796, -78.6382, 'US/Eastern')
    pprint(foo)
