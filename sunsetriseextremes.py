import unittest
from pprint import pprint
from skyfield import api, almanac
import pandas as pd
import simple_cache
import numpy as np
pd.set_option('display.max_columns', 999)
import datetime
import isodate
from pytz import timezone
import argparse

ts = api.load.timescale(builtin=True)
load = api.Loader('/var/data')
equinoxen = ['Vernal', 'Autumnal']

utcnow = datetime.datetime.utcnow()


@simple_cache.cache_it(filename=".seasons.cache", ttl=1200000)
def get_seasons(timezone_name, year):
    ts = api.load.timescale(builtin=True)
    load = api.Loader('/var/data')
    eph = load('de430t.bsp')
    localtz = timezone(timezone_name)
    t0 = ts.utc(year)  # midnight Jan 1
    t1 = ts.utc(year + 1)  # midnight Jan 1 following year
    t, y = almanac.find_discrete(t0, t1, almanac.seasons(eph))
    season = {}
    for yi, ti in zip(y, t):
        season[almanac.SEASON_EVENTS[yi]] = ti.utc_datetime().astimezone(localtz)
    eph.close()
    return season


def sunsetriseextremes(lat, lng, timezone_name, year=None):
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
            days.append({'risedt': risedt.isoformat(), 'riset': risedt.astimezone(localtz).time(),
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
        daylightlength=equaluxdata['deltat'] / np.timedelta64(1, 's')
        record = {'dt': isodate.parse_datetime(equaluxdata['deltafrom12hrsdt']).astimezone(localtz),
                  'value': datetime.timedelta(seconds=float(daylightlength))}
        if record['dt'].strftime('%m')=='03':
            result['equilux']['Vernal'] = record
        elif record['dt'].strftime('%m') == '09':
            result['equilux']['Autumnal'] = record
        else:
            raise ValueError(f'calculated equalux {record["dt"]} outside of March or September')

    eph.close()
    return result

if __name__ == '__main__':

    tzname = 'US/Eastern'
    coords = (35.7796, -78.6382)
    foo = sunsetriseextremes(coords[0], coords[1], tzname)

    print(f"Sun extremes for {city} {coords} in the {tzname} timezone\n")
    for event in ['rise', 'set']:
        print(f"Sun{event}")
        print(f"  earliest {foo[event]['min'].strftime('%-I:%M:%S %p %Z')} on {foo[event]['min'].strftime('%b %d, %Y')}")
        print(f"  latest   {foo[event]['max'].strftime('%-I:%M:%S %p %Z')} on {foo[event]['min'].strftime('%b %d, %Y')}")
        print()
    for event in ['min', 'max']:
        td = foo['delta']['value'][event]
        hours, minutes, seconds  = td.seconds // 3600, td.seconds % 3600 // 60, (td.seconds%3600)%60
        print(f"{event}imum daylight {foo['delta'][event].strftime('%b %d, %Y')}, {hours} hours {minutes} minutes {seconds} seconds")
    print()

    for equilux, data in foo['equilux'].items():
        td = foo['equilux'][equilux]['value']
        hours, minutes, seconds  = td.seconds // 3600, td.seconds % 3600 // 60, (td.seconds%3600)%60
        print(f"{equilux:8} equilux {foo['equilux'][equilux]['dt'].strftime('%b %d, %Y')}, {hours} hours {minutes} minutes {seconds} seconds")


