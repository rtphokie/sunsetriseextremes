import unittest
import datetime
from sunsuperlatives import sunsuperlatives, equinoxen
from skyfield import api, almanac
import pytz
import os


def get_seasons(timezone_name, year):
    ts = api.load.timescale(builtin=True)
    load = api.Loader('/var/data')
    eph = load('de430t.bsp')
    localtz = pytz.timezone(timezone_name)
    t0 = ts.utc(year)  # midnight Jan 1
    t1 = ts.utc(year + 1)  # midnight Jan 1 following year
    t, y = almanac.find_discrete(t0, t1, almanac.seasons(eph))
    season = {}
    for yi, ti in zip(y, t):
        season[almanac.SEASON_EVENTS[yi]] = ti.utc_datetime().astimezone(localtz)
    return season
class MyTestCase(unittest.TestCase):
    def setUp(self):
        # clear cache
        try:
            os.remove('.sun_superlatives.cache')
        except OSError:
            pass

    def test_sunriseset_extremes(self):
        foo = sunsuperlatives(35.78, -78.64, "US/Eastern")
        self.assertEqual('12 05', foo['set']['min'].strftime('%m %d'))
        self.assertEqual('06 28', foo['set']['max'].strftime('%m %d'))
        self.assertEqual('10 31', foo['rise']['max'].strftime('%m %d'))
        self.assertEqual('06 12', foo['rise']['min'].strftime('%m %d'))

    def test_longest_shortest_days(self):
        year=2020
        timezone_name="US/Eastern"
        foo = sunsuperlatives(35.7796, -78.6382, timezone_name, year=year)
        season = get_seasons(timezone_name, year)

        # min/max daylight should fall on the solstices
        self.assertEqual(foo['delta']['min'].strftime('%m %d'), season['Winter Solstice'].strftime('%m %d'))
        self.assertEqual(foo['delta']['max'].strftime('%m %d'), season['Summer Solstice'].strftime('%m %d'))


    def test_equiluxes(self):
        year=2020
        timezone_name="US/Eastern"
        foo = sunsuperlatives(35.7796, -78.6382, timezone_name, year=year)
        season = get_seasons(timezone_name, year)

        # calculated equilux days should be within a few days of the equinox.
        for equinox in equinoxen:
            self.assertTrue(season[f'{equinox} Equinox'] - datetime.timedelta(days=7) <=
                                foo['equilux'][equinox]['dt'] <=
                                     season[f'{equinox} Equinox'] + datetime.timedelta(days=7))


if __name__ == '__main__':
    unittest.main()
