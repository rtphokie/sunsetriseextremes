import unittest
import datetime
from sunsetriseextremes import sunsetriseextremes, get_seasons
import os



class MyTestCase(unittest.TestCase):
    def setUp(self):
        # clear cache
        try:
            os.remove('.sun_superlatives.cache')
        except OSError:
            pass

    def test_sunriseset_extremes(self):
        foo = sunsetriseextremes(35.78, -78.64, "US/Eastern")
        self.assertEqual('06 12', foo['rise']['min'].strftime('%m %d'))
        self.assertEqual('AM', foo['rise']['min'].strftime('%p'))
        self.assertEqual('10 31', foo['rise']['max'].strftime('%m %d'))
        self.assertEqual('AM', foo['rise']['max'].strftime('%p'))

        self.assertEqual('12 05', foo['set']['min'].strftime('%m %d'))
        self.assertEqual('PM', foo['set']['min'].strftime('%p'))
        self.assertEqual('06 28', foo['set']['max'].strftime('%m %d'))
        self.assertEqual('PM', foo['set']['max'].strftime('%p'))


    def test_longest_shortest_days(self):
        year=2020
        timezone_name="US/Eastern"
        foo = sunsetriseextremes(35.7796, -78.6382, timezone_name, year=year)
        season = get_seasons(timezone_name, year)

        # min/max daylight should fall on the solstices
        self.assertEqual(foo['delta']['min'].strftime('%m %d'), season['Winter Solstice'].strftime('%m %d'))
        self.assertEqual(foo['delta']['max'].strftime('%m %d'), season['Summer Solstice'].strftime('%m %d'))


    def test_equiluxes(self):
        year=2020
        timezone_name="US/Eastern"
        foo = sunsetriseextremes(35.7796, -78.6382, timezone_name, year=year)
        season = get_seasons(timezone_name, year)

        # calculated equilux days should be within a few days of the equinox.
        for equinox in foo['equilux'].keys():
            self.assertTrue(season[f'{equinox} Equinox'] - datetime.timedelta(days=7) <=
                                foo['equilux'][equinox]['dt'] <=
                                     season[f'{equinox} Equinox'] + datetime.timedelta(days=7))


if __name__ == '__main__':
    unittest.main()
