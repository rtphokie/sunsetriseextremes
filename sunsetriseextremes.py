import unittest
from tqdm import tqdm
# from tqdm.gui import trange, tqdm
from pprint import pprint
import requests, requests_cache  # https://requests-cache.readthedocs.io/en/latest/
from skyfield import api, almanac
import pandas as pd
import plotly.express as px
import random
import threading
import simple_cache
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

requests_cache.install_cache('test_cache', backend='sqlite', expire_after=3600)
ts = api.load.timescale(builtin=True)
load = api.Loader('/var/data')
eph = load('de430t.bsp')
from pytz import timezone

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

eastern = timezone('US/Eastern')

def _earliestsunrise(lat,lng):
    foo = _earliestsunrise(lat, lng)
    return foo


# @simple_cache.cache_it(filename="earliest_sunrise.cache", ttl=1200000)
# @threaded
def earliestsunrise(lat,lng):
    observer = api.Topos(lat,lng)
    t0 = ts.utc(2020, 1, 1, 1)
    t1 = ts.utc(2020, 12, 31, 23)
    try:
        t, y = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, observer))
    except:
        pprint(observer)
        raise
    result={}
    for ti, yi in zip(t, y):
        if not yi:
            dt = ti.utc_datetime()
            ldt = dt.astimezone(eastern)
            key = ldt.strftime("%H:%M:%S")
            result[key]=ldt
    er =sorted(result.keys())[0]
    la =sorted(result.keys())[-1]
    print(er, result[er], la, result[la])
    earliest = sorted(result.keys())[0]
    return result[earliest]
    # return result[earliest]

def daylength(lat,lng):
    observer = api.Topos(lat,lng)
    t0 = ts.utc(2020, 1, 1, 1)
    t1 = ts.utc(2020, 12, 31, 23)
    try:
        t, y = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, observer, h))
    except:
        pprint(observer)
        raise
    result={}
    sunrise = None
    for ti, yi in zip(t, y):
        if yi:
            sunrise = ti.utc_datetime()
        else:
            if not sunrise:
                continue
            sunset = ti.utc_datetime()
            delta = sunset - sunrise
            result[sunset.strftime('%c')] =delta.total_seconds()
    key_max = max(result.keys(), key=(lambda k: result[k]))
    key_min = min(result.keys(), key=(lambda k: result[k]))
    print (f"min {key_min} {result[key_min]}")
    print (f"max {key_max} {result[key_max]}")
    # er =sorted(result.keys())[0]
    # la =sorted(result.keys())[-1]
    # print(er, result[er], la, result[la])
    # earliest = sorted(result.keys())[0]
    # return result[earliest]
    # return result[earliest]

class MyTestCase(unittest.TestCase):

    def test_map(self):
        lats=[33.2, 36.9]
        lngs=[-84.4, -75.2]

        df=pd.read_json('earliest_sunset_cache.json') # read calculations from cache
        lats = df['Latitude'].to_numpy()
        lngs = df['Longitude'].to_numpy()
        hrs = df['Sunset Hour'].to_numpy()
        # hrs = np.interp(df['Sunset Hour'].to_numpy(), (df['Sunset Hour'].min(), df['Sunset Hour'].max()), (1, 0))
        earliest=df['Sunset Hour'].min()
        latest=df['Sunset Hour'].max()


        fig = plt.figure(num=None, figsize=(12, 8))
        m = Basemap(projection='mill', lat_ts=10, llcrnrlon=min(lngs),
                    urcrnrlon=max(lngs), llcrnrlat=min(lats), urcrnrlat=max(lats), \
                    resolution='h')
        m.drawmapboundary(fill_color='#fff')
        m.drawcountries(linewidth=2, linestyle='solid', color='k')
        m.drawstates(linewidth=0.8, linestyle='solid', color='k')
        # m.drawrivers(linewidth=0.5, linestyle='solid', color='blue')
        m.drawcoastlines(linewidth=0.5)
        m.fillcontinents(color='#666', lake_color='#136')

        x, y = m(lngs, lats)
        # plt.savefig('ncmap.png', dpi=300)
        cmap = matplotlib.cm.autumn

        labels=[]
        lons = [-78.6382, -77.8868, -83.3206, -80.8431, -77.79, -77.3664, -82.5515, -80.2442, -81.6746]
        lats = [35.7796,34.2104, 35.4771, 35.2271, 35.9382, 35.6127, 35.5951, 36.0999,36.2168]
        cities = ['Raleigh', 'Wilmington', 'Cherokee', 'Charlotte', 'Rocky Mount', 'Greenville',
                  'Asheville', 'Winston-Salem', 'Boone']
        xcities, ycities = m(lons, lats)
        m.plot(xcities, ycities, 'bo', markersize=6, c='black')
        for city, lon, lat in zip(cities, lons, lats):
            es = earliestsunrise(lat, lon)
            print(city, es)
            labels.append(f"{city}\n{es.strftime('%-I:%M:%S %p')}")

        for label, xpt, ypt in zip(labels, xcities, ycities):
            if label in ['Wilmington', 'Mt Airy', 'Cherokee']:
                yspacing=-5000
                print('hit')
            else:
                yspacing=100
            plt.text(xpt + 10000, ypt + yspacing,label, c='black', fontsize=10)

        m.scatter(x, y, c=hrs, zorder =10, marker='s', cmap='gnuplot', s=25.0, alpha=.25)
        cbar = m.colorbar(location='bottom', label='December 5 sunset')

        cbar.set_ticks(np.arange(16.0, 18.0, 5/60.0))
        cbar.ax.invert_yaxis()


        plt.savefig('ncsearliestsunset.png', dpi=300)

    def testextremes(self):
        lons = [ -75.467412, -75.594534, -84.0346, -78.6382]
        lats = [ 35.593802,  35.908977, 35.0876, 35.7796]
        cities = ['Rodanthe', 'Nags Head', 'Murphy', 'Raleigh']
        for city, lon, lat in zip(cities, lons, lats):
            es = earliestsunrise(lat, lon)
            print(city, es)


    def test_sunsets(self):
        stuff=[]
        lats=(33.842316, 36.588117)
        lngs=(-84.32186, -75.460621)
        processes={}
        maxprocesses=6
        cache=pd.read_json('foo.json')
        latrange = np.arange(lats[0], lats[1], .01)
        lngrange = np.arange(lngs[0], lngs[1], .01)
        for lat2check in tqdm(latrange, desc='lat', leave=True):
            for lng2check in lngrange:
                lat=round(lat2check,2)
                lng=round(lng2check,2)
                if ((cache['Latitude'] == lat) & (cache['Longitude'] == lng)).any():
                    hour = cache.loc[(cache['Latitude'] == lat) & (cache['Longitude'] == lng)]['Sunset Hour'].values[0]
                    stuff.append({'Latitude': lat, 'Longitude': lng, 'Sunset Hour': round(hour, 2)})
                    # print('cache')
                    continue
                y = earliestsunrise(lat, lng)
                processkey = f"{lat:10}{lng:10}"
                processes[processkey] = y
                if len(processes) >= maxprocesses or (lat == latrange[-1] and lng == lngrange[-1]):
                    for processkey, process in processes.items():
                        foo = process.result_queue.get()
                        # foo = earliestsunrise(lat/10.0, lng/10.0)
                        hour = foo.hour + foo.minute / 60.0 + foo.second / (60.0 * 60)
                        stuff.append({'Latitude': lat , 'Longitude': lng , 'Sunset Hour': round(hour,3)})
                    processes = {}
            df = pd.DataFrame.from_dict(stuff)
            cache=pd.concat([df,cache]).drop_duplicates(["Latitude", "Longitude", 'Sunset Hour']).reset_index(drop=True)
            cache.sort_values(by=["Latitude", "Longitude"])
            cache.to_json('foo.json', indent=4)




        plt.figure(0)

        map.drawcoastlines()
        map.readshapefile('comarques', 'comarques')

        map.hexbin(array(x), array(y))
        plt.show()
        #
        # map.colorbar(location='bottom')
        #
        # plt.figure(1)
        #
        # map.drawcoastlines()
        # map.readshapefile('../sample_files/comarques', 'comarques')
        #
        # map.hexbin(array(x), array(y), gridsize=20, mincnt=1, cmap='summer', bins='log')
        #
        # map.colorbar(location='bottom', format='%.1f', label='log(# lightnings)')
        #
        # plt.figure(2)
        #
        # map.drawcoastlines()
        # map.readshapefile('../sample_files/comarques', 'comarques')
        #
        # map.hexbin(array(x), array(y), gridsize=20, mincnt=1, cmap='summer', norm=colors.LogNorm())
        #
        # cb = map.colorbar(location='bottom', format='%d', label='# lightnings')
        #
        # cb.set_ticks([1, 5, 10, 15, 20, 25, 30])
        # cb.set_ticklabels([1, 5, 10, 15, 20, 25, 30])
        #
        # plt.figure(3)
        #
        # map.drawcoastlines()
        # map.readshapefile('../sample_files/comarques', 'comarques')
        #
        # map.hexbin(array(x), array(y), C=array(c), reduce_C_function=max, gridsize=20, mincnt=1, cmap='YlOrBr',
        #            linewidths=0.5, edgecolors='k')
        #
        # map.colorbar(location='bottom', label='Mean amplitude (kA)')

        plt.show()
    def test_something(self):
        df=pd.read_json('foo.json')
        print (len(df))
        fig = px.density_mapbox(df, lat='Latitude', lon='Longitude', z='Sunset Hour',radius=20,
                                center=dict(lat=35, lon=-80), zoom=6, opacity=.5,
                                mapbox_style="stamen-terrain")
        # fig.update_layout(
        #     mapbox_style="white-bg",
        #     mapbox_layers=[
        #         {
        #             "below": 'traces',
        #             "sourcetype": "raster",
        #             "sourceattribution": "United States Geological Survey",
        #             "source": [
        #                 "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
        #             ]
        #         }
        #     ])
        # fig.update_geos(
        #     visible=False,
        #     showcountries=True, showstates=True, countrycolor="RebeccaPurple"
        # )
        fig.show()

    def test_heatmap(self):
        # example showing how to plot scattered data with hexbin.
        from numpy.random import uniform, arange
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.basemap import Basemap

        # create north polar stereographic basemap
        m = Basemap(lon_0=270, boundinglat=20, projection='npstere', round=True)
        # m = Basemap(lon_0=-105,lat_0=40,projection='ortho')

        # number of points, bins to plot.
        npts = 50000
        bins = 30

        # generate random points on a sphere,
        # so that every small area on the sphere is expected
        # to have the same number of points.
        # http://mathworld.wolfram.com/SpherePointPicking.html
        u = uniform(0., 1., size=npts)
        v = uniform(0., 1., size=npts)
        lons = 360. * u
        lats = (180. / np.pi) * np.arccos(2 * v - 1) - 90.
        # toss points outside of map region.
        lats = np.compress(lats > 20, lats)
        lons = np.compress(lats > 20, lons)
        # convert to map projection coordinates.
        x1, y1 = m(lons, lats)
        # remove points outside projection limb.
        x = np.compress(np.logical_or(x1 < 1.e20, y1 < 1.e20), x1)
        y = np.compress(np.logical_or(x1 < 1.e20, y1 < 1.e20), y1)
        # function to plot at those points.
        xscaled = 4. * (x - 0.5 * (m.xmax - m.xmin)) / m.xmax
        yscaled = 4. * (y - 0.5 * (m.ymax - m.ymin)) / m.ymax
        z = xscaled * np.exp(-xscaled ** 2 - yscaled ** 2)

        # make plot using hexbin
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(122)
        CS = m.hexbin(x, y, C=z, gridsize=bins, cmap=plt.cm.jet)
        # draw coastlines, lat/lon lines.
        # m.drawcoastlines()
        m.drawparallels(np.arange(0, 81, 20))
        m.drawmeridians(np.arange(-180, 181, 60))
        m.colorbar(location="bottom", label="Z")  # draw colorbar
        plt.title('hexbin', fontsize=20)

        # use histogram2d instead of hexbin.
        ax = fig.add_subplot(121)
        # remove points outside projection limb.
        bincount, xedges, yedges = np.histogram2d(x, y, bins=bins)
        mask = bincount == 0
        # reset zero values to one to avoid divide-by-zero
        bincount = np.where(bincount == 0, 1, bincount)
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=z)
        H = np.ma.masked_where(mask, H / bincount)
        # set color of masked values to axes background (hexbin does this by default)
        palette = plt.cm.jet
        palette.set_bad(ax.get_axis_bgcolor(), 1.0)
        CS = m.pcolormesh(xedges, yedges, H.T, shading='flat', cmap=palette)
        # draw coastlines, lat/lon lines.
        m.drawcoastlines()
        m.drawparallels(np.arange(0, 81, 20))
        m.drawmeridians(np.arange(-180, 181, 60))
        m.colorbar(location="bottom", label="Z")  # draw colorbar
        plt.title('histogram2d', fontsize=20)

        plt.gcf().set_size_inches(18, 10)
        plt.show()




if __name__ == '__main__':
    unittest.main()
