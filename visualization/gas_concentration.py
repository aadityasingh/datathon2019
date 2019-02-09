import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

gas_conc = pd.read_csv('filter_pack_concentrations_weekly_cleaned.csv')
gas_by_place = gas_conc.groupby(['county', 'state_code'])

places = [key for key, item in gas_by_place]
# Pick one county to graph to reduce busyness in graph
county = places[int(random.random() * len(places))]
print(county)

county = ('Clallam', 'WA')

gases1 = ['CA', 'MG', 'NA', 'K', 'CL']
gases2 = ['TSO4', 'TNH4', 'NSO4', 'NHNO3', 'WSO2', 'WNO3']
columns = ['date_on'] + gases1 + gases2

data = gas_by_place.get_group(county)

# Graphing
fig = plt.figure()
ax = fig.add_subplot(111)

color_cycler = cycler(color='bgrcmyk')
plt.rc('axes', prop_cycle=color_cycler)

for idx in [1, 2]:
    gases_dict = {1: gases1, 2: gases2}
    for gas in gases_dict[idx]:
        plt.plot(data['date_on'].values, data[gas].values, label=gas)

    # label bars with counts
    plt.xlabel('Time')
    plt.ylabel('Concentration (uq/m^3)')
    ax.set_title('Gas concentration over time in ' + county[0] + ', ' + county[1])
    # plt.tight_layout()

    # Shrink current axis by 10% to fit legend outside of graph
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    fig.legend()

    plt.show()
    # fig.savefig('assets/gas_over_time2.png', dpi = 300)
