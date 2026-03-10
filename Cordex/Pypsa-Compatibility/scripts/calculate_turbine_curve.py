# cd2es - covert cordex climate data to energy system input data
# Copyright (C) 2024 Leonie Sara Plaga

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""1. Determine wind turbine parameters based on whether it is offshore or onshore.
2. Calculate wind capacity factors for different wind speeds and heights.
3. Fit a curve to the calculated capacity factors using ``curve_fit``.
4. Save the optimized parameters to a file specified in ``snakemake.output[0]``.

Variables:
    ``config`` (yaml): Snakemake configuration
    ``hub_height`` (float): Height of the wind turbine hub
    ``v_in`` (float): Cut-in wind speed
    ``v_r`` (float): Rated wind speed
    ``v_out`` (float): Cut-out wind speed
    ``height`` (float): Height for sfcWind calculation
    ``cf_wind``(dict): Dictionary to store wind capacity factors

Returns:
    dict: wind speed with corrsponding capacity factors
"""

from scipy.optimize import curve_fit
import numpy as np
import pickle

config = snakemake.config

if 'offshore' in snakemake.output[0]:
    hub_height = config['offwind']['hub_height']
    v_in = config['offwind']['v_in']
    v_r = config['offwind']['v_r']
    v_out = config['offwind']['v_out']
else:
    hub_height = config['onwind']['hub_height']
    v_in = config['onwind']['v_in']
    v_r = config['onwind']['v_r']
    v_out = config['onwind']['v_out']

height = 10  # for sfcWind

cf_wind = {}

for v in np.arange(0, 40, 0.01):
    def power_curve(v, v_in, v_r, v_out):
        """Calculated capacity factor for wind turbine with standardized production function.

        Args:
            v (float): wind speed
            v_in (float): cut in velocity
            v_r (float): rated velocity
            v_out (float): cut out velocity

        Returns:
            float: capacity factor
        """        
        if v < v_in:
            return 0
        elif v < v_r:
            return (v**3-v_in**3)/(v_r**3-v_in**3)
        elif v < v_out:
            return 1
        else:
            return 0

    sigma = 0.6 + 0.2 * v
    v_height = v * (hub_height/height)**(1/7)

    cf = 0
    weight = 0
    for n in np.arange(-4*sigma, +4*sigma, 0.01):
        cf += power_curve(v_height-n, v_in, v_r, v_out) / \
            np.sqrt(2*np.pi*sigma)*np.exp(-n**2/2/sigma**2)
        weight += 1/np.sqrt(2*np.pi*sigma)*np.exp(-n**2/2/sigma**2)

    cf_wind[str(np.round(v, 2))] = cf/weight

# fit function


def f(x, a, b, c, d):    
    """fit function for turbine curve

    Args:
        x (float): wind speed
        a, b, c, d (float): constant from fit in calculate turbine curve

    Returns:
       float: capacity factor of turbine at wind speed x
    """    
    return np.exp(-x**2*a)*(b*x+c*x**2+d*x**3)


popt, pcov = curve_fit(f, np.arange(0, 40, 0.01), list(
    cf_wind.values()), p0=[0.01, -0.047,  0.014,  0.0016])

file = open(snakemake.output[0], 'wb')
pickle.dump(popt, file)
file.close()
