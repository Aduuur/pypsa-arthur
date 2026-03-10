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

"""Modifies backbone input Excel file with updated data in the specified sheets.

Returns:
    excel file: modified backbone excel file
"""

from os import times
import pandas as pd

config = snakemake.config

backbone_input_file = snakemake.input.backbone_file
solar_file = snakemake.input.solar_file
onwind_file = snakemake.input.wind_onshore_file
offwind_file = snakemake.input.wind_offshore_file
hydro_inflow_file = snakemake.input.hydro_inflow_file
avai_tpp_file = snakemake.input.avai_tpp_file
path_jrc = 'resources/JRC-PPDB-OPEN.ver1.0/JRC_OPEN_UNITS.csv'

solar_cf = pd.read_csv(solar_file, delimiter=';', index_col=0)
onwind_cf = pd.read_csv(onwind_file, delimiter=';', index_col=0)
offwind_cf = pd.read_csv(offwind_file, delimiter=';', index_col=0)
hydro_inflow = pd.read_csv(hydro_inflow_file, delimiter=';', index_col=0)

# change cooling types of exisiting power plants, add investment option for all cooling types

# read in cooling types
jrc_list = pd.read_csv(path_jrc)

# list of European countries and their codes
europe2Letter = ['AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU',
                 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK']
europeCountries = ['Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria', 'Switzerland', 'Czechia',
                   'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'United Kingdom', 'Greece', 'Croatia',
                   'Hungary', 'Ireland', 'Italy', 'Lithuania', 'Luxembourg', 'Latvia', 'Montenegro', 'North Macedonia',
                   'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Serbia', 'Sweden', 'Slovenia', 'Slovakia']
countryToTwo = dict(zip(europeCountries, europe2Letter))

costChange = pd.DataFrame([[config['backbone_tpps']['inv_cost_per_cooling_type']['once-through'],
                            config['backbone_tpps']['efficiency_drop_per_cooling_type']['once-through']],
                           [config['backbone_tpps']['inv_cost_per_cooling_type']['closed-loop'],
                            config['backbone_tpps']['efficiency_drop_per_cooling_type']['closed-loop']],
                           [config['backbone_tpps']['inv_cost_per_cooling_type']['dry-cooling'],
                            config['backbone_tpps']['efficiency_drop_per_cooling_type']['dry-cooling']]],
                          columns=['investment', 'efficiency'],
                          index=['once-through', 'closed-loop', 'dry-cooling'])

# read in excel sheets
unit_df = pd.read_excel(backbone_input_file, sheet_name='unit')
# unitUnittype = pd.read_excel(backbone_input_file, sheet_name='unitUnittype')
utAvailabilityLimits = pd.read_excel(
    backbone_input_file, sheet_name='utAvailabilityLimits')
effLevelGroupUnit = pd.read_excel(
    backbone_input_file, sheet_name='effLevelGroupUnit')
p_gnu_io = pd.read_excel(backbone_input_file, sheet_name='p_gnu_io')
p_unit = pd.read_excel(backbone_input_file, sheet_name='p_unit')
gnugroup = pd.read_excel(backbone_input_file, sheet_name='gnugroup')
ts_unit = pd.DataFrame()

for i, tech in enumerate(config['backbone_tpps']['types']):
    # read in shares for different cooling types at the moment
    shares = jrc_list.loc[jrc_list['type_g'] == tech].groupby(['country', 'cooling_type'])\
        .sum(numeric_only=True)['capacity_g']
    shares = shares.rename({'Mechanical Draught Tower': 'closed-loop', 'Natural Draught Tower': 'closed-loop',
                            'Air Cooling': 'dry-cooling', 'No Cooling': 'dry-cooling', 'Once-through': 'once-through'})\
        .groupby(['country', 'cooling_type']).sum(numeric_only=True).reset_index()
    # change index to 2 letter country codes as in pypsa-eur
    shares['country'] = [countryToTwo[country]
                         for country in shares['country']]
    
    if config['aggregateNodes']:
        countriesToAggregate = {}
        for key in config['nodesToAggregate'].keys():
            countriesToAggregate[key[0:2]] = [element[0:2] for element in config['nodesToAggregate'][key]]
        for country in countriesToAggregate.keys():
            for oldCountry in countriesToAggregate[country]:
                shares['country'] = shares['country'].replace(oldCountry, country)
        shares = shares.groupby(['country', 'cooling_type']).sum().reset_index().set_index('country')
    else:
        shares = shares.set_index('country')

    # calculate shares of different cooling technologies per country (in case of more than one node per country)
    shares = shares.combine_first(shares.reset_index().groupby('country').sum(numeric_only=True)
                                  .rename(columns={'capacity_g': 'total_cap'}))
    shares['share'] = shares['capacity_g']/shares['total_cap']

    # to avoid that the charge units for hydrogen are also affected
    if tech == "H2":
        techBBName = "H2 discharge"
    else:
        techBBName = tech

    # split exisiting units into three technology units
    # no dry cooling for nuclear
    if tech == 'Nuclear':
        newUnits = pd.concat([unit_df.loc[unit_df['unit'].str.contains(techBBName.lower())]+' closed-loop',
                              unit_df.loc[unit_df['unit'].str.contains(techBBName.lower())]+' once-through'], ignore_index=True)
    else:
        newUnits = pd.concat([unit_df.loc[unit_df['unit'].str.lower().str.contains(techBBName.lower())]+' dry-cooling',
                              unit_df.loc[unit_df['unit'].str.lower().str.contains(
                                  techBBName.lower())]+' closed-loop',
                              unit_df.loc[unit_df['unit'].str.lower().str.contains(techBBName.lower())]+' once-through'], ignore_index=True)

    # add new units to sheet unit
    unit_df = pd.concat(
        [unit_df.loc[~unit_df['unit'].str.lower().str.contains(techBBName.lower())], newUnits])
    
    # add new unit to gnugroup
    if gnugroup["group"].str.contains("biomass").any():
        biomass_units = newUnits.loc[newUnits["unit"].str.contains("biomass"), "unit"]
        gnugroup_groups = [f"{unit[0:2]} biomass cap group" for unit in biomass_units]
        gnugroup_new = pd.DataFrame(gnugroup_groups, index = biomass_units, columns=["group"]).reset_index(names="unit")
        gnugroup_new["grid"] = "fuel"
        gnugroup_new["node"] = "biomass"
        gnugroup = pd.concat(
            [gnugroup.loc[~gnugroup['unit'].str.lower().str.contains(techBBName.lower())], gnugroup_new])

    # add units to unitUnittype
    # unitUnittypeNew = newUnits.copy()
    # unitUnittypeNew['unittype'] = newUnits['unit'].str.split(' ', expand=True)[
    #     2]
    # unitUnittype = pd.concat(
    #     [unitUnittype.loc[~unitUnittype['unit'].str.contains(techBBName.lower())], unitUnittypeNew])

    # add new units to sheet effLevelGroupUnit
    effLevelNewUnitsComp = pd.DataFrame()
    for j in [1, 2, 3]:
        effLevelNewUnits = newUnits.copy()
        effLevelNewUnits['effLevel'] = f'level{j}'
        effLevelNewUnits['effSelector'] = 'directOff'
        effLevelNewUnitsComp = pd.concat(
            [effLevelNewUnitsComp, effLevelNewUnits])

    effLevelGroupUnit = pd.concat([effLevelGroupUnit.loc[~effLevelGroupUnit['unit'].str.lower().str.contains(techBBName.lower())],
                                   effLevelNewUnitsComp], ignore_index=True)

    # add new units to sheet utAvailability
    utAvailabilityLimitsNewUnits = newUnits.copy()
    utAvailabilityLimitsNewUnits['timesteps'] = 't000001'
    utAvailabilityLimitsNewUnits['becomeAvailable'] = 1
    utAvailabilityLimits = pd.concat([utAvailabilityLimits.loc[~utAvailabilityLimits['unit'].str.lower().str.contains(techBBName.lower())],
                                      utAvailabilityLimitsNewUnits], ignore_index=True)

    # add units to sheet p_gnu_io
    p_gnu_io.set_index('unit', inplace=True)
    p_gnu_io_adds = pd.DataFrame()
    for cooling_type in ['once-through', 'closed-loop', 'dry-cooling']:
        newUnitsCool = newUnits.copy(
        ).loc[newUnits['unit'].str.contains(cooling_type)]
        # continue if there is no unit with this cooling type
        if newUnitsCool.empty:
            continue
        newUnitsCool.index = newUnitsCool['unit'].str.rsplit(' ', n=1, expand=True)[
            0]
        p_gnu_io_add = p_gnu_io.loc[newUnitsCool.index]
        # add fom/ivestment costs for different cooling types
        fomRate = p_gnu_io_add['fomCosts']/p_gnu_io_add['invCosts']
        p_gnu_io_add['invCosts'] = p_gnu_io_add['invCosts'] + \
            costChange.loc[cooling_type, 'investment']
        p_gnu_io_add['fomCosts'] = p_gnu_io_add['fomCosts'] + \
            costChange.loc[cooling_type, 'investment']*fomRate
        # change exisiting capacities based on shares reported in JRC
        sharesCT = shares.loc[shares['cooling_type'] == cooling_type]
        p_gnu_io_add['country'] = p_gnu_io_add['node'].str[0:2]
        p_gnu_io_add = p_gnu_io_add.reset_index(
            names='unit').set_index('country')
        p_gnu_io_add = p_gnu_io_add.combine_first(sharesCT).fillna(0)
        p_gnu_io_add['capacity'] = p_gnu_io_add['capacity'].astype(object).replace(
            {'eps': 0})*p_gnu_io_add['share']
        p_gnu_io_add = p_gnu_io_add.reset_index().set_index('unit')

        p_gnu_io_add = p_gnu_io_add.combine_first(newUnitsCool)[['grid', 'node', 'unit', 'input output',
                                                                 'conversionCoeff', 'capacity',
                                                                'unitSize', 'invCosts', 'fomCosts', 'vomCosts',
                                                                 'annuityFactor',
                                                                 'upperLimitCapacityRatio']].reset_index(drop=True).dropna().astype(object)

        # set 0 capacities to eps to avoid backbone problems
        p_gnu_io_add.loc[(p_gnu_io_add['input output'] == 'output') & (
            p_gnu_io_add['capacity'] == 0), 'capacity'] = 'eps'

        p_gnu_io_adds = pd.concat(
            [p_gnu_io_adds, p_gnu_io_add], ignore_index=True)
    p_gnu_io = pd.concat([p_gnu_io.reset_index().loc[~p_gnu_io.reset_index()['unit'].str.lower().str.contains(techBBName.lower())],
                          p_gnu_io_adds], ignore_index=True)

    # add new units to sheet p_unit
    p_unit_adds = pd.DataFrame()
    p_unit.set_index('unit', inplace=True)
    for cooling_type in ['once-through', 'closed-loop', 'dry-cooling']:
        newUnitsCool = newUnits.copy(
        ).loc[newUnits['unit'].str.contains(cooling_type)]
        # continue if there is no unit with this cooling type
        if newUnitsCool.empty:
            continue
        newUnitsCool.index = newUnitsCool['unit'].str.rsplit(' ', n=1, expand=True)[
            0]
        p_unit_add = p_unit.loc[newUnitsCool.index]
        # change efficiency
        p_unit_add['eff00'] = p_unit_add['eff00'] * \
            costChange.loc[cooling_type, 'efficiency']
        p_unit_add = p_unit_add.combine_first(newUnitsCool)[['unit', 'eff00', 'availability',
                                                            'maxUnitCount']].reset_index(drop=True)
        # don't use time series for unavailable units (save for later)
        unavailableUnits = p_unit_add['availability'] == 0

        # climate effects can be neglected for dry cooling (no availability time series)
        if not cooling_type == 'dry-cooling':
            p_unit_add['availability'] = ''
            p_unit_add['useTimeseriesAvailability'] = 1

        # don't use time series for unavailable units (execute with saved units)
        p_unit_add.loc[unavailableUnits, 'availability'] = 0
        p_unit_add.loc[unavailableUnits, 'useTimeseriesAvailability'] = 0

        p_unit_adds = pd.concat([p_unit_adds, p_unit_add], ignore_index=True)
    p_unit = pd.concat([p_unit.reset_index().loc[~p_unit.reset_index()['unit'].str.lower().str.contains(techBBName.lower())], p_unit_adds],
                       ignore_index=True).fillna(0)

    # read in time series for power plants
    for j, cooling_type in enumerate(['once-through', 'closed-loop']):
        ts_unit_add = pd.DataFrame()
        avai_tpp = pd.read_csv(
            avai_tpp_file[2*i+j], delimiter=';', index_col=0)
        newUnitsCool = newUnits.copy(
        ).loc[newUnits['unit'].str.contains(cooling_type)]
        ts_unit_add['unit'] = newUnitsCool
        ts_unit_add['param_unit'] = 'availability'
        ts_unit_add['forecast index'] = 'f00'
        ts_unit_add['node'] = newUnitsCool['unit'].str[0:5]
        ts_unit_add.set_index('node', inplace=True)
        ts_unit_add = ts_unit_add.combine_first(avai_tpp)
        # resort the dataframe
        ts_unit_add = ts_unit_add[['unit', 'param_unit', 'forecast index']
                                  + ts_unit_add.columns.difference(['unit', 'param_unit', 'forecast index']).to_list()]
        ts_unit = pd.concat([ts_unit, ts_unit_add])

# sort p_gnu_io
p_gnu_io = p_gnu_io[['grid', 'node', 'unit', 'input output', 'conversionCoeff', 'capacity', 'unitSize', 'invCosts', 'fomCosts',
                    'vomCosts', 'annuityFactor', 'upperLimitCapacityRatio']].fillna(0)

# create ts_unit if no cooling type is considered to avoid error when writing to excel
if len(config['backbone_tpps']['types']) == 0:
    ts_unit = pd.DataFrame()

ror_cf = hydro_inflow

flow = pd.DataFrame(
    [['solar'], ['offwind'], ['onwind'], ['ror']], columns=['flow'])

flowUnit = pd.DataFrame([], index=['flow', 'unit'])

timesteps = [f't{i:06d}' for i in range(1, 8761)]
ts_cf = pd.DataFrame([], index=['flow', 'node', 'forecast index']+timesteps)

nodes = hydro_inflow.index
capacity_hydro = pd.DataFrame(
    [], columns=['hydro', 'ror', 'hydro factor', 'ror factor'], index=nodes).fillna(0.0)

for i, unit in zip(p_gnu_io.index, p_gnu_io['unit']):
    if 'AL' in unit:  # excludes Albania from hydro calculation, because there is no data for regression - to be fixed?
        continue
    if 'ror' in unit:
        node = unit[0:5]
        if type(p_gnu_io['capacity'][i]) == float:
            capacity_hydro.loc[node, 'ror'] += p_gnu_io['capacity'][i]
    elif 'hydro' in unit:
        node = unit[0:5]
        if type(p_gnu_io['capacity'][i]) == float:
            capacity_hydro.loc[node, 'hydro'] += p_gnu_io['capacity'][i]

capacity_hydro.fillna(0, inplace=True)

capacity_hydro['hydro factor'] = capacity_hydro['hydro'] / \
    (capacity_hydro['hydro']+capacity_hydro['ror'])
capacity_hydro['ror factor'] = capacity_hydro['ror'] / \
    (capacity_hydro['hydro']+capacity_hydro['ror'])

ror_cf = (hydro_inflow).mul(capacity_hydro['ror factor'], axis=0).div(
    capacity_hydro['ror'], axis=0).dropna()
ror_cf[ror_cf > 1] = 1
ts_influx = (hydro_inflow).mul(
    capacity_hydro['hydro factor'], axis=0).fillna(0)
ts_influx.index = [i+' hydro' for i in ts_influx.index]
ts_influx['sum'] = ts_influx.sum(axis=1)
ts_influx = ts_influx.loc[ts_influx['sum'] != 0].drop('sum', axis=1)
ts_influx['grid'] = 'storage'

if not 'heat' in snakemake.output[0]:
    # add demand to ts_influx
    demand_file = snakemake.input.demand_file
    demand = pd.read_csv(demand_file, delimiter=';', index_col=0)
    demand = -demand  # demand must be negative in backbone
    demand['grid'] = 'elec'
else:  # do not do demand regression if temperature series are used for house demand
    demand = pd.read_excel(backbone_input_file, sheet_name='ts_influx')
    demand = demand.loc[demand['grid'] == 'elec'].set_index('node')

ts_influx = pd.concat([demand, ts_influx])
ts_influx['forecast index'] = 'f00'
ts_influx = ts_influx.reset_index().rename(columns={'index': 'node'})
ts_influx = ts_influx[['grid', 'node', 'forecast index']+timesteps]

# add inflow to ts_node as maxSpill
ts_node = ts_influx.copy().loc[ts_influx['grid'] == 'storage']
ts_node['param_gnBoundaryTypes'] = 'maxSpill'
ts_node = ts_node[['grid', 'node', 'param_gnBoundaryTypes', 'forecast index']
                  + ts_node.columns.difference(['grid', 'node', 'param_gnBoundaryTypes', 'forecast index']).to_list()]

if 'heat' in snakemake.output[0]:
    temperature = pd.read_csv(
        snakemake.input.temperature_file, delimiter=';', index_col=0)
    temperature.index = temperature.index + ' heat outdoor'
    temperature['grid'] = 'heat'
    temperature['param_gnBoundaryTypes'] = 'reference'
    temperature['forecast'] = 'f00'
    temperature = temperature.reset_index().rename({'index': 'node'}, axis=1)
    ts_node = pd.concat([ts_node, temperature])

# add flows for renewable units to flow_unit sheet
for unit in unit_df['unit']:
    if 'solar' in unit:
        new_row = pd.Series(data={'flow': 'solar', 'unit': unit}).T
        flowUnit = pd.concat([flowUnit, new_row], axis=1, ignore_index=True)
    elif 'onwind' in unit:
        new_row = pd.Series(data={'flow': 'onwind', 'unit': unit}).T
        flowUnit = pd.concat([flowUnit, new_row], axis=1, ignore_index=True)
    elif 'offwind' in unit:
        new_row = pd.Series(data={'flow': 'offwind', 'unit': unit}).T
        flowUnit = pd.concat([flowUnit, new_row], axis=1, ignore_index=True)
    elif 'ror' in unit:
        new_row = pd.Series(data={'flow': 'ror', 'unit': unit}).T
        flowUnit = pd.concat([flowUnit, new_row], axis=1, ignore_index=True)

flowUnit = flowUnit.T

# add cf to ts_cf
for node in solar_cf.index:
    new_row = pd.Series(
        data={'flow': 'solar', 'node': node, 'forecast index': 'f00'})
    new_row = pd.concat([new_row, solar_cf.loc[node]])
    ts_cf = pd.concat([ts_cf, new_row], axis=1, ignore_index=True)

for node in onwind_cf.index:
    new_row = pd.Series(
        data={'flow': 'onwind', 'node': node, 'forecast index': 'f00'})
    new_row = pd.concat([new_row, onwind_cf.loc[node]])
    ts_cf = pd.concat([ts_cf, new_row], axis=1, ignore_index=True)

for node in offwind_cf.index:
    new_row = pd.Series(
        data={'flow': 'offwind', 'node': node, 'forecast index': 'f00'})
    new_row = pd.concat([new_row, offwind_cf.loc[node]])
    ts_cf = pd.concat([ts_cf, new_row], axis=1, ignore_index=True)

for node in ror_cf.index:
    new_row = pd.Series(
        data={'flow': 'ror', 'node': node, 'forecast index': 'f00'})
    new_row = pd.concat([new_row, ror_cf.loc[node]])
    ts_cf = pd.concat([ts_cf, new_row], axis=1, ignore_index=True)

ts_cf = ts_cf.T

changed_sheets = ['unit', 'effLevelGroupUnit', 'utAvailabilityLimits', 'p_gnu_io', 'p_unit', 'ts_influx', 'flow',
                  'flowUnit', 'ts_cf', 'ts_unit', 'ts_node', 'gnugroup', 'unitUnittype']

backbone_excel = pd.ExcelFile(backbone_input_file)
sheet_names = backbone_excel.sheet_names

with pd.ExcelWriter(snakemake.output[0]) as writer:

    # read exisiting sheets and write them to the new file
    for sheet_name in sheet_names:
        if not sheet_name in changed_sheets:
            pd.read_excel(backbone_input_file, sheet_name=sheet_name).to_excel(
                excel_writer = writer, sheet_name = '%s' % (sheet_name), index=False)

    # adding in the new worksheet
    unit_df.to_excel(excel_writer = writer, sheet_name = 'unit', index=False)
    pd.DataFrame().to_excel(excel_writer = writer, sheet_name = 'unitUnittype', index=False)
    effLevelGroupUnit.to_excel(excel_writer = writer, sheet_name = 'effLevelGroupUnit', index=False)
    utAvailabilityLimits.to_excel(excel_writer = writer, sheet_name = 'utAvailabilityLimits', index=False)
    p_gnu_io.to_excel(excel_writer = writer, sheet_name = 'p_gnu_io', index=False)
    p_unit.to_excel(excel_writer = writer, sheet_name = 'p_unit', index=False)
    ts_influx.to_excel(excel_writer = writer, sheet_name = 'ts_influx', index=False)
    flow.to_excel(excel_writer = writer, sheet_name = 'flow', index=False)
    flowUnit.to_excel(excel_writer = writer, sheet_name = 'flowUnit', index=False)
    ts_unit.to_excel(excel_writer = writer, sheet_name = 'ts_unit', index=False)
    ts_cf.to_excel(excel_writer = writer, sheet_name = 'ts_cf', index=False)
    ts_node.to_excel(excel_writer = writer, sheet_name = 'ts_node', index=False)
    gnugroup.to_excel(excel_writer = writer, sheet_name = 'gnugroup', index=False)
