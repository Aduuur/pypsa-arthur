---
title: 'cd2es: Converting climate data to energy system input data'
tags:
  - Python
  - snakemake
  - climate data
  - energy system analysis
authors:
  - name: Leonie Sara Plaga
    orcid: 0000-0001-6339-6100
    corresponding: True
    affiliation: 1
  - name: Valentin Bertsch
    orcid: 0000-0001-9477-4805
    affiliation: 1
affiliations:
 - name: Ruhr-Universität Bochum, Universitätsstr. 150, 44801 Bochum, Germany
   index: 1
date: 10 July 2024
bibliography: paper.bib
---

# Summary

The tool *cd2es* converts climate projections from CORDEX [@WCRP.2024] to energy system model input data.
The tool is written in Python and uses snakemake for workflow management.
*cd2es* can calculate the impact of climate change on input data time series for energy system models (e.g. capacity factors).
In contrast to existing tools, *cd2es* can use a variety of different climate models instead of historic weather data.
*cd2es* can automatically download CORDEX model outputs.
Optionally, a bias-adaption with ERA5-reanalysis data can be performed.
The outputs are time series for renewable capacity factors, demand and the availability of thermal power plants aggregated to user specific geometries with an hourly resolution for easy implementation into different energy system models.

# Statement of need
Global Circulation Models (GCMs) modelling the properties of the atmosphere and oceans project future climate developments [@Jacob.2014]. 
Those GCMs project a significant change in climate variables such as temperature and precipitation under climate change [@Dosio.2018]. 
As many components of the energy system depend on climate variables, climate change should be considered when planning future energy systems. 
It is therefore important to convert climate projections into input data for energy system models, which are commonly used for energy systems planning [@DeCarolis.2017; @Plaga.2023].

There is only a limited number of tools available which convert climate variables into energy system model inputs.
renewables.ninja calculates solar and wind capacity factors from historical reanalysis data [@Staffell.2016; @Pfenninger.2016], but there is no open-source code available for the conversion, furthermore, the tool is limited to solar and wind capacity factors and historic data.
The Python library pvlib [@Anderson.2023] allows for a detailed calculation of solar capacity factors, yet is limited to historic data and solar capacity factors.
Pypsa/atlite [@Hofmann.2021] converts historic reanalysis data to energy system input data. Yet, it cannot account for climate projections and does not calculate climate influences on thermal power plants.
@Formayer.2023 provide a data set for temperature, radiation, wind power and hydro power based on future climate projections. However, they only provide results for one climate model and the continent Europe and no ready to use code to enlarge the findings to other climate models or other continents.
In summary, there is a lack of open source tools to include climate projections into energy system planning in a comprehensive matter.

*cd2es* provides a tool for automatically downloading climate projections and the converting them to energy system input data. 
It supports the calculation of wind and solar photovoltaic capacity factors, concentrated solar power, availability of thermal power plants, hydro power and electricity demand based on CORDEX climate data [@WCRP.2024].
The tool can process a variety of different climate models hosted on CORDEX for a wide geographical scope.
It allows for an optional bias correction of the climate data based on ERA5-reanalysis data [@MunozSabater.2019].
The output of *cd2es* are csv files with time series in hourly resolution for a user-chosen geography.
Therefore, they can be easily included in different energy system optimization models.
The *cd2es* tool was used in @Plaga.2022 and is currently in use in two research projects, StEAM and REWARDS [@Rewards.2024].
*cd2es* is aimed at energy system modelers who want to include climate data into their energy system models.

# Software dependencies

The software is written in Python and uses the workflow management tool snakemake [@Molder.2021].
For processing climate data, the open-source tool *cdo* [@Schulzweida.2020] is used.
On Windows, the open-source tool *wsl*  is necessary to run *cdo* on Windows [@microsoft.2024].

# Methods for converting climate variables to energy system input data

Most conversion methods are based on @Plaga.2023 and described in detail in the documentation of the tool *cd2es*.
However, we will also include a short overview here.

## Bias adaption

The climate data is bias adapted using a quantile delta mapping approach [@Cannon.2015].

## Wind power

The wind speed $v$ is first interpolated from the height reported in the data to turbine height.
Then capacity factor $cf_\text{wind, single}(v)$ can be derived via a standardized production curve:
$$
    cf_\text{wind, single}(v) = 
    \begin{cases}
    0, & v < v_{\text{in}}, \\
    \frac{v^3- v_{\text{in}}^3}{v_{\text{r}}^3- v_{\text{in}}^3}, & v_{\text{in}} \leq v <  v_{\text{r}}, \\
    1, &  v_{\text{r}} \leq v <  v_{\text{out}}, \\
    0, & v \geq v_{\text{out}},\\
    \end{cases}
$$
with cut-in velocity $v_{\text{in}}$, rated velocity $v_{\text{r}}$ and cut-out velocity $v_{\text{out}}$ [@vanderWiel.2019].
We smooth the production curve with a gaussian filter to account for multiple turbines [@Staffell.2016].

## Solar photovoltaics

Photovoltaic cells are influenced by climate variables in two ways: 
the solar irradiance influences the available incoming energy, while the temperature influences the cell's efficiency.
The *cd2es* tools supports three different models for calculating photovoltaic time series, see @Jerez.2015.

## Availability of thermal power plants

As thermal power plants need cooling, their availability decreases with rising temperatures.
*cd2es* distinguishes between once-through cooled plants and closed-loop cooled plants.
For once-through plants, not only the temperature but also available water is considered.
The availability of the plants follows piecewise linear equations, which were implemented as described in @Abdin.2019.

## Hydropower

To calculate hydro power output, historic runoff at the hydro power plants' locations are evaluated.
It is assumed, that power plants reach their installed capacity $P_0$ at the average historic runoff $\bar{Q}_{\text{hist}}$ at their location (optionally multiplied by factor $a$).
Then, future hydro power of one power plant $P(t)$ can be calculated with 
$$
    P(t) = Q(t) \cdot \frac{P_0}{a \cdot \bar{Q}_{\text{hist}}}
$$
using the linear relation between runoff and hydro power production and the future runoff $Q(t)$ [@Schlott.2018].

## Demand

To calculate future demand, a quadratic regression is performed between historic temperatures and historic demand data as in @Zhang.2020.
The parameters derived here are then used to scale future demand time series.

# Acknowledgements

The tool was developed in the projects StEAM (funding code 03EI1043A) and REWARDS (funding code 03EI1083D), both funded by the German Federal Ministry for Economic Affairs and Climate Action.

# References