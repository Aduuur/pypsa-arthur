"""
Script for performing linear regression of heating and cooling electricity demand
against HDD/CDD per country, predicting values for a fixed year, computing scaling ratios,
and plotting historical data, regression lines, and predicted values.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import logging
from sklearn.linear_model import LinearRegression

from scripts._helpers import configure_logging, set_scenario_config


def load_and_reindex_demand(thermal_demand_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load daily heating or cooling demand and reindex to hourly.

    Parameters
    ----------
    thermal_demand_path : str
        Path to CSV file containing daily thermal demand.

    Returns
    -------
    data_daily : pd.DataFrame
        Daily demand indexed by date.
    data_hourly : pd.DataFrame
        Hourly demand, forward filled.
    """
    data_daily = pd.read_csv(thermal_demand_path, index_col=0, parse_dates=True)
    data_daily.index.name = "time"

    leap_days = data_daily.index[(data_daily.index.month == 2) & (data_daily.index.day == 29)]
    if len(leap_days) > 0:
        logging.warning(f"Leap days present: {leap_days}")
    else:
        logging.info("No leap days present.")

    hourly_index = pd.date_range(
        start=data_daily.index.min(),
        end=data_daily.index.max() + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
        freq="h"
    )
    data_hourly = data_daily.reindex(hourly_index).ffill()
    data_hourly.index.name = "time"
    return data_daily, data_hourly


def perform_regression(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Perform linear regression per country.

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex DataFrame with levels (country, year).
    x_col : str
        Name of predictor column (e.g., 'sum_hdd').
    y_col : str
        Name of target column (e.g., 'sum_elec_space_heat').

    Returns
    -------
    pd.DataFrame
        Regression results per country with columns:
        ['country', 'coef', 'intercept', 'r2', 'x_col', 'y_col']
    """
    results = []
    for country, df_country in df.groupby(level="country"):
        X = df_country[[x_col]].values
        y = df_country[y_col].values
        if len(df_country) > 1:
            model = LinearRegression().fit(X, y)
            results.append({
                "country": country,
                "coef": model.coef_[0],
                "intercept": model.intercept_,
                "r2": model.score(X, y),
                "x_col": x_col,
                "y_col": y_col
            })
    return pd.DataFrame(results)


def predict_and_scale(df: pd.DataFrame, results_df: pd.DataFrame, annual_demand: pd.Series,
                      year: int, x_col: str, y_col: str,
                      max_scale) -> pd.DataFrame:
    """
    Predict values for a fixed year using regression and compute scaling ratios,
    limited to [min_scale, max_scale].

    Parameters
    ----------
    df : pd.DataFrame
        Historical multiindex DataFrame (country, year).
    results_df : pd.DataFrame
        Regression results from perform_regression.
    annual_demand : pd.Series
        Annual sum of HDD/CDD indexed by country.
    year : int
        Year for which to compute scaling ratios.
    x_col : str
        Name of predictor column.
    y_col : str
        Name of target column.
    max_scale : float, optional
        Maximum allowed scaling ratio, by default 2.0.

    Returns
    -------
    pd.DataFrame
        Columns: ['country', 'scaling_ratio', 'predicted', f'y_actual_{year}']
    """
    results = []
    for _, row in results_df.iterrows():
        country = row["country"]
        coef = row["coef"]
        intercept = row["intercept"]

        if country not in annual_demand.index:
            continue
        x_val = annual_demand[country]
        y_pred = coef * x_val + intercept
        
        y_actual = df.loc[(country, year), y_col]
        scaling_ratio = y_pred / y_actual if y_actual != 0 else np.nan
        
        # Limit scaling ratio
        scaling_ratio = np.clip(scaling_ratio, 1/max_scale, max_scale)

        results.append({
            "country": country,
            "scaling_ratio": scaling_ratio,
            "predicted": y_pred,
            f"y_actual_{year}": y_actual
        })
    return pd.DataFrame(results)



def merge_results(results_heat_df: pd.DataFrame, results_cool_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge heat and cool regression results.

    Parameters
    ----------
    results_heat_df : pd.DataFrame
        Regression results for heating.
    results_cool_df : pd.DataFrame
        Regression results for cooling.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame of heat and cool results.
    """
    merged = results_heat_df.merge(
        results_cool_df,
        on="country",
        suffixes=("_heat", "_cool")
    )
    return merged


def plot_country_with_predictions(country: str, df: pd.DataFrame, results_heat_df: pd.DataFrame,
                                  results_cool_df: pd.DataFrame, annual_hdd: pd.Series,
                                  annual_cdd: pd.Series, year: int,
                                  save_path: str = None, max_scale: float=None) -> None:
    """
    Plot heating and cooling demand with regression line and predicted value for a country.

    Parameters
    ----------
    country : str
        Country code to plot.
    df : pd.DataFrame
        Historical multiindex DataFrame (country, year).
    results_heat_df : pd.DataFrame
        Regression results for heating.
    results_cool_df : pd.DataFrame
        Regression results for cooling.
    annual_hdd : pd.Series
        Annual HDD sum per country.
    annual_cdd : pd.Series
        Annual CDD sum per country.
    year : int
        Year for prediction.
    save_path : str, optional
        Path to save the plot image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_country = df.xs(country, level="country")

    # -------- Heat --------
    res = results_heat_df[results_heat_df["country"] == country]
    if not res.empty:
        coef = res["coef"].iloc[0]
        intercept = res["intercept"].iloc[0]
        r2 = res["r2"].iloc[0]

        X = df_country["sum_hdd"].values
        y = df_country["sum_elec_space_heat"].values
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred_line = coef * x_range + intercept

        axes[0].scatter(X, y, label="Historical data")
        axes[0].plot(x_range, y_pred_line, color="red", label="Regression")

        if country in annual_hdd.index:
            x_new = annual_hdd[country]
            y_new = coef * x_new + intercept
            try:
                y_actual = df.loc[(country, year), "sum_elec_space_heat"]
                scaling_ratio = y_new / y_actual if y_actual != 0 else np.nan
            except KeyError:
                y_actual, scaling_ratio = np.nan, np.nan
            
            scaling_ratio = np.clip(scaling_ratio, 1/max_scale, max_scale)

            axes[0].scatter([x_new], [y_new], color="green", s=100, marker="x", label=f"Predicted {year}")
            axes[0].annotate(f"Scaling={scaling_ratio:.2f}", (x_new, y_new),
                             textcoords="offset points", xytext=(5, -10), color="green")

        axes[0].set_title(f"{country} – Heating (R²={r2:.2f})")
        axes[0].set_xlabel("sum_hdd")
        axes[0].set_ylabel("sum_elec_space_heat")
        axes[0].legend()

    # -------- Cool --------
    res = results_cool_df[results_cool_df["country"] == country]
    if not res.empty:
        coef = res["coef"].iloc[0]
        intercept = res["intercept"].iloc[0]
        r2 = res["r2"].iloc[0]

        X = df_country["sum_cdd"].values
        y = df_country["sum_elec_space_cool"].values
        x_range = np.linspace(X.min(), X.max(), 100)
        y_pred_line = coef * x_range + intercept

        axes[1].scatter(X, y, label="Historical data")
        axes[1].plot(x_range, y_pred_line, color="red", label="Regression")

        if country in annual_cdd.index:
            x_new = annual_cdd[country]
            y_new = coef * x_new + intercept
            try:
                y_actual = df.loc[(country, year), "sum_elec_space_cool"]
                scaling_ratio = y_new / y_actual if y_actual != 0 else np.nan
            except KeyError:
                y_actual, scaling_ratio = np.nan, np.nan
            axes[1].scatter([x_new], [y_new], color="green", s=100, marker="x", label=f"Predicted {year}")
            axes[1].annotate(f"Scaling={scaling_ratio:.2f}", (x_new, y_new),
                             textcoords="offset points", xytext=(5, -10), color="green")

        axes[1].set_title(f"{country} – Cooling (R²={r2:.2f})")
        axes[1].set_xlabel("sum_cdd")
        axes[1].set_ylabel("sum_elec_space_cool")
        axes[1].legend()

    plt.tight_layout()
    if save_path:
        # Ensure folder exists
        folder = os.path.dirname(save_path)
        os.makedirs(folder, exist_ok=True)

        # Save figure without displaying
        plt.savefig(save_path, dpi=150)

    plt.close(fig)  # Close figure to free memory


# --- Main execution --- #
if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake
        snakemake = mock_snakemake("build_electric_thermal_demand_regression")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    year_et = int(snakemake.params.energy_totals_year)
    max_scale = float(snakemake.params.max_scale)
    scaling_method = snakemake.params.scaling_method
    
    # Load historical demand
    df = pd.read_csv(
        snakemake.input.hist_demand_day_calc,
        index_col=['country','year']
    )

    # Load HDD/CDD
    hdd, _ = load_and_reindex_demand(snakemake.input.hdd)
    cdd, _ = load_and_reindex_demand(snakemake.input.cdd)
    
    annual_hdd = hdd.sum(axis=0)
    annual_cdd = cdd.sum(axis=0)

    #otherwise snakemake misses output
    output_folder = snakemake.output.elec_heat_reg_plot  # target folder
    os.makedirs(output_folder, exist_ok=True)  # create folder if it does not exist


    if scaling_method == 'singleYear':
        logger.info(f'Scale heating/cooling demand with method >>singleYear<< using year {year_et}')
        ratio_heat = annual_hdd / df['sum_hdd'].xs(year_et, level='year')
        ratio_heat = np.clip(ratio_heat, 1/max_scale, max_scale)
        ratio_heat.name='scaling_ratio_heat'
        
        y_actual_heat=df['sum_elec_space_heat'].xs(year_et, level='year')
        y_actual_heat.name=f'y_actual_{year_et}_heat'
        
        scaled_heat=y_actual_heat*ratio_heat
        scaled_heat.name='scaled_heat'
        
        ratio_cool = annual_cdd / df['sum_cdd'].xs(year_et, level='year')
        ratio_col = np.clip(ratio_cool, 1/max_scale, max_scale)
        ratio_cool.name='scaling_ratio_cool'
        
        y_actual_cool=df['sum_elec_space_cool'].xs(year_et, level='year')
        y_actual_cool.name=f'y_actual_{year_et}_cool'
        
        scaled_cool=y_actual_cool*ratio_cool
        scaled_cool.name='scaled_cool'
        
        scaling_all=pd.concat([ratio_heat,scaled_heat,y_actual_heat,ratio_cool,scaled_cool,y_actual_cool],axis=1)

    elif scaling_method== 'regression':
        reg_min_year=int(min(df.index.get_level_values("year")))
        reg_max_year=int(max(df.index.get_level_values("year")))

        logger.info(f'Scale heating/cooling demand with method >>regression<< using years from {reg_min_year} to {reg_max_year}')
        # Regressions
        results_heat_df = perform_regression(df, "sum_hdd", "sum_elec_space_heat")
        results_cool_df = perform_regression(df, "sum_cdd", "sum_elec_space_cool")
        
        # Compute scaling ratios
        scaling_heat_df = predict_and_scale(df, results_heat_df, annual_hdd, year_et,
                                            "sum_hdd", "sum_elec_space_heat", max_scale)
        scaling_cool_df = predict_and_scale(df, results_cool_df, annual_cdd, year_et,
                                            "sum_cdd", "sum_elec_space_cool",max_scale)

        scaling_all = scaling_heat_df.merge(scaling_cool_df, on="country", suffixes=("_heat", "_cool"))
        



        for country in df.index.get_level_values("country").unique():
            save_path = os.path.join(output_folder, f"{country}_heat_cool_{year_et}.png")
            plot_country_with_predictions(
                country, df, results_heat_df, results_cool_df,
                annual_hdd, annual_cdd, year=year_et,
                save_path=save_path, max_scale = max_scale
            )
    elif 'none':
        logger.info(f'No scaling of heating/cooling demand!')

        ratio_heat = annual_hdd
        ratio_heat[:]=1
        ratio_heat.name='scaling_ratio_heat'
        
        y_actual_heat=df['sum_elec_space_heat'].xs(year_et, level='year')
        y_actual_heat.name=f'y_actual_{year_et}_heat'
        
        scaled_heat=y_actual_heat*ratio_heat
        scaled_heat.name='scaled_heat'
        
        ratio_cool = annual_cdd
        ratio_cool[:]=1
        ratio_cool.name='scaling_ratio_cool'
        
        y_actual_cool=df['sum_elec_space_cool'].xs(year_et, level='year')
        y_actual_cool.name=f'y_actual_{year_et}_cool'
        
        scaled_cool=y_actual_cool*ratio_cool
        scaled_cool.name='scaled_cool'
        
        scaling_all=pd.concat([ratio_heat,scaled_heat,y_actual_heat,ratio_cool,scaled_cool,y_actual_cool],axis=1)
    else:
        raise ValueError('No proper scaling method (singleYear, regression, none) given!')
    
    scaling_all.index.name='country'
    scaling_all.to_csv(snakemake.output.et_scale)