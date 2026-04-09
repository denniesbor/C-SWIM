"""
Author: Dennies Bor, Ed Oughton, Lucy, Bob
Role: GIC prediction using alpha-beta models with bootstrap uncertainty estimation.
      Predicts GIC at each substation using trained regression models and
      geomagnetic coordinate transformations.
Inputs:
    - data/admittance_matrix/substation_info.csv
    - SWERVE ranked_models.pkl
    - SWERVE config
Outputs:
    - data/regression/substations_with_gic_uncertainty.geojson
"""

import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.io import loadmat
from scipy.interpolate import LinearNDInterpolator
from spacepy.time import Ticktock
import spacepy.coordinates as coord
from multiprocessing import Pool
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path("/data/archives/nfs/SWERVE/")))

from swerve import config as swerve_config
from configs import DATA_DIR, SWERVE_DIR, setup_logger

logger = setup_logger(log_file="logs/val_alpha_beta_reg.log")

OUTPUT_DIR = DATA_DIR / "regression"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class GICPredictor:
    """Predicts GIC values at substations using alpha-beta regression models."""

    def __init__(self, config):
        self.config = config
        self.models = None

    def load_models(self, models_file):
        """Load trained regression models from pickle file."""
        logger.info(f"Loading models from {models_file}")
        with open(models_file, "rb") as f:
            self.models = pickle.load(f)
        logger.info(f"Loaded {len(self.models)} regression models")
        return self.models

    def get_beta_factor(self, site_name, lat, lon):
        """Calculate beta factor using spatial interpolation from SWERVE data."""
        data = loadmat(self.config["files"]["beta"])
        waveform = data["waveform"][0]
        ott_data = waveform[1][0][0]
        betas = ott_data[1][0]

        rows = []
        for i in range(len(betas)):
            beta = betas[i][0][0][1][0][0]
            beta_lat = betas[i][0][0][2][0][0]
            beta_lon = betas[i][0][0][3][0][0]
            rows.append([beta, beta_lat, beta_lon])

        df = pd.DataFrame(rows, columns=["beta", "lat", "lon"])
        interpolator = LinearNDInterpolator(df[["lat", "lon"]], df["beta"])
        beta_factor = interpolator(lat, lon)

        if np.isnan(beta_factor):
            raise ValueError(f"Beta factor is NaN for {site_name} at ({lat}, {lon})")
        return beta_factor

    def get_alpha_factor(self, site_name, lat, lon):
        """Calculate alpha factor using geomagnetic coordinate transformation."""
        Re = 6371.0
        storm_date = self.config["limits"]["data"][0].strftime("%Y-%m-%dT%H:%M:%S")
        c = coord.Coords([[(Re) / Re, lat, lon]], "GEO", "sph", ["Re", "deg", "deg"])
        c.ticks = Ticktock([storm_date], "UTC")
        c = c.convert("MAG", "sph")
        mag_lat = c.data[0][1]
        alpha = 0.001 * np.exp(0.115 * mag_lat)

        if np.isnan(alpha):
            raise ValueError(f"Alpha factor is NaN for {site_name} at ({lat}, {lon})")
        return alpha

    def _get_magnetic_latitude(self, lat, lon):
        """Convert geographic coordinates to magnetic latitude."""
        Re = 6371.0
        storm_date = self.config["limits"]["data"][0].strftime("%Y-%m-%dT%H:%M:%S")
        c = coord.Coords([[(Re) / Re, lat, lon]], "GEO", "sph", ["Re", "deg", "deg"])
        c.ticks = Ticktock([storm_date], "UTC")
        c = c.convert("MAG", "sph")
        return c.data[0][1]

    def predict_gic(self, site_name, lat, lon, model_rank=0):
        """Predict GIC for a single substation using the specified model."""
        if self.models is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        if model_rank >= len(self.models):
            raise ValueError(
                f"Model rank {model_rank} exceeds available models ({len(self.models)})"
            )

        model = self.models[model_rank]
        alpha = self.get_alpha_factor(site_name, lat, lon)
        beta = self.get_beta_factor(site_name, lat, lon)
        mag_lat = self._get_magnetic_latitude(lat, lon)

        features, feature_names = [], []
        for input_name in model["inputs"]:
            if input_name == "alpha":
                features.append(alpha)
                feature_names.append("alpha")
            elif input_name == "interpolated_beta":
                features.append(beta)
                feature_names.append("beta")
            elif input_name == "alpha*interpolated_beta":
                features.append(alpha * beta)
                feature_names.append("alpha*beta")
            elif input_name == "mag_lat":
                features.append(mag_lat)
                feature_names.append("mag_lat")
            elif input_name == "mag_lat*interpolated_beta":
                features.append(mag_lat * beta)
                feature_names.append("mag_lat*beta")

        features = np.array(features).reshape(1, -1)
        prediction = model["model"].predict(features)[0]

        return {
            "site_name": site_name,
            "coordinates": (lat, lon),
            "alpha": alpha,
            "beta": beta,
            "features": dict(zip(feature_names, features[0])),
            "predicted_gic": prediction,
            "model_info": {
                "inputs": model["inputs"],
                "equation": model["equation"],
                "cc": model["cc"],
                "rmse": model["rmse"],
            },
        }


def predict_for_substation(args):
    """Multiprocessing worker for substation GIC prediction."""
    idx, row, model_idx, predictor = args
    try:
        lon, lat = row.geometry.x, row.geometry.y
        site_name = f"Substation_{row['name']}"
        result = predictor.predict_gic(site_name, lat, lon, model_rank=model_idx)
        return idx, result["predicted_gic"], result["model_info"]["rmse"]
    except Exception as e:
        logger.error(f"Error predicting GIC for substation {row['name']}: {e}")
        return idx, np.nan, np.nan


def process_model_parallel(trafo_gdf, predictor, model_idx, n_processes=10):
    """Run GIC predictions for all substations using multiprocessing."""
    model_name = f"model_{model_idx + 1}"
    logger.info(f"Processing {model_name} with {n_processes} processes...")

    args = [(idx, row, model_idx, predictor) for idx, row in trafo_gdf.iterrows()]
    with Pool(n_processes) as pool:
        results = list(
            tqdm(
                pool.imap(predict_for_substation, args),
                total=len(args),
                desc=model_name,
            )
        )

    results.sort(key=lambda x: x[0])
    return model_name, [r[1] for r in results], [r[2] for r in results]


def bootstrap_uncertainty(predictions_dict, rmse_dict, n_bootstrap=500):
    """Estimate prediction uncertainty via bootstrap resampling across models."""
    n_substations = len(predictions_dict["model_1"])
    logger.info(
        f"Bootstrap uncertainty: {n_bootstrap} iterations, {n_substations} substations"
    )

    bootstrap_stats = []
    for sub_idx in tqdm(range(n_substations), desc="Bootstrap"):
        all_samples = []
        for model_name in predictions_dict:
            pred = predictions_dict[model_name][sub_idx]
            rmse = rmse_dict[model_name][sub_idx]
            if not (np.isnan(pred) or np.isnan(rmse)):
                all_samples.extend(pred + np.random.normal(0, rmse, n_bootstrap))

        if all_samples:
            s = np.array(all_samples)
            stats = {
                "mean_prediction": np.mean(s),
                "std_uncertainty": np.std(s),
                "conf_68_lower": np.percentile(s, 16.0),
                "conf_68_upper": np.percentile(s, 84.0),
            }
        else:
            stats = {
                k: np.nan
                for k in [
                    "mean_prediction",
                    "std_uncertainty",
                    "conf_68_lower",
                    "conf_68_upper",
                ]
            }

        bootstrap_stats.append(stats)

    logger.info("Bootstrap uncertainty analysis completed")
    return bootstrap_stats


def generate_substation_gic_analysis(
    substation_file,
    config,
    output_file,
    cc_threshold=0.75,
    n_processes=10,
    n_bootstrap=500,
    models_file=None,
):
    """Run full GIC prediction and uncertainty pipeline for all substations."""
    if models_file is None:
        raise ValueError("Please provide a valid models_file path")

    logger.info(f"Loading substation data from {substation_file}")
    trafo_df = pd.read_csv(substation_file)
    if "geometry" not in trafo_df.columns:
        trafo_gdf = gpd.GeoDataFrame(
            trafo_df,
            geometry=gpd.points_from_xy(trafo_df.longitude, trafo_df.latitude),
            crs="EPSG:4326",
        )
    logger.info(f"Loaded {len(trafo_gdf)} substations")

    predictor = GICPredictor(config)
    models = predictor.load_models(models_file=models_file)
    good_models = [i for i, m in enumerate(models) if m["cc"] > cc_threshold]
    logger.info(
        f"Using {len(good_models)} models with cc > {cc_threshold} (of {len(models)} total)"
    )

    predictions_dict, rmse_dict = {}, {}
    for i, model_idx in enumerate(good_models):
        model_name, predictions, rmse_values = process_model_parallel(
            trafo_gdf, predictor, model_idx, n_processes
        )
        trafo_gdf[model_name] = predictions
        predictions_dict[model_name] = predictions
        rmse_dict[model_name] = rmse_values

    bootstrap_stats = bootstrap_uncertainty(predictions_dict, rmse_dict, n_bootstrap)

    for i, stats in enumerate(bootstrap_stats):
        for key, val in stats.items():
            trafo_gdf.loc[i, key] = val

    trafo_gdf["conf_68_lower"] = trafo_gdf["conf_68_lower"].clip(lower=0)

    for col in [f"model_{i+1}" for i in range(len(good_models))]:
        logger.info(
            f"{col}: mean={trafo_gdf[col].mean():.1f} A, max={trafo_gdf[col].max():.1f} A"
        )
    logger.info(f"Mean uncertainty: {trafo_gdf['std_uncertainty'].mean():.1f} A")

    logger.info(f"Saving results to {output_file}")
    trafo_gdf.to_file(output_file, driver="GeoJSON")
    logger.info("GIC prediction analysis complete")
    return trafo_gdf


if __name__ == "__main__":
    CONFIG = swerve_config()
    results_dir = os.path.join(CONFIG["dirs"]["data"], "_results")
    models_file = os.path.join(results_dir, "ranked_models.pkl")

    result_gdf = generate_substation_gic_analysis(
        substation_file=DATA_DIR / "admittance_matrix" / "substation_info.csv",
        config=CONFIG,
        output_file=OUTPUT_DIR / "substations_with_gic_uncertainty.geojson",
        cc_threshold=0.75,
        n_processes=10,
        n_bootstrap=500,
        models_file=models_file,
    )
