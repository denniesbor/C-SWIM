"""Economic analysis module for GIC impact assessment"""

import copy
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from pyomo.environ import value
from pyomo.opt import SolverFactory

from configs import USE_ALPHA_BETA_SCENARIO, FIGURES_DIR, setup_logger
from viz import plot_econo_naics, plot_vuln_trafos, plot_socio_economic_impact
from l_prepr_data import (
    load_and_aggregate_tiles, load_gic_results, process_vulnerability_chunks, 
    find_vulnerable_substations, load_network_data
)
from models.io_model import InputOutputModel
from models.cge_data_model import SAMDataManager, run_cge_example

logger = setup_logger("econ impact assessment")


def load_and_process_data():
    """Load and process all necessary data for economic analysis"""
    aggregate_gdf = load_and_aggregate_tiles()
    
    combined_ds, combined_vuln, vuln_table = load_gic_results()
    
    mean_vuln_all = process_vulnerability_chunks(combined_vuln, chunk_size=50, max_realizations=2000)
    
    df_lines, df_substations = load_network_data()
    
    common_vulnerable, vulnerability_matrix, target_scenarios = find_vulnerable_substations(mean_vuln_all)
    
    return {
        'aggregate_gdf': aggregate_gdf,
        'combined_ds': combined_ds,
        'combined_vuln': combined_vuln,
        'mean_vuln_all': mean_vuln_all,
        'df_lines': df_lines,
        'df_substations': df_substations,
        'common_vulnerable': common_vulnerable,
        'vulnerability_matrix': vulnerability_matrix,
        'target_scenarios': target_scenarios
    }


def get_confidence_intervals(ds, alpha_beta_scenario=USE_ALPHA_BETA_SCENARIO):
    """Calculate confidence intervals for I-O modeling"""
    
    results = []
    
    if alpha_beta_scenario:
        return_periods = ['75yr', '100yr', '125yr', '150yr', '175yr', '200yr', '225yr', '250yr']
        
        for rp in return_periods:
            lower_scenario = f'gic_{rp}_conf_68_lower'
            mean_scenario = f'gic_{rp}_mean_prediction'  
            upper_scenario = f'gic_{rp}_conf_68_upper'
            
            scenario_group = f'gic_{rp}'
            
            lower_data = ds.sel(scenario=lower_scenario) if lower_scenario in ds.scenario.values else None
            mean_data = ds.sel(scenario=mean_scenario) if mean_scenario in ds.scenario.values else None
            upper_data = ds.sel(scenario=upper_scenario) if upper_scenario in ds.scenario.values else None
            
            if mean_data is not None:
                gdp_vars = [var for var in ds.data_vars if var.startswith('GDP_') and var.endswith('_affected')]
                for gdp_var in gdp_vars:
                    sector = gdp_var.replace('GDP_', '').replace('_affected', '')
                    
                    results.append({
                        'scenario': scenario_group,
                        'variable': f'GDP_{sector}',
                        'mean': float(mean_data[gdp_var].values),
                        'p5': float(lower_data[gdp_var].values) if lower_data is not None else float(mean_data[gdp_var].values),
                        'p95': float(upper_data[gdp_var].values) if upper_data is not None else float(mean_data[gdp_var].values)
                    })
                
                est_vars = [var for var in ds.data_vars if var.startswith('EST_') and var.endswith('_affected')]
                est_mean = sum(float(mean_data[var].values) for var in est_vars)
                est_lower = sum(float(lower_data[var].values) for var in est_vars) if lower_data is not None else est_mean
                est_upper = sum(float(upper_data[var].values) for var in est_vars) if upper_data is not None else est_mean
                
                results.append({
                    'scenario': scenario_group,
                    'variable': 'EST_TOTAL',
                    'mean': est_mean,
                    'p5': est_lower,
                    'p95': est_upper
                })
                
                results.append({
                    'scenario': scenario_group,
                    'variable': 'POP_AFFECTED',
                    'mean': float(mean_data['mean_pop_affected'].values),
                    'p5': float(lower_data['mean_pop_affected'].values) if lower_data is not None else float(mean_data['mean_pop_affected'].values),
                    'p95': float(upper_data['mean_pop_affected'].values) if upper_data is not None else float(mean_data['mean_pop_affected'].values)
                })
    
    else:
        for scenario in ds.scenario.values:
            scenario_data = ds.sel(scenario=scenario)
            
            gdp_vars = [var for var in ds.data_vars if var.startswith('GDP_') and var.endswith('_affected')]
            for gdp_var in gdp_vars:
                sector = gdp_var.replace('GDP_', '').replace('_affected', '')
                values = scenario_data[gdp_var].values
                results.append({
                    'scenario': scenario,
                    'variable': f'GDP_{sector}',
                    'mean': np.mean(values),
                    'p5': np.percentile(values, 5),
                    'p95': np.percentile(values, 95)
                })
            
            est_vars = [var for var in ds.data_vars if var.startswith('EST_') and var.endswith('_affected')]
            total_est = sum(scenario_data[var].values for var in est_vars)
            results.append({
                'scenario': scenario,
                'variable': 'EST_TOTAL',
                'mean': np.mean(total_est),
                'p5': np.percentile(total_est, 5),
                'p95': np.percentile(total_est, 95)
            })
            
            pop_values = scenario_data['mean_pop_affected'].values
            results.append({
                'scenario': scenario,
                'variable': 'POP_AFFECTED',
                'mean': np.mean(pop_values),
                'p5': np.percentile(pop_values, 5),
                'p95': np.percentile(pop_values, 95)
            })
    
    df = pd.DataFrame(results)
    df['ci_width'] = df['p95'] - df['p5']
    df['uncertainty_pct'] = (df['ci_width'] / df['mean']) * 100
    
    return df.round(2)


def run_io_analysis(confidence_df):
    """Run Input-Output model analysis"""
    
    io = InputOutputModel("10sector")
    results = []

    for scenario in confidence_df['scenario'].unique():
        scenario_data = confidence_df[confidence_df['scenario'] == scenario]
        gdp_data = scenario_data[scenario_data['variable'].str.startswith('GDP_')]
        
        for conf_level in ['mean', 'p5', 'p95']:
            va_shock = pd.Series(0.0, index=io.sectors)
            
            for _, row in gdp_data.iterrows():
                sector = row['variable'].replace('GDP_', '')
                if sector in io.sectors:
                    va_shock[sector] = -row[conf_level]
            
            output_impacts = io.total_output_from_value_added(va_shock)
            
            for sector in io.sectors:
                results.append({
                    'scenario': scenario.replace('e_', '').replace('-hazard A/ph', ''),
                    'confidence': conf_level,
                    'sector': sector,
                    'direct_shock': va_shock[sector],
                    'total_impact': output_impacts[sector],
                    'multiplier_effect': output_impacts[sector] - va_shock[sector]
                })

    return pd.DataFrame(results)


def run_policy(model_instance, policy_params, scenario_name="Policy Simulation"):
    """Run a policy simulation using the standard CGE model"""
    
    baseline = {
        'Y': {i: model_instance.Y[i].value for i in model_instance.i},
        'Z': {i: model_instance.Z[i].value for i in model_instance.i},
        'Q': {i: model_instance.Q[i].value for i in model_instance.i},
        'Xp': {i: model_instance.Xp[i].value for i in model_instance.i},
        'Xg': {i: model_instance.Xg[i].value for i in model_instance.i},
        'Xv': {i: model_instance.Xv[i].value for i in model_instance.i},
        'E': {i: model_instance.E[i].value for i in model_instance.i},
        'M': {i: model_instance.M[i].value for i in model_instance.i},
        'F': {(h, i): model_instance.F[h, i].value for h in model_instance.h for i in model_instance.i},
        'pq': {i: model_instance.pq[i].value for i in model_instance.i},
        'pf': {h: model_instance.pf[h].value for h in model_instance.h},
        'pz': {i: model_instance.pz[i].value for i in model_instance.i},
        'Td': model_instance.Td.value,
        'Sp': model_instance.Sp.value,
        'Sg': model_instance.Sg.value,
        'epsilon': model_instance.epsilon.value,
        'welfare': model_instance.obj.expr()
    }
    
    policy_model = copy.deepcopy(model_instance)
    
    logger.info(f"Applying policy scenario: {scenario_name}")
    for param_name, param_value in policy_params.items():
        if '.' in param_name:
            base_param, index = param_name.split('.')
            if hasattr(policy_model, base_param):
                if index in getattr(policy_model, base_param):
                    setattr(getattr(policy_model, base_param)[index], 'value', param_value)
                    logger.debug(f"Set {base_param}[{index}] = {param_value}")
                else:
                    logger.warning(f"Index {index} not found in {base_param}")
            else:
                logger.warning(f"Parameter {base_param} not found in model")
        else:
            if hasattr(policy_model, param_name):
                setattr(policy_model, param_name, param_value)
                logger.debug(f"Set {param_name} = {param_value}")
            else:
                logger.warning(f"Parameter {param_name} not found in model")
    
    solver = SolverFactory('ipopt', executable="/home/pve_ubuntu/miniconda3/envs/spw-env/bin/ipopt")
    solver.options['max_iter'] = 5000
    solver.options['tol'] = 1e-6
    
    logger.info("Solving policy scenario...")
    results = solver.solve(policy_model, tee=False)
    
    baseline_gdp = 0
    policy_gdp = 0
    
    for h in policy_model.h:
        for i in policy_model.i:
            baseline_gdp += baseline['F'][(h, i)]
            policy_gdp += policy_model.F[h, i].value
    
    baseline['gdp'] = baseline_gdp
    
    policy_results = {
        'Y': {i: policy_model.Y[i].value for i in policy_model.i},
        'Z': {i: policy_model.Z[i].value for i in policy_model.i},
        'Q': {i: policy_model.Q[i].value for i in policy_model.i},
        'Xp': {i: policy_model.Xp[i].value for i in policy_model.i},
        'Xg': {i: policy_model.Xg[i].value for i in policy_model.i},
        'Xv': {i: policy_model.Xv[i].value for i in policy_model.i},
        'E': {i: policy_model.E[i].value for i in policy_model.i},
        'M': {i: policy_model.M[i].value for i in policy_model.i},
        'F': {(h, i): policy_model.F[h, i].value for h in policy_model.h for i in policy_model.i},
        'pq': {i: policy_model.pq[i].value for i in policy_model.i},
        'pf': {h: policy_model.pf[h].value for h in policy_model.h},
        'pz': {i: policy_model.pz[i].value for i in policy_model.i},
        'Td': policy_model.Td.value,
        'Sp': policy_model.Sp.value,
        'Sg': policy_model.Sg.value,
        'epsilon': policy_model.epsilon.value,
        'welfare': policy_model.obj.expr(),
        'gdp': policy_gdp
    }
    
    pct_changes = {}
    for var in ['Y', 'Z', 'Q', 'Xp', 'Xg', 'Xv', 'E', 'M', 'pq', 'pz']:
        pct_changes[var] = {
            i: ((policy_results[var][i] / baseline[var][i] - 1) * 100) 
            if baseline[var][i] != 0 else float('nan')
            for i in policy_model.i
        }
    
    pct_changes['pf'] = {
        h: ((policy_results['pf'][h] / baseline['pf'][h] - 1) * 100)
        if baseline['pf'][h] != 0 else float('nan')
        for h in policy_model.h
    }
    
    pct_changes['F'] = {
        (h, i): ((policy_results['F'][(h, i)] / baseline['F'][(h, i)] - 1) * 100)
        if baseline['F'][(h, i)] != 0 else float('nan')
        for h in policy_model.h for i in policy_model.i
    }
    
    for var in ['Td', 'Sp', 'Sg', 'epsilon', 'welfare', 'gdp']:
        pct_changes[var] = ((policy_results[var] / baseline[var] - 1) * 100) \
            if baseline[var] != 0 else float('nan')

    logger.info(f"RESULTS FOR {scenario_name}")
    
    return {
        'baseline': baseline,
        'policy': policy_results,
        'pct_changes': pct_changes,
        'solver_status': {
            'status': results.solver.status,
            'termination_condition': results.solver.termination_condition
        }
    }


def apply_cge_to_confidence_intervals(confidence_df, sam_type="us"):
    """Apply CGE model to confidence intervals"""
    baseline = run_cge_example(sam_type=sam_type, display_results=False)

    va0 = {i: sum(value(baseline.F[h, i]) for h in baseline.h) for i in baseline.i}
    b0  = {i: value(baseline.b[i]) for i in baseline.i}

    out = []
    for sc in confidence_df['scenario'].unique():
        df_sc = confidence_df[
            (confidence_df['scenario'] == sc) &
            (confidence_df['variable'].str.startswith('GDP_'))
        ]

        for lev in ['mean', 'p5', 'p95']:
            policy_params, direct = {}, {}

            for _, r in df_sc.iterrows():
                i = r['variable'].replace('GDP_', '')
                if i in va0 and va0[i] > 0:
                    pct = float(r[lev]) / va0[i]
                    pct = max(0.0, min(0.99, pct))
                    policy_params[f"b.{i}"] = b0[i] * (1.0 - pct)
                    direct[i] = -float(r[lev])

            res = run_policy(baseline, policy_params, scenario_name=f"{sc}_{lev}")

            for i in baseline.i:
                dZ = res['policy']['Z'][i] - res['baseline']['Z'][i]
                out.append({
                    'scenario': sc.replace('e_', '').replace('-hazard A/ph', ''),
                    'confidence': lev,
                    'sector': i,
                    'direct_shock': direct.get(i, 0.0),
                    'total_impact': dZ,
                    'multiplier_effect': dZ - direct.get(i, 0.0),
                    'price_effect': res['pct_changes'].get('pq', {}).get(i, 0.0),
                    'gdp_contribution': sum(
                        res['policy']['F'][(h, i)] - res['baseline']['F'][(h, i)]
                        for h in baseline.h
                    )
                })

    return pd.DataFrame(out).round(2)


def main():
    """Main analysis function"""
    
    logger.info("Loading and Processing Data")
    data = load_and_process_data()
    
    logger.info("Calculating Confidence Intervals")
    confidence_df = get_confidence_intervals(data['combined_ds'])
    
    logger.info("Running I-O Model Analysis")
    io_results_df = run_io_analysis(confidence_df)
    
    logger.info("Running CGE Model Analysis")
    cge_results_df = apply_cge_to_confidence_intervals(confidence_df, sam_type="us")
    
    logger.info("Generating Visualizations")
    plot_vuln_trafos(data['mean_vuln_all'], data['df_lines'])
    plot_econo_naics(io_results_df, model_type="io")
    # plot_econo_naics(cge_results_df, model_type="cge")
    
    plot_socio_economic_impact(io_results_df, confidence_df, model_type="io")
    # plot_socio_economic_impact(cge_results_df, confidence_df, model_type="cge")

    logger.info("Saving Results")
    io_results_df.to_csv(FIGURES_DIR / "io_model_results.csv", index=False)
    # cge_results_df.to_csv(FIGURES_DIR / "cge_model_results.csv", index=False)
    confidence_df.to_csv(FIGURES_DIR / "confidence_intervals.csv", index=False)
    
    return {
        'io_results': io_results_df,
        # 'cge_results': cge_results_df,
        'confidence_intervals': confidence_df,
        'data': data
    }


if __name__ == "__main__":
    results = main()
    logger.info("Economic analysis completed successfully!")