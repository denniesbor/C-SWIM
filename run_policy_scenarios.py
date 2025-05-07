import pandas as pd
import numpy as np
from pyomo.environ import value
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import custom modules
from io_model import InputOutputModel
from cge_data_model import SAMDataManager, run_cge_example


def compare_electricity_failure_impacts(
    affected_sectors: Dict[str, float],
    sam_type: str = "us",
    output_dir: str = "results"
):
    """
    Compare the economic impacts of electricity failures using both 
    IO and CGE modeling approaches.
    
    Parameters:
    -----------
    affected_sectors : Dict[str, float]
        Dictionary of sectors and their affected production capacity (0-1)
        E.g., {'MANUF': 0.3} means 30% of manufacturing capacity is affected
    sam_type : str
        Type of SAM to use ("us", "japan", etc.)
    output_dir : str
        Directory to save results
    
    Returns:
    --------
    Dict
        Comparison results between IO and CGE models
    """
    print(f"Analyzing economic impacts of electricity failure on {len(affected_sectors)} sectors")
    
    # Step 1: Run IO model analysis
    io_results = run_io_analysis(affected_sectors, sam_type)
    
    # Step 2: Run CGE model analysis
    cge_results = run_cge_analysis(affected_sectors, sam_type)
    
    # Step 3: Compare results
    comparison = compare_model_results(io_results, cge_results)
    
    return {
        "io_results": io_results,
        "cge_results": cge_results,
        "comparison": comparison
    }


def run_io_analysis(affected_sectors: Dict[str, float], sam_type: str = "us"):
    """
    Run Input-Output analysis for electricity failure impacts.
    
    Parameters:
    -----------
    affected_sectors : Dict[str, float]
        Dictionary of sectors and their affected production capacity percentages (0-1)
    sam_type : str
        Type of SAM to use
        
    Returns:
    --------
    Dict
        IO model results
    """
    print("\n=== Running Input-Output Analysis ===")
    
    # Initialize IO model
    io = InputOutputModel("10sector")
    
    # Load the value-added data that's already in the model
    # The model automatically loads this from value_added.csv
    va_data = None
    try:
        # Read value-added data directly from the CSV
        va_df = pd.read_csv(io.path / "value_added.csv", index_col=0)
        # Sum all components (labor, capital, etc.) to get total value added by sector
        va_data = pd.Series(va_df.sum(axis=0), index=io.sectors)
        print("  Loaded value-added data successfully")
    except Exception as e:
        print(f"  Warning: Could not load value-added data: {e}")
        # Use output values as fallback (typical value-added ratio could be around 40-50%)
        va_data = pd.Series(io.X * 0.45, index=io.sectors)
        print("  Using estimated value-added data instead")
    
    # Create value-added shock vector
    d_va = pd.Series(0, index=io.sectors)
    
    # Apply shocks to affected sectors based on their actual value-added amounts
    for sector, impact_pct in affected_sectors.items():
        if sector in d_va.index:
            # Calculate absolute shock value based on the sector's value added
            sector_va = va_data[sector]
            shock_value = -sector_va * impact_pct  # Negative shock
            d_va[sector] = shock_value
            print(f"  Applied {impact_pct*100:.1f}% reduction to {sector} value-added → shock of {shock_value:.2f}")
            
    print("\nValue-added shock vector:")
    print(d_va.round(2))
    
    # Calculate total output impacts using supply-side model
    dx_va = io.total_output_from_value_added(d_va)
    
    # Calculate output multipliers
    multipliers = io.output_multipliers()
    
    # Display results
    print("\nIO Model Results - Output changes by sector:")
    print(dx_va.round(2))
    print(f"\nTotal output impact: {dx_va.sum():.2f}")
    
    # Calculate sector contributions to total impact
    total_impact = dx_va.sum()
    sector_contributions = (dx_va / total_impact * 100) if total_impact != 0 else pd.Series(0, index=io.sectors)
    
    print("\nSector contribution to total impact:")
    for sector in io.sectors:
        if abs(sector_contributions[sector]) > 1:  # Only show significant contributions
            print(f"  {sector}: {sector_contributions[sector]:.1f}%")
    
    return {
        "output_changes": dx_va,
        "shock_vector": d_va,
        "value_added": va_data,
        "multipliers": multipliers,
        "sector_contributions": sector_contributions,
        "total_impact": total_impact,
        "model_type": "IO"
    }
    
    
def run_policy(model_instance, policy_params, scenario_name="Policy Simulation"):
    """
    Run a policy simulation using the standard CGE model.
    
    Args:
        model_instance: A concrete instance of the CGE model
        policy_params: Dictionary of policy parameters to be modified
        scenario_name: Name of the policy scenario (for reporting)
        
    Returns:
        results: Dictionary containing simulation results
    """
    import copy
    from pyomo.opt import SolverFactory
    
    # Store the baseline values for comparison
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
    
    # Create a copy of the model instance to avoid modifying the original
    policy_model = copy.deepcopy(model_instance)
    
    # Apply policy changes
    print(f"\nApplying policy scenario: {scenario_name}")
    for param_name, param_value in policy_params.items():
        if '.' in param_name:
            # Handle compound parameters (e.g., 'taum.AGR')
            base_param, index = param_name.split('.')
            if hasattr(policy_model, base_param):
                if index in getattr(policy_model, base_param):
                    setattr(getattr(policy_model, base_param)[index], 'value', param_value)
                    print(f"  Set {base_param}[{index}] = {param_value}")
                else:
                    print(f"  Warning: Index {index} not found in {base_param}")
            else:
                print(f"  Warning: Parameter {base_param} not found in model")
        else:
            # Handle scalar parameters
            if hasattr(policy_model, param_name):
                setattr(policy_model, param_name, param_value)
                print(f"  Set {param_name} = {param_value}")
            else:
                print(f"  Warning: Parameter {param_name} not found in model")
    
    # Solve the policy model
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 5000
    solver.options['tol'] = 1e-6
    
    print("\nSolving policy scenario...")
    results = solver.solve(policy_model, tee=True)
    
    # Calculate GDP (value added approach)
    baseline_gdp = 0
    policy_gdp = 0
    
    # Sum factor incomes (value added)
    for h in policy_model.h:
        for i in policy_model.i:
            baseline_gdp += baseline['F'][(h, i)]
            policy_gdp += policy_model.F[h, i].value
    
    # Store GDP values
    baseline['gdp'] = baseline_gdp
    
    # Collect policy results
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
    
    # Calculate percentage changes
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

    # Report key results
    print("\n" + "="*80)
    print(f"RESULTS FOR {scenario_name}")
    print("="*80)
    
    # Return detailed results
    return {
        'baseline': baseline,
        'policy': policy_results,
        'pct_changes': pct_changes,
        'solver_status': {
            'status': results.solver.status,
            'termination_condition': results.solver.termination_condition
        }
    }


def run_cge_analysis(affected_sectors: Dict[str, float], sam_type: str):
    """
    Run CGE analysis for electricity failure impacts.
    
    Parameters:
    -----------
    affected_sectors : Dict[str, float]
        Dictionary of sectors and their affected production capacity
    sam_type : str
        Type of SAM to use
        
    Returns:
    --------
    Dict
        CGE model results
    """
    print("\n=== Running CGE Analysis ===")
    
    # Load SAM data
    sam_manager = SAMDataManager()
    sam_data = sam_manager.load_sam(sam_type)
    sam_df, h_list, i_list, u_list = sam_data
    
    # Run baseline CGE model
    print("  Running baseline model...")
    baseline_instance = run_cge_example(sam_type=sam_type, display_results=False)
    
    # Create policy parameters for the shock
    # We'll model electricity failure as reduced factor productivity
    policy_params = {}
    
    # For each affected sector, reduce the productivity parameter (b)
    for sector, impact in affected_sectors.items():
        if sector in baseline_instance.i:
            # Reduce productivity parameter by impact percentage
            current_b = value(baseline_instance.b[sector])
            policy_params[f'b.{sector}'] = current_b * (1 - impact)
            print(f"  Reducing productivity in {sector} by {impact*100:.1f}%")
    
    # Run policy scenario
    print("  Running policy scenario...")
    results = run_policy(baseline_instance, policy_params, 
                         scenario_name="Electricity Failure Scenario")
    
    # Extract key results
    output_changes = {}
    for i in baseline_instance.i:
        baseline_output = results['baseline']['Z'][i]
        policy_output = results['policy']['Z'][i]
        output_changes[i] = policy_output - baseline_output
    
    output_changes_series = pd.Series(output_changes)
    
    return {
        "output_changes": output_changes_series,
        "baseline": results['baseline'],
        "policy": results['policy'],
        "pct_changes": results['pct_changes'],
        "gdp_impact": results['policy']['gdp'] - results['baseline']['gdp'],
        "model_type": "CGE"
    }


def compare_model_results(io_results, cge_results):
    """
    Compare IO and CGE model results.
    
    Parameters:
    -----------
    io_results : Dict
        Results from IO analysis
    cge_results : Dict
        Results from CGE analysis
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    print("\n=== Comparing IO and CGE Results ===")
    
    # Align sectors across both models
    all_sectors = sorted(set(io_results["output_changes"].index) | 
                         set(cge_results["output_changes"].index))
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(index=all_sectors)
    
    # Add output changes from both models
    comparison['IO_Output_Change'] = io_results["output_changes"]
    comparison['CGE_Output_Change'] = cge_results["output_changes"]
    
    # Calculate percentage of total impact
    io_total = io_results["output_changes"].sum()
    cge_total = cge_results["output_changes"].sum()
    
    comparison['IO_Pct_of_Total'] = comparison['IO_Output_Change'] / io_total * 100
    comparison['CGE_Pct_of_Total'] = comparison['CGE_Output_Change'] / cge_total * 100
    
    # Calculate difference between models
    comparison['Difference'] = comparison['CGE_Output_Change'] - comparison['IO_Output_Change']
    comparison['Difference_Pct'] = comparison['Difference'] / comparison['IO_Output_Change'].abs() * 100
    
    # Fill NaN values for clean display
    comparison = comparison.fillna(0)
    
    # Print comparison
    print("\nSectoral impacts comparison (IO vs CGE):")
    print(comparison.round(2))
    
    # Print aggregate impacts
    print("\nAggregate impacts:")
    print(f"  IO Total Output Impact: {io_total:.2f}")
    print(f"  CGE Total Output Impact: {cge_total:.2f}")
    print(f"  Difference: {cge_total - io_total:.2f} ({(cge_total - io_total)/io_total*100:.2f}%)")
    
    # Display GDP impact only if available in both models
    if 'gdp_impact' in cge_results:
        print(f"  CGE GDP Impact: {cge_results['gdp_impact']:.2f}")
    
    return comparison


def plot_comparison(comparison, output_dir, filename_base):
    """
    Create visualizations of the comparison results.
    
    Parameters:
    -----------
    comparison : pd.DataFrame
        Comparison results
    output_dir : str
        Directory to save plots
    filename_base : str
        Base name for output files
    """
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Bar chart of output changes by sector
    plt.figure(figsize=(12, 8))
    sectors = comparison.index
    x = np.arange(len(sectors))
    width = 0.4
    
    plt.bar(x - width/2, comparison['IO_Output_Change'], width, label='IO Model')
    plt.bar(x + width/2, comparison['CGE_Output_Change'], width, label='CGE Model')
    
    plt.xlabel('Sectors')
    plt.ylabel('Output Change')
    plt.title('IO vs CGE: Output Changes by Sector')
    plt.xticks(x, sectors, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_base}_output_changes.png")
    plt.close()
    
    # Bar chart of percentage differences
    plt.figure(figsize=(12, 8))
    plt.bar(sectors, comparison['Difference_Pct'], color='steelblue')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.xlabel('Sectors')
    plt.ylabel('Difference (%)')
    plt.title('Percentage Difference: CGE vs IO Impacts')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename_base}_pct_difference.png")
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    # Define sectors affected by electricity failure
    # The values represent the portion of capacity affected (0-1)
    affected_sectors = {
        'UTIL_CONST': 0.20,  # 20% of utilities affected
        'MANUF': 0.15,       # 15% of manufacturing affected
        'INFO': 0.10,        # 10% of information sector affected
    }
    
    # Run the analysis
    results = compare_electricity_failure_impacts(
        affected_sectors, 
        sam_type="us",
        output_dir="electricity_failure_results"
    )