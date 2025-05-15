#!/usr/bin/env python3
"""
Run GIC analysis scenarios.
1. Calculate storm maxima
2. Fit power law  
3. Estimate GIC (includes admittance matrix)
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from configs import setup_logger

logger = setup_logger(log_file="logs/run_scenarios.log")

def run_script(script_name):
    """Run a script in the scripts folder."""
    script_path = Path(__file__).parent / "scripts" / script_name
    
    # Set PYTHONPATH to include parent directory for configs
    env = os.environ.copy()
    parent_dir = str(Path(__file__).parent)
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{parent_dir}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = parent_dir
    
    logger.info(f"Running {script_name}")
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True, env=env)
        logger.info(f"✓ {script_name} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {script_name} failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run GIC analysis scripts")
    parser.add_argument("script", choices=["storm", "stat", "gic", "admittance", "all"],
                       help="Which script to run")
    args = parser.parse_args()
    
    script_map = {
        "storm": ["calc_storm_maxes.py"],
        "stat": ["stat_analysis.py"],
        "gic": ["est_gic.py"],
        "admittance": ["build_admittance_matrix.py"],
        "all": ["calc_storm_maxes.py", "stat_analysis.py", "est_gic.py"]
    }
    
    scripts = script_map[args.script]
    
    for script in scripts:
        if not run_script(script):
            logger.error("Pipeline stopped due to error")
            sys.exit(1)
    
    logger.info("Completed successfully")

if __name__ == "__main__":
    main()