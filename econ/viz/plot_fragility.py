"""Reliability analysis visualization for transformer fragility and aging."""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm, weibull_min

from configs import (
    setup_logger,
    get_data_dir,
    FIGURES_DIR
)


DATA_LOC = get_data_dir(econ=True)
logger = setup_logger("visualization module")

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

warnings.filterwarnings("ignore")


theta0 = 75
beta_ln_range = [0.25, 0.35, 0.50]
theta_values = [50, 75, 100, 150]
beta_wb_range = [1.0, 2.0, 3.0]

x_gic = np.linspace(1, 200, 500)
x_age = np.linspace(0, 60, 500)

y_loc = 1.01
dy = 0.08

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8, 9.5), sharex=False)

colors_beta = ['#2E86AB', '#A23B72', '#F18F01']
colors_theta = ['#06A77D', '#2E86AB', '#D62839', '#8B1E3F']
colors_age = ['#0B3954', '#087E8B', '#BFD7EA']

ax = axes[0, 0]
theta_fix = 75
for beta, c in zip(beta_ln_range, colors_beta):
    mu = np.log(theta_fix)
    pdf = (1/(beta*x_gic*np.sqrt(2*np.pi))) * np.exp(-0.5*((np.log(x_gic)-mu)/beta)**2)
    ax.plot(x_gic, pdf, color=c, linewidth=1.2, label=f'$\\beta$ = {beta}')

ax.axvline(theta_fix, ls='--', lw=0.7, color='gray', alpha=0.6)
ax.set_xlim(0, 180)
ax.set_xlabel('GIC (A/ph)')
ax.set_ylabel('Probability Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.85)
ax.legend(loc='upper right', frameon=False, handlelength=1.5)
ax.text(0.0, y_loc+dy, '(a) GIC Fragility', fontsize=11, transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.0, y_loc, f'Effect of dispersion $\\beta$, $\\theta$ = {theta_fix} A/ph', 
        fontsize=10, transform=ax.transAxes, ha='left', va='bottom')

ax = axes[0, 1]
beta_fix = 0.35
for theta, c in zip(theta_values, colors_theta):
    Pf = norm.cdf((np.log(x_gic) - np.log(theta)) / beta_fix)
    ax.plot(x_gic, Pf, color=c, linewidth=1.2, label=f'$\\theta$ = {theta} A/ph')

ax.axhline(0.5, ls=':', lw=0.7, color='gray', alpha=0.5)
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.02)
ax.set_xlabel('GIC (A/ph)')
ax.set_ylabel('Failure Probability $P_f$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.85)
ax.legend(loc='lower right', frameon=False, handlelength=1.5, bbox_to_anchor=(1.05, 0.0))
ax.text(0.0, y_loc+dy, '(b) GIC Failure Probability', fontsize=11, transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.0, y_loc, f'Effect of threshold $\\theta$, $\\beta$ = {beta_fix}', 
        fontsize=10, transform=ax.transAxes, ha='left', va='bottom')

ax = axes[1, 0]
beta_age_vals = [1.5, 2.0, 2.5]
eta_age = 40

for beta_age, c in zip(beta_age_vals, colors_age):
    F_age = weibull_min.cdf(x_age, c=beta_age, scale=eta_age)
    theta_age = theta0 * (1 - 0.6 * F_age)
    ax.plot(x_age, theta_age, color=c, linewidth=1.2, label=f'$\\beta_{{age}}$ = {beta_age}')

ax.axhline(theta0, ls='--', lw=0.7, color='gray', alpha=0.6)
ax.set_xlim(0, 60)
ax.set_xlabel('Transformer Age (years)')
ax.set_ylabel('GIC Threshold $\\theta(t)$ (A/ph)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.85)
ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.05, 0.75), handlelength=1.5)
ax.text(0.0, y_loc+dy, '(c) Age-Dependent Degradation', fontsize=11, transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.0, y_loc, f'$\\theta(t)=\\theta_0(1-0.6F(t))$, $\\eta$ = {eta_age}y', 
        fontsize=10, transform=ax.transAxes, ha='left', va='bottom')

ax = axes[1, 1]
eta_fix = 40
for beta_wb, c in zip(beta_wb_range, colors_age):
    pdf = weibull_min.pdf(x_age, beta_wb, scale=eta_fix)
    ax.plot(x_age, pdf, color=c, linewidth=1.2, label=f'$\\beta$ = {beta_wb}')

ax.set_xlim(0, 60)
ax.set_xlabel('Transformer Age (years)')
ax.set_ylabel('Probability Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.85)
ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.05, 0.98), handlelength=1.2)
ax.text(0.0, y_loc+dy, '(d) Weibull Age Distribution', fontsize=11, transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.0, y_loc, f'Effect of shape $\\beta$, $\\eta$ = {eta_fix}y', 
        fontsize=10, transform=ax.transAxes, ha='left', va='bottom')

ax = axes[2, 0]
eta_fix = 40
for beta_wb, c in zip(beta_wb_range, colors_age):
    hazard = (beta_wb/eta_fix) * (x_age/eta_fix)**(beta_wb-1)
    ax.plot(x_age, hazard, color=c, linewidth=1.2, label=f'$\\beta$ = {beta_wb}')

ax.set_xlim(0, 60)
ax.set_ylim(0, 0.08)
ax.set_xlabel('Transformer Age (years)')
ax.set_ylabel('Failure Rate $\\lambda(t)$ (1/year)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.85)
ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.05, 0.98), handlelength=1.5)
ax.text(0.0, y_loc+dy, '(e) Age-Dependent Failure Rate', fontsize=11, transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.0, y_loc, f'Effect of shape $\\beta$, $\\eta$ = {eta_fix}y', 
        fontsize=10, transform=ax.transAxes, ha='left', va='bottom')

ax = axes[2, 1]
ages_demo = [10, 30, 50]
beta_frag_1 = 0.35
beta_frag_2 = 0.50
colors_demo = ['#06A77D', '#F18F01', '#D62839']

for age_demo, c in zip(ages_demo, colors_demo):
    beta_age = 2.0
    eta_age = 40
    F_age = weibull_min.cdf(age_demo, c=beta_age, scale=eta_age)
    theta_at_age = theta0 * (1 - 0.6 * F_age)
    
    Pf = norm.cdf((np.log(x_gic) - np.log(theta_at_age)) / beta_frag_1)
    ax.plot(x_gic, Pf, color=c, linewidth=1.2, linestyle='-',
             label=f'Age {age_demo}y, $\\beta$={beta_frag_1}')

for age_demo, c in zip(ages_demo, colors_demo):
    beta_age = 2.0
    eta_age = 40
    F_age = weibull_min.cdf(age_demo, c=beta_age, scale=eta_age)
    theta_at_age = theta0 * (1 - 0.6 * F_age)
    
    Pf = norm.cdf((np.log(x_gic) - np.log(theta_at_age)) / beta_frag_2)
    ax.plot(x_gic, Pf, color=c, linewidth=1.2, linestyle=':',
             label=f'Age {age_demo}y, $\\beta$={beta_frag_2}')

ax.axhline(0.5, ls='--', lw=0.7, color='gray', alpha=0.5)
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.02)
ax.set_xlabel('GIC (A/ph)')
ax.set_ylabel('Failure Probability $P_f$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, axis='y', linestyle=':', linewidth=0.6, alpha=0.85)
ax.legend(loc='lower right', frameon=False, fontsize=8, handlelength=1.2)
ax.text(0.0, y_loc+dy, '(f) Combined Effect', fontsize=11, transform=ax.transAxes, ha='left', va='bottom')
ax.text(0.0, y_loc, f'Age-dependent fragility with dispersion effect', 
        fontsize=10, transform=ax.transAxes, ha='left', va='bottom')

plt.tight_layout(h_pad=0.8)
plt.savefig(FIGURES_DIR / 'reliability_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'reliability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()