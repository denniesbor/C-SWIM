import numpy as np
from pyomo.environ import *


class StdModelDef:
    """
    Standard Computable General Equilibrium (CGE) Model
    ==================================================
    
    This class implements a standard multi-sector CGE model based on the framework
    presented in "Textbook of Computable General Equilibrium Modelling: Programming 
    and Simulations" by Hosoe, Gasawa, and Hashimoto (2010).
    
    Original GAMS implementation: stdcge.gms from GAMS Model Library
    Python/Pyomo port created by: cmb11 (July 10, 2017)
    This implementation adapts and extends that work with enhancements for
    numerical stability and additional features.
    
    Model Structure:
    ---------------
    1. Production Block:
       - Nested production structure with Cobb-Douglas value-added function
       - Leontief fixed-coefficient technology for intermediate inputs
       - Perfect competition in all markets
       - Constant returns to scale
    
    2. Household Block:
       - Representative household maximizing Cobb-Douglas utility
       - Income from factor ownership (labor and capital)
       - Expenditure on consumption, savings, and direct taxes
    
    3. Government Block:
       - Revenue from direct taxes, production taxes, and import tariffs
       - Expenditure on consumption and transfers
       - Fixed government savings rate
    
    4. Savings-Investment Block:
       - Savings from households, government, and foreign sources
       - Investment allocated across sectors using fixed coefficients
    
    5. International Trade Block:
       - Armington assumption for imports (imperfect substitution between domestic and imported goods)
       - Constant Elasticity of Transformation (CET) function for export supply
       - Small open economy (world prices are exogenous)
       - Flexible exchange rate to clear the balance of payments
    
    6. Market Clearing Conditions:
       - Supply equals demand for all commodities
       - Factor markets clear (full employment)
       - Macro balances (savings-investment, government budget, external account)
    
    Mathematical Formulation:
    ------------------------
    The model consists of 24 sets of equations:
    
    1. Composite factor aggregation function (Cobb-Douglas):
       Y[i] = b[i] * prod(h, F[h,i]^beta[h,i])
    
    2. Factor demand function:
       F[h,i] = beta[h,i] * py[i] * Y[i] / pf[h]
    
    3. Intermediate demand function:
       X[i,j] = ax[i,j] * Z[j]
    
    4. Composite factor demand function:
       Y[i] = ay[i] * Z[i]
    
    5. Unit cost function:
       pz[j] = ay[j] * py[j] + sum(i, ax[i,j] * pq[i])
    
    6. Direct tax revenue function:
       Td = taud * sum(h, pf[h] * FF[h])
    
    7. Production tax revenue function:
       Tz[i] = tauz[i] * pz[i] * Z[i]
    
    8. Import tariff revenue function:
       Tm[i] = taum[i] * pm[i] * M[i]
    
    9. Government demand function:
       Xg[i] = mu[i] * (Td + sum(j,Tz[j]) + sum(j,Tm[j]) - Tr - Sg) / pq[i]
    
    10. Investment demand function:
        Xv[i] = lambd[i] * (Sp + Sg + epsilon * Sf) / pq[i]
    
    11. Private saving function:
        Sp = ssp * sum(h, pf[h] * FF[h])
    
    12. Government saving function:
        Sg = ssg * (Td + sum(j,Tz[j]) + sum(j,Tm[j]))
    
    13. Household demand function:
        Xp[i] = alpha[i] * (sum(h, pf[h] * FF[h]) + Tr - Sp - Td) / pq[i]
    
    14. World export price equation:
        pe[i] = epsilon * pWe[i]
    
    15. World import price equation:
        pm[i] = epsilon * pWm[i]
    
    16. Balance of payments:
        sum(i, pWe[i] * E[i]) + Sf = sum(i, pWm[i] * M[i])
    
    17. Armington's function:
        Q[i] = gamma[i] * (deltam[i]*M[i]^eta[i] + deltad[i]*D[i]^eta[i])^(1/eta[i])
    
    18. Import demand function:
        M[i] = (gamma[i]^eta[i] * deltam[i] * pq[i] / ((1+taum[i])*pm[i]))^(1/(1-eta[i])) * Q[i]
    
    19. Domestic good demand function:
        D[i] = (gamma[i]^eta[i] * deltad[i] * pq[i] / pd[i])^(1/(1-eta[i])) * Q[i]
    
    20. Transformation function:
        Z[i] = theta[i] * (xie[i]*E[i]^phi[i] + xid[i]*D[i]^phi[i])^(1/phi[i])
    
    21. Export supply function:
        E[i] = (theta[i]^phi[i] * xie[i] * (1+tauz[i]) * pz[i] / pe[i])^(1/(1-phi[i])) * Z[i]
    
    22. Domestic good supply function:
        D[i] = (theta[i]^phi[i] * xid[i] * (1+tauz[i]) * pz[i] / pd[i])^(1/(1-phi[i])) * Z[i]
    
    23. Market clearing for composite good:
        Q[i] = Xp[i] + Xg[i] + Xv[i] + sum(j, X[i,j])
    
    24. Factor market clearing condition:
        sum(i, F[h,i]) = FF[h]
    
    Calibration:
    -----------
    The model is calibrated using a Social Accounting Matrix (SAM) which represents
    the circular flow of all transactions in the base year. From this SAM, the model
    derives:
    
    - Share parameters (alpha, beta, delta, etc.)
    - Scale parameters (b, gamma, theta)
    - Tax rates (taud, tauz, taum)
    - Input-output coefficients (ax, ay)
    - Initial values for all variables
    
    The model assumes that the base SAM represents an equilibrium state of the economy.
    
    Closure Rules:
    -------------
    - Neoclassical savings-driven closure: Investment is determined by savings
    - Fixed factor supplies (full employment)
    - Flexible exchange rate to clear the external balance
    - Government savings is a fixed proportion of revenue
    
    Usage:
    -----
    This class provides the model definition but requires data (sets and parameters)
    from a properly formatted SAM to create a concrete instance. The model() method
    returns a Pyomo abstract model that can be instantiated with data and solved.
    
    The model is typically used for conducting counterfactual policy simulations to
    analyze the economy-wide impacts of various policy changes such as:
    - Tax reforms
    - Trade liberalization
    - External shocks
    - Sectoral policy interventions
    
    References:
    ----------
    Hosoe, N., Gasawa, K., & Hashimoto, H. (2010). Textbook of Computable General 
    Equilibrium Modelling: Programming and Simulations. Palgrave Macmillan.
    
    Original Python/Pyomo port by cmb11 (July 10, 2017)
    
    Returns:
    -------
    AbstractModel: Pyomo abstract model object representing the CGE model structure
    """

    def model(self):
        # ------------------------------------------- #
        # MODEL OBJECT: "Container for problem"
        self.m = AbstractModel()

        # ------------------------------------------- #
        # DEFINE SETS
        self.m.i = Set(doc='Goods')
        self.m.h = Set(doc='Factors')
        self.m.u = Set(doc='SAM entry (row/column labels)')

        # ------------------------------------------- #
        # DEFINE PARAMETERS
        self.m.sam = Param(self.m.u, self.m.u,
                           doc='Social Accounting Matrix')

        # === TAX AND FACTOR VALUES === #
        
        def Td0_init(model):
            # Td0 = SAM("GOV", "HOH")  # direct tax
            return model.sam['GOV', 'HOH']

        self.m.Td0 = Param(
            initialize=Td0_init,
            doc='Direct tax', mutable=True)

        def Tz0_init(model, i):
            # Tz0(i) = SAM("IDT", i)  # production tax
            return model.sam['IDT', i]

        self.m.Tz0 = Param(
            self.m.i, initialize=Tz0_init,
            doc='Production tax', mutable=True)

        def Tm0_init(model, i):
            # Tm0(i) = SAM("TRF", i)  # import tariff
            return model.sam['TRF', i]

        self.m.Tm0 = Param(
            self.m.i, initialize=Tm0_init,
            doc='Import tariff', mutable=True)

        def F0_init(model, h, i):
            # F0(h,i) = SAM(h, i)  # factor use
            return model.sam[h, i]

        self.m.F0 = Param(
            self.m.h, self.m.i, initialize=F0_init,
            doc='The h-th factor input by the i-th firm', mutable=True)

        def Y0_init(model, i):
            # Y0(i) = sum(h, F0(h,i))  # composite factor (value added)
            return sum(model.F0[h, i] for h in model.h)

        self.m.Y0 = Param(
            self.m.i, initialize=Y0_init,
            doc='Composite factor', mutable=True)

        # === INTERMEDIATE INPUTS, OUTPUT, IMPORTS === #
        
        def X0_init(model, i, j):
            # X0(i,j) = SAM(i, j)  # intermediate input
            return model.sam[i, j]

        self.m.X0 = Param(
            self.m.i, self.m.i, initialize=X0_init,
            doc='Intermediate input', mutable=True)

        def Z0_init(model, j):
            # Z0(j) = Y0(j) + sum(i, X0(i,j))  # total output
            return model.Y0[j] + sum(model.X0[i, j] for i in model.i)

        self.m.Z0 = Param(
            self.m.i, initialize=Z0_init,
            doc='Output of the j-th good', mutable=True)

        def M0_init(model, i):
            # M0(i) = SAM("EXT", i)  # imports
            return model.sam['EXT', i]

        self.m.M0 = Param(
            self.m.i, initialize=M0_init,
            doc='Imports', mutable=True)

        # === TAX RATES === #
        
        def tauz_init(model, i):
            # tauz(i) = Tz0(i) / Z0(i)  # production tax rate
            return model.Tz0[i] / (model.Z0[i] + 1e-10)

        self.m.tauz = Param(
            self.m.i, initialize=tauz_init,
            doc='Production tax rate', mutable=True)

        def taum_init(model, i):
            # taum(i) = Tm0(i) / M0(i)  # import tariff rate
            return model.Tm0[i] / (model.M0[i] + 1e-10)

        self.m.taum = Param(
            self.m.i, initialize=taum_init,
            doc='Import tariff rate', mutable=True)

        # === FINAL DEMANDS & FACTOR ENDOWMENT === #
        
        def Xp0_init(model, i):
            # Xp0(i) = SAM(i, "HOH")  # household consumption
            return model.sam[i, 'HOH']

        self.m.Xp0 = Param(
            self.m.i, initialize=Xp0_init,
            doc='Household consumption of the i-th good', mutable=True)

        def FF_init(model, h):
            # FF(h) = SAM("HOH", h)  # factor endowment
            return model.sam['HOH', h]

        self.m.FF = Param(
            self.m.h, initialize=FF_init,
            doc='Factor endowment of the h-th factor', mutable=True)

        def Xg0_init(model, i):
            # Xg0(i) = SAM(i, "GOV")  # government consumption
            return model.sam[i, 'GOV']

        self.m.Xg0 = Param(
            self.m.i, initialize=Xg0_init,
            doc='Government consumption', mutable=True)

        def Xv0_init(model, i):
            # Xv0(i) = SAM(i, "INV")  # investment demand (with small positive floor)
            small_positive = 1e-6
            raw_value = model.sam[i, 'INV']
            return max(raw_value, small_positive)
        
        self.m.Xv0 = Param(
            self.m.i, initialize=Xv0_init,
            doc='Investment demand', mutable=True)
        
        def E0_init(model, i):
            # E0(i) = SAM(i, "EXT")  # exports
            return model.sam[i, 'EXT']

        self.m.E0 = Param(
            self.m.i, initialize=E0_init,
            doc='Exports', mutable=True)

        # === COMPOSITE DEMAND & DOMESTIC SUPPLY === #
        
        def Q0_init(model, i):
            # Q0(i) = Xp0(i) + Xg0(i) + Xv0(i) + sum(j, X0(i,j))  # Armington composite
            return (model.Xp0[i]
                    + model.Xg0[i]
                    + model.Xv0[i]
                    + sum(model.X0[i, j] for j in model.i))

        self.m.Q0 = Param(
            self.m.i, initialize=Q0_init,
            doc="Armington's composite good", mutable=True)

        def D0_init(model, i):
            # D0(i) = (1 + tauz(i)) * Z0(i) - E0(i)  # domestic goods supply
            return (1 + model.tauz[i]) * model.Z0[i] - model.E0[i]

        self.m.D0 = Param(
            self.m.i, initialize=D0_init,
            doc='Domestic good', mutable=True)

        # === SAVINGS === #
        
        def Sp0_init(model):
            # Sp0 = SAM("INV", "HOH")  # private saving
            return model.sam['INV', 'HOH']

        self.m.Sp0 = Param(
            initialize=Sp0_init,
            doc='Private saving', mutable=True)

        def Sg0_init(model):
            # Sg0 = SAM("INV", "GOV")  # government saving
            return model.sam['INV', 'GOV']

        self.m.Sg0 = Param(
            initialize=Sg0_init,
            doc='Government saving', mutable=True)

        def Sf_init(model):
            # Sf = SAM("INV", "EXT")  # foreign saving in US dollars
            return model.sam['INV', 'EXT']

        self.m.Sf = Param(
            initialize=Sf_init,
            doc='Foreign saving in US dollars', mutable=True)

        # === NUMERAIRE FIXING === #
        
        def pWe_init(model, i):
            # pWe(i) = 1  # world export price (numeraire)
            return 1

        self.m.pWe = Param(
            self.m.i, initialize=pWe_init,
            doc='Export price in US dollars', mutable=True)

        def pWm_init(model, i):
            # pWm(i) = 1  # world import price (numeraire)
            return 1

        self.m.pWm = Param(
            self.m.i, initialize=pWm_init,
            doc='Import price in US dollars', mutable=True)
        
        
        # ------------------------------------------- #
        # CALIBRATION

        # === ELASTICITIES === #

        def sigma_init(model, i):
            # Elasticity of substitution (CES Armington function parameter)
            return 2  # For almost realistic US economy

        self.m.sigma = Param(
            self.m.i, initialize=sigma_init,
            doc='Elasticity of substitution')

        def psi_init(model, i):
            # Elasticity of transformation (CET function parameter)
            return 2  # For a realistic US economy

        self.m.psi = Param(
            self.m.i, initialize=psi_init,
            doc='Elasticity of transformation')

        def eta_init(model, i):
            # eta(i) = (sigma(i) - 1) / sigma(i)  # Substitution parameter
            return (model.sigma[i] - 1) / model.sigma[i]

        self.m.eta = Param(
            self.m.i, initialize=eta_init,
            doc='Substitution elasticity parameter')

        def phi_init(model, i):
            # phi(i) = (psi(i) + 1) / psi(i)  # Transformation parameter
            return (model.psi[i] + 1) / model.psi[i]

        self.m.phi = Param(
            self.m.i, initialize=phi_init,
            doc='Transformation elasticity parameter')

        # === UTILITY FUNCTION === #

        def alpha_init(model, i):
            # alpha(i) = Xp0(i) / sum(j, Xp0(j))  # Utility share parameter
            return model.Xp0[i] / sum(model.Xp0[j] for j in model.i)

        self.m.alpha = Param(
            self.m.i, initialize=alpha_init,
            doc='Share parameter in utility func.')

        # === PRODUCTION FUNCTION (COBB-DOUGLAS) === #

        def beta_init(model, h, i):
            # beta(h,j) = F0(h,j) / sum(k, F0(k,j))  # Factor share parameter
            return model.F0[h, i] / sum(model.F0[k, i] for k in model.h)

        self.m.beta = Param(
            self.m.h, self.m.i, initialize=beta_init,
            doc='Share parameter in production func.')

        def b_init(model, i):
            # b(j) = Y0(j) / prod(h, F0(h,j)^beta(h,j))  # Scale parameter
            return (model.Y0[i]
                    / np.prod([model.F0[h, i]**model.beta[h, i]
                            for h in model.h]))

        self.m.b = Param(
            self.m.i, initialize=b_init,
            doc='Scale parameter in production func.', mutable=True)

        # === INTERMEDIATE INPUTS === #

        def ax_init(model, i, j):
            # ax(i,j) = X0(i,j) / Z0(j)  # Input-output coefficient
            return model.X0[i, j] / model.Z0[j]

        self.m.ax = Param(
            self.m.i, self.m.i, initialize=ax_init,
            doc='Intermediate input requirement coeff.')

        def ay_init(model, i):
            # ay(j) = Y0(j) / Z0(j)  # Value-added coefficient
            return model.Y0[i] / model.Z0[i]

        self.m.ay = Param(
            self.m.i, initialize=ay_init,
            doc='Composite factor input req. coeff.')

        # === GOVERNMENT & INVESTMENT DEMAND SHARES === #

        def mu_init(model, i):
            # mu(i) = Xg0(i) / sum(j, Xg0(j))  # Government consumption share
            return model.Xg0[i] / sum(model.Xg0[j] for j in model.i)

        self.m.mu = Param(
            self.m.i, initialize=mu_init,
            doc='Government consumption share')

        def lambd_init(model, i):
            # lambda(i) = Xv0(i) / (Sp0 + Sg0 + Sf)  # Investment demand share
            return (model.Xv0[i]
                    / (model.Sp0 + model.Sg0 + model.Sf))

        self.m.lambd = Param(
            self.m.i, initialize=lambd_init,
            doc='Investment demand share')

        # === ARMINGTON (CES IMPORT DEMAND) === #

        def deltam_init(model, i):
            # deltam(i) = ((1 + taum(i)) * M0(i)^(1 - eta(i))) / 
            #             ((1 + taum(i)) * M0(i)^(1 - eta(i)) + D0(i)^(1 - eta(i)))
            return (
                (1 + model.taum[i])*model.M0[i]**(1 - model.eta[i])
                / (
                    (1 + model.taum[i])*model.M0[i]**(1 - model.eta[i])
                    + model.D0[i]**(1 - model.eta[i])
                )
            )

        self.m.deltam = Param(
            self.m.i, initialize=deltam_init,
            doc='Share par. in Armington func.')

        def deltad_init(model, i):
            # deltad(i) = D0(i)^(1 - eta(i)) / 
            #             ((1 + taum(i)) * M0(i)^(1 - eta(i)) + D0(i)^(1 - eta(i)))
            return (
                model.D0[i]**(1 - model.eta[i])
                / (
                    (1 + model.taum[i])*model.M0[i]**(1 - model.eta[i])
                    + model.D0[i]**(1 - model.eta[i])
                )
            )

        self.m.deltad = Param(
            self.m.i, initialize=deltad_init,
            doc='Share par. in Armington func.')

        def gamma_init(model, i):
            # gamma(i) = Q0(i) / ((deltam(i) * M0(i)^eta(i) + deltad(i) * D0(i)^eta(i))^(1 / eta(i)))
            denom = (model.deltam[i]
                    * model.M0[i]**model.eta[i]
                    + model.deltad[i]
                    * model.D0[i]**model.eta[i])
            return model.Q0[i] / (denom**(1 / model.eta[i]))

        self.m.gamma = Param(
            self.m.i, initialize=gamma_init,
            doc='Scale par. in Armington func.')

        # === CET EXPORT TRANSFORMATION === #

        def xie_init(model, i):
            # xie(i) = E0(i)^(1 - phi(i)) / (E0(i)^(1 - phi(i)) + D0(i)^(1 - phi(i)))
            # Use a small positive number to avoid division by zero
            epsilon = 1e-10
            
            # Add epsilon to both E0 and D0 to avoid any division by zero or 0^negative issues
            E_term = (model.E0[i] + epsilon)**(1 - model.phi[i])
            D_term = (model.D0[i] + epsilon)**(1 - model.phi[i])
            
            # Calculate the share parameter
            return E_term / (E_term + D_term)

        self.m.xie = Param(
            self.m.i, initialize=xie_init,
            doc='Share par. in transformation func.')

        def xid_init(model, i):
            # xid(i) = D0(i)^(1 - phi(i)) / (E0(i)^(1 - phi(i)) + D0(i)^(1 - phi(i)))
            epsilon = 1e-10
            
            E_term = (model.E0[i] + epsilon)**(1 - model.phi[i])
            D_term = (model.D0[i] + epsilon)**(1 - model.phi[i])
            
            return D_term / (E_term + D_term)

        self.m.xid = Param(
            self.m.i, initialize=xid_init,
            doc='Share par. in transformation func.')

        def theta_init(model, i):
            # theta(i) = Z0(i) / ((xie(i) * E0(i)^phi(i) + xid(i) * D0(i)^phi(i))^(1 / phi(i)))
            denom = (model.xie[i]*model.E0[i]**model.phi[i]
                    + model.xid[i]*model.D0[i]**model.phi[i])
            return model.Z0[i] / (denom**(1 / model.phi[i]))

        self.m.theta = Param(
            self.m.i, initialize=theta_init,
            doc='Scale par. in transformation func.')

        # === SAVINGS RATES AND DIRECT TAX RATE === #

        def ssp_init(model):
            # ssp = Sp0 / sum(h, FF(h))  # Private savings rate
            return model.Sp0 / sum(model.FF[h] for h in model.h)

        self.m.ssp = Param(
            initialize=ssp_init,
            doc='Average propensity for private saving')

        def ssg_init(model):
            # ssg = Sg0 / (Td0 + sum(j, Tz0(j)) + sum(j, Tm0(j)))  # Government savings rate
            return (model.Sg0
                    / (
                        model.Td0
                        + sum(model.Tz0[i] for i in model.i)
                        + sum(model.Tm0[i] for i in model.i)
                    )
                )

        self.m.ssg = Param(
            initialize=ssg_init,
            doc='Average propensity for gov. saving')

        def taud_init(model):
            # taud = Td0 / sum(h, FF(h))  # Direct tax rate
            return (model.Td0
                    / sum(model.FF[h] for h in model.h))

        self.m.taud = Param(
            initialize=taud_init,
            doc='Direct tax rate')

        # === GOVERNMENT TRANSFERS === #

        def Tr0_init(model):
            # Tr0 = SAM("HOH", "GOV")  # Government transfers to households
            return model.sam['HOH', 'GOV']
                
        self.m.Tr0 = Param(
            initialize=Tr0_init,
            doc='Government transfers to households', mutable=True)
        
        # ------------------------------------------- #
        # DEFINE VARIABLES

        # === PRODUCTION VARIABLES === #

        self.m.Y = Var(
            self.m.i,
            initialize=Y0_init,
            within=PositiveReals,
            doc='(Eq Y var) Composite factor')

        self.m.F = Var(
            self.m.h, self.m.i,
            initialize=F0_init,
            within=PositiveReals,
            doc='(Eq F var) Factor input by the i-th firm')

        self.m.X = Var(
            self.m.i, self.m.i,
            initialize=X0_init,
            within=NonNegativeReals,
            doc='(Eq X var) Intermediate input')

        self.m.Z = Var(
            self.m.i,
            initialize=Z0_init,
            within=PositiveReals,
            doc='(Eq Z var) Output of the good')

        # === FINAL DEMAND VARIABLES === #

        self.m.Xp = Var(
            self.m.i,
            initialize=Xp0_init,
            within=NonNegativeReals,
            doc='(Eq Xp var) Household consumption')

        self.m.Xg = Var(
            self.m.i,
            initialize=Xg0_init,
            within=NonNegativeReals,
            doc='(Eq Xg var) Government consumption')

        self.m.Xv = Var(
            self.m.i,
            initialize=Xv0_init,
            within=NonNegativeReals,
            doc='(Eq Xv var) Investment demand')

        # === TRADE VARIABLES === #

        self.m.E = Var(
            self.m.i,
            initialize=E0_init,
            within=NonNegativeReals,
            doc='(Eq E var) Exports')

        EPS = 1e-8          # Tiny positive number; small enough not to matter for results
        # Imports (was NonNegativeReals → now strictly ≥ EPS)
        self.m.M = Var(
            self.m.i,
            initialize=M0_init,
            within=NonNegativeReals,      # OK to keep this
            bounds=(EPS, None),           # <-- new bound to prevent numerical issues
            doc="(Eq M var) Imports")

        # === ARMINGTON VARIABLES === #

        # Armington composite good (already PositiveReals, but tighten anyway)
        self.m.Q = Var(
            self.m.i,
            initialize=Q0_init,
            within=PositiveReals,
            bounds=(EPS, None),           # <-- small lower bound for numerical stability
            doc="(Eq Q var) Armington's composite good")

        # Domestic good (already PositiveReals)
        self.m.D = Var(
            self.m.i,
            initialize=D0_init,
            within=PositiveReals,
            bounds=(EPS, None),           # <-- small lower bound for numerical stability
            doc="(Eq D var) Domestic good")

        # === PRICE VARIABLES === #

        def p_init(model, v):
            return 1                      # Initialize all prices to 1 (normalization)

        self.m.pf = Var(
            self.m.h,
            initialize=p_init,
            within=PositiveReals,
            doc='The h-th factor price')

        self.m.py = Var(
            self.m.i,
            initialize=p_init,
            within=PositiveReals,
            doc='Composite factor price')

        self.m.pz = Var(
            self.m.i,
            initialize=p_init,
            within=PositiveReals,
            doc='Supply price of the i-th good')

        self.m.pq = Var(
            self.m.i,
            initialize=p_init,
            within=PositiveReals,
            doc="Armington's composite good price")

        self.m.pe = Var(
            self.m.i,
            initialize=p_init,
            within=PositiveReals,
            doc='Export price in local currency')

        self.m.pm = Var(
            self.m.i,
            initialize=p_init,
            within=PositiveReals,
            doc='Import price in local currency')

        self.m.pd = Var(
            self.m.i,
            initialize=p_init,
            within=PositiveReals,
            doc='The i-th domestic good price')

        # === EXCHANGE RATE, SAVINGS, AND TAX VARIABLES === #

        # When defining the exchange rate variable
        self.m.epsilon = Var(
            initialize=1,
            within=PositiveReals,
            bounds=(0.95, 1.05),          # Restricts movement to ±5%
            doc='(Eq epsilon var) Exchange rate')

        self.m.Sp = Var(
            initialize=Sp0_init,
            within=PositiveReals,
            doc='(Eq Sp var) Private saving')

        self.m.Sg = Var(
            initialize=Sg0_init,
            within=Reals,                 # Can be negative (government deficit)
            doc='(Eq Sg var) Government saving')

        self.m.Td = Var(
            initialize=Td0_init,
            within=PositiveReals,
            doc='(Eq Td var) Direct tax')

        self.m.Tz = Var(
            self.m.i,
            initialize=Tz0_init,
            within=NonNegativeReals,
            doc='(Eq Tz var) Production tax')

        self.m.Tm = Var(
            self.m.i,
            initialize=Tm0_init,
            within=NonNegativeReals,
            doc='(Eq Tm var) Import tariff')

        # === GOVERNMENT TRANSFERS === #

        # Govt transfers
        self.m.Tr = Var(
            initialize=Tr0_init,
            within=PositiveReals,
            doc='Government transfers to households')
        
        # ------------------------------------------- #
        # DEFINE EQUATIONS

        # === PRODUCTION BLOCK === #

        # (Eq PY)
        def eqpy_rule(model, i):
            # Y[i] = b[i] * prod(h, F[h,i]^beta[h,i])  # Cobb-Douglas production function
            return model.Y[i] == model.b[i] * np.prod([
                model.F[h, i]**model.beta[h, i] for h in model.h
            ])

        self.m.eqpy = Constraint(
            self.m.i, rule=eqpy_rule,
            doc='(1) Composite factor aggregation function')

        # (Eq F)
        def eqF_rule(model, h, i):
            # F[h,i] = beta[h,i] * py[i] * Y[i] / pf[h]  # Factor demand (first-order condition)
            return model.F[h, i] == (model.beta[h, i] * model.py[i]
                                    * model.Y[i] / model.pf[h])

        self.m.eqF = Constraint(
            self.m.h, self.m.i, rule=eqF_rule,
            doc='(2) Factor demand function')

        # (Eq X)
        def eqX_rule(model, i, j):
            # X[i,j] = ax[i,j] * Z[j]  # Leontief intermediate input demand
            return model.X[i, j] == model.ax[i, j] * model.Z[j]

        self.m.eqX = Constraint(
            self.m.i, self.m.i, rule=eqX_rule,
            doc='(3) Intermediate demand function')

        # (Eq Y)
        def eqY_rule(model, i):
            # Y[i] = ay[i] * Z[i]  # Leontief value-added demand
            return model.Y[i] == model.ay[i] * model.Z[i]

        self.m.eqY = Constraint(
            self.m.i, rule=eqY_rule,
            doc='(4) Composite factor demand function')

        # (Eq pzs)
        def eqpzs_rule(model, j):
            # pz[j] = ay[j] * py[j] + sum(i, ax[i,j] * pq[i])  # Unit cost function
            return model.pz[j] == (model.ay[j] * model.py[j]
                                + sum(model.ax[i, j] * model.pq[i]
                                        for i in model.i))

        self.m.eqpzs = Constraint(
            self.m.i, rule=eqpzs_rule,
            doc='(5) Unit cost function')

        # === GOVERNMENT BLOCK === #

        # (Eq Td)
        def eqTd_rule(model):
            # Td = taud * sum(h, pf[h] * FF[h])  # Direct tax revenue
            return model.Td == model.taud * sum(model.pf[h] * model.FF[h]
                                                for h in model.h)

        self.m.eqTd = Constraint(
            rule=eqTd_rule,
            doc='(6) Direct tax revenue function')

        # (Eq Tz)
        def eqTz_rule(model, i):
            # Tz[i] = tauz[i] * pz[i] * Z[i]  # Production tax revenue
            return model.Tz[i] == model.tauz[i] * model.pz[i] * model.Z[i]

        self.m.eqTz = Constraint(
            self.m.i, rule=eqTz_rule,
            doc='(7) Production tax revenue function')

        # (Eq Tm)
        def eqTm_rule(model, i):
            # Tm[i] = taum[i] * pm[i] * M[i]  # Import tariff revenue
            return model.Tm[i] == model.taum[i] * model.pm[i] * model.M[i]

        self.m.eqTm = Constraint(
            self.m.i, rule=eqTm_rule,
            doc='(8) Import tariff revenue function')

        # (Eq Xg)
        def eqXg_rule(model, i):
            # Xg[i] = mu[i] * (Td + sum(j,Tz[j]) + sum(j,Tm[j]) - Tr - Sg) / pq[i]  # Govt consumption
            govt_income = (model.Td + sum(model.Tz[j] for j in model.i) 
                        + sum(model.Tm[j] for j in model.i))
            govt_spending = model.Tr + model.Sg  # Transfers + Savings
            remaining_for_consumption = govt_income - govt_spending
            return model.Xg[i] == (model.mu[i] * remaining_for_consumption / model.pq[i])

        self.m.eqXg = Constraint(
            self.m.i, rule=eqXg_rule,
            doc='(9) Government demand function')

        # === INVESTMENT BLOCK === #

        # (Eq Xv)
        def eqXv_rule(model, i):
            # Xv[i] = lambd[i] * (Sp + Sg + epsilon * Sf) / pq[i]  # Investment demand
            return model.Xv[i] == (model.lambd[i]
                                * (model.Sp + model.Sg
                                    + model.epsilon * model.Sf)
                                / model.pq[i])

        self.m.eqXv = Constraint(
            self.m.i, rule=eqXv_rule,
            doc='(10) Investment demand function')

        # === SAVINGS BLOCK === #

        # (Eq Sp)
        def eqSp_rule(model):
            # Sp = ssp * sum(h, pf[h] * FF[h])  # Private savings
            return model.Sp == model.ssp * sum(model.pf[h] * model.FF[h]
                                            for h in model.h)

        self.m.eqSp = Constraint(
            rule=eqSp_rule,
            doc='(11) Private saving function')

        # (Eq Sg)
        def eqSg_rule(model):
            # Sg = ssg * (Td + sum(j, Tz[j]) + sum(j, Tm[j]))  # Government savings
            govt_income = (model.Td
                        + sum(model.Tz[j] for j in model.i)
                        + sum(model.Tm[j] for j in model.i))
            return model.Sg == model.ssg * govt_income

        self.m.eqSg = Constraint(
            rule=eqSg_rule,
            doc='(12) Government saving function')

        # === HOUSEHOLD CONSUMPTION BLOCK === #

        # (Eq Xp)
        def eqXp_rule(model, i):
            # Xp[i] = alpha[i] * (sum(h, pf[h] * FF[h]) + Tr - Sp - Td) / pq[i]  # Household demand
            disposable_income = (sum(model.pf[h] * model.FF[h] for h in model.h)
                                + model.Tr  # Add transfers to income
                                - model.Sp - model.Td)
            return model.Xp[i] == (model.alpha[i] * disposable_income / model.pq[i])

        self.m.eqXp = Constraint(
            self.m.i, rule=eqXp_rule,
            doc='(13) Household demand function')

        # === INTERNATIONAL TRADE BLOCK === #

        # (Eq pe)
        def eqpe_rule(model, i):
            # pe[i] = epsilon * pWe[i]  # Export price in local currency
            return model.pe[i] == (model.epsilon * model.pWe[i])

        self.m.eqpe = Constraint(
            self.m.i, rule=eqpe_rule,
            doc='(14) World export price equation')

        # (Eq pm)
        def eqpm_rule(model, i):
            # pm[i] = epsilon * pWm[i]  # Import price in local currency
            return model.pm[i] == (model.epsilon * model.pWm[i])

        self.m.eqpm = Constraint(
            self.m.i, rule=eqpm_rule,
            doc='(15) World import price equation')

        # (Eq epsilon)
        def eqepsilon_rule(model):
            # sum(i, pWe[i] * E[i]) + Sf = sum(i, pWm[i] * M[i])  # Balance of payments
            return (sum(model.pWe[i] * model.E[i] for i in model.i)
                    + model.Sf
                    == sum(model.pWm[i] * model.M[i]
                        for i in model.i))

        self.m.eqepsilon = Constraint(
            rule=eqepsilon_rule,
            doc='(16) Balance of payments')

        # === ARMINGTON BLOCK === #

        # (Eq pqs)
        def eqpqs_rule(model, i):
            # Q[i] = gamma[i] * (deltam[i]*M[i]^eta[i] + deltad[i]*D[i]^eta[i])^(1/eta[i])  # CES aggregation
            return (model.Q[i] == model.gamma[i]
                    * (model.deltam[i]*model.M[i]**model.eta[i]
                    + model.deltad[i]*model.D[i]**model.eta[i]
                    )**(1 / model.eta[i]))

        self.m.eqpqs = Constraint(
            self.m.i, rule=eqpqs_rule,
            doc="(17) Armington's function")

        # (Eq M)
        def eqM_rule(model, i):
            # M[i] = (gamma[i]^eta[i] * deltam[i] * pq[i] / ((1+taum[i])*pm[i]))^(1/(1-eta[i])) * Q[i]
            return (model.M[i] == (
                (model.gamma[i]**model.eta[i]
                * model.deltam[i] * model.pq[i]
                / ((1 + model.taum[i]) * model.pm[i])
                )**(1 / (1 - model.eta[i]))
                * model.Q[i]))

        self.m.eqM = Constraint(
            self.m.i, rule=eqM_rule,
            doc='(18) Import demand function')

        # (Eq D)
        def eqD_rule(model, i):
            # D[i] = (gamma[i]^eta[i] * deltad[i] * pq[i] / pd[i])^(1/(1-eta[i])) * Q[i]
            return (model.D[i] == (
                (model.gamma[i]**model.eta[i]
                * model.deltad[i] * model.pq[i]
                / model.pd[i]
                )**(1 / (1 - model.eta[i]))
                * model.Q[i]))

        self.m.eqD = Constraint(
            self.m.i, rule=eqD_rule,
            doc='(19) Domestic good demand function')

        # === CET TRANSFORMATION BLOCK === #

        # (Eq pzd)
        def eqpzd_rule(model, i):
            # Z[i] = theta[i] * (xie[i]*E[i]^phi[i] + xid[i]*D[i]^phi[i])^(1/phi[i])  # CET transformation
            return (model.Z[i]
                    == model.theta[i]
                    * (model.xie[i]*model.E[i]**model.phi[i]
                    + model.xid[i]*model.D[i]**model.phi[i]
                    )**(1 / model.phi[i]))

        self.m.eqpzd = Constraint(
            self.m.i, rule=eqpzd_rule,
            doc='(20) Transformation function')

        # (Eq E)
        def eqE_rule(model, i):
            # E[i] = (theta[i]^phi[i] * xie[i] * (1+tauz[i]) * pz[i] / pe[i])^(1/(1-phi[i])) * Z[i]
            return (model.E[i] == (
                (model.theta[i]**model.phi[i]
                * model.xie[i]
                * (1 + model.tauz[i])
                * model.pz[i]
                / model.pe[i]
                )**(1 / (1 - model.phi[i]))
                * model.Z[i]))

        self.m.eqE = Constraint(
            self.m.i, rule=eqE_rule,
            doc='(21) Export supply function')

        # (Eq Ds)
        def eqDs_rule(model, i):
            # D[i] = (theta[i]^phi[i] * xid[i] * (1+tauz[i]) * pz[i] / pd[i])^(1/(1-phi[i])) * Z[i]
            return (model.D[i] == (
                (model.theta[i]**model.phi[i]
                * model.xid[i]
                * (1 + model.tauz[i])
                * model.pz[i]
                / model.pd[i]
                )**(1 / (1 - model.phi[i]))
                * model.Z[i]))

        self.m.eqDs = Constraint(
            self.m.i, rule=eqDs_rule,
            doc='(22) Domestic good supply function')

        # === MARKET CLEARING CONDITIONS === #

        # (Eq pqd)
        def eqpqd_rule(model, i):
            # Q[i] = Xp[i] + Xg[i] + Xv[i] + sum(j, X[i,j])  # Composite good market clearing
            return (model.Q[i]
                    == (model.Xp[i]
                        + model.Xg[i]
                        + model.Xv[i]
                        + sum(model.X[i, j] for j in model.i)))

        self.m.eqpqd = Constraint(
            self.m.i, rule=eqpqd_rule,
            doc='(23) Market clearing for composite good')

        # (Eq pf)
        def eqpf_rule(model, h):
            # sum(i, F[h,i]) = FF[h]  # Factor market clearing
            return sum(model.F[h, i] for i in model.i) == model.FF[h]

        self.m.eqpf = Constraint(
            self.m.h, rule=eqpf_rule,
            doc='(24) Factor market clearing condition')

        # === GOVERNMENT TRANSFERS === #

        # Add constraint to fix transfers in baseline
        def eqTr_rule(model):
            # Tr = Tr0  # Government transfers fixed at base level
            return model.Tr == model.Tr0

        self.m.eqTr = Constraint(
            rule=eqTr_rule,
            doc='Government transfers')

        # ------------------------------------------- #
        # DEFINE OBJECTIVE

        def obj_rule(model):
            # U = prod(i, Xp[i]^alpha[i])  # Cobb-Douglas utility function
            return np.prod([
                model.Xp[i]**model.alpha[i]
                for i in model.i
            ])

        self.m.obj = Objective(
            rule=obj_rule,
            sense=maximize,
            doc='Utility function [fictitious]')

        print("stdcge model loaded")
        return self.m
