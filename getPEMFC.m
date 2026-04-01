% =========================================================================
% getPEMFC.m
% PEMFC Stack Model
% Implements Equations 20-27 from:
% "PV/Hydrogen DC microgrid control using distributed economic MPC"
% Renewable Energy 222 (2024) 119871
%
% Inputs:
%   ipe   : PEMFC stack current (A) - scalar or vector
%
% Outputs:
%   vpe   : PEMFC stack output voltage (V)
%   Ppe   : PEMFC stack output power (W)
%   qH2   : Hydrogen consumption rate (mol/s)
% =========================================================================

function [vpe, Ppe, qH2] = getPEMFC(ipe)

    % =====================================================================
    % PEMFC STACK PARAMETERS  (Table 4 of reference paper)
    % =====================================================================
    Npe     = 300;              % Number of fuel cells in series          [-]
    Tpe     = 323.15;           % Fuel cell operating temperature         [K]
    PH2     = 3;                % Partial pressure of hydrogen            [atm]
    PO2     = 3;                % Partial pressure of oxygen              [atm]
    Ape_cm2 = 50;               % Membrane activation area                [cm^2]
    Ape_m2  = Ape_cm2 * 1e-4;  % Membrane activation area                [m^2]
    lambda  = 20;               % Water content of membrane               [-]
    l_mem   = 5.1e-5;           % Membrane thickness                      [m]
    Rc      = 3e-4;             % Contact resistance                      [Ohm]
    Jmax    = 1.5;              % Maximum current density                 [A/cm^2]

    % Physical constant
    F       = 96485.3;          % Faraday constant                        [C/mol]

    % =====================================================================
    % STEP 1: Oxygen concentration at cathode-catalyst interface  [Eq. 25]
    % CO2 = PO2 / (5.08e6 * exp(-498 / Tpe))
    % Units: PO2 in atm, result in mol/cm^3
    % =====================================================================
    CO2 = PO2 / (5.08e6 * exp(-498 / Tpe));

    % =====================================================================
    % STEP 2: Nernst thermodynamic EMF per cell                   [Eq. 21]
    % Enernst = 1.229 - 8.45e-4*(Tpe-298.15)
    %         + 4.3085e-5 * Tpe * (ln(PH2) + 0.5*ln(PO2))
    % =====================================================================
    Enernst = 1.229 ...
            - 8.45e-4  * (Tpe - 298.15) ...
            + 4.3085e-5 * Tpe * (log(PH2) + 0.5*log(PO2));

    % =====================================================================
    % STEP 3: Activation polarization voltage per cell             [Eq. 22]
    % vact = -0.9514 + 3.12e-3*Tpe + 7.4e-5*Tpe*ln(CO2)
    %        - 1.87e-4*Tpe*ln(ipe)
    % Note: ipe must be > 0 for log to be valid
    % =====================================================================
    ipe_safe = max(ipe, 1e-6);   % Guard against log(0)

    vact = -0.9514 ...
         + 3.12e-3  * Tpe ...
         + 7.4e-5   * Tpe * log(CO2) ...
         - 1.87e-4  * Tpe * log(ipe_safe);

    % =====================================================================
    % STEP 4: Membrane resistivity (Nafion)                        [Eq. 26]
    % Current density J = ipe / Ape   [A/cm^2] (Ape in cm^2)
    %
    % rm = 181.6 * [1 + 0.03*J + 0.062*(Tpe/303)^2 * J^2.5]
    %            / [(lambda - 0.634 - 3*J) * exp(4.18*(Tpe-303)/Tpe)]
    %
    % Units of rm: Ohm*m  (resistivity)
    % =====================================================================
    J = ipe_safe / Ape_cm2;    % Current density [A/cm^2]

    rm_num = 181.6 .* (1 + 0.03.*J + 0.062.*(Tpe/303).^2 .* J.^2.5);
    rm_den = (lambda - 0.634 - 3.*J) .* exp(4.18*(Tpe - 303)/Tpe);

    % Guard against denominator going to zero or negative
    rm_den  = max(rm_den, 1e-10);
    rm      = rm_num ./ rm_den;

    % =====================================================================
    % STEP 5: Ohmic polarization voltage per cell                  [Eq. 23]
    % vohm = -ipe * (rm*l / Ape + Rc)
    % rm*l/Ape: Ape in m^2, l in m, rm in Ohm*m  =>  result in Ohm
    % =====================================================================
    l_cm = l_mem * 100;

    vohm = -ipe_safe .* (rm .* l_cm ./ Ape_cm2 + Rc);

    % =====================================================================
    % STEP 6: Concentration polarization voltage per cell          [Eq. 24]
    % vcon = 0.016 * ln(1 - ipe / (Ape*Jmax))
    % Ape in cm^2, Jmax in A/cm^2 => Ape*Jmax in A (max current limit)
    % =====================================================================
    I_max   = Ape_cm2 * Jmax;                      % Max stack current [A]
    ratio   = ipe_safe ./ I_max;
    ratio   = min(ratio, 1 - 1e-6);                % Keep argument of log < 0
    vcon    = 0.016 .* log(1 - ratio);

    % =====================================================================
    % STEP 7: Stack output voltage                                 [Eq. 20]
    % vpe = Npe * (Enernst + vact + vohm + vcon)
    % Note: vact, vohm, vcon are all negative (losses)
    % =====================================================================
    vpe = Npe .* (Enernst + vact + vohm + vcon);
    vpe = max(vpe, 0);          % Physical lower bound

    % =====================================================================
    % STEP 8: Stack power output                                   [Eq. 27]
    % Ppe = vpe * ipe
    % =====================================================================
    Ppe = vpe .* ipe;

    % =====================================================================
    % STEP 9: Hydrogen consumption rate  (Faraday's law)
    % qH2 = Npe * ipe / (2*F)    [mol/s]
    % =====================================================================
    qH2 = Npe .* ipe_safe ./ (2 * F);

end