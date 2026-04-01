% =========================================================================
% getAE.m
% Alkaline Electrolyzer (AE) Stack Model
% Implements Equations 15-17 from:
% "PV/Hydrogen DC microgrid control using distributed economic MPC"
% Renewable Energy 222 (2024) 119871
%
% Inputs:
%   iae   : AE stack current (A) - scalar or vector
%
% Outputs:
%   vae   : AE stack voltage (V)
%   Pae   : AE stack power consumed (W)
%   NH2   : Hydrogen production rate (mol/s)
%   etaF  : Faraday (current) efficiency [-]
% =========================================================================

function [vae, Pae, NH2, etaF] = getAE(iae)

    % =====================================================================
    % AE STACK PARAMETERS  (Table 4 of reference paper)
    % =====================================================================
    Nae     = 30;           % Number of electrolyzer cells in series      [-]
    Tae     = 25;           % Electrolyzer cell temperature                [deg C]
    urev    = 1.229;        % Reversible cell voltage at 25C, 1atm        [V]
    r1      = 7.33e-5;      % Ohmic resistance coefficient 1              [Ohm*m^2]
    r2      = -1.11e-7;     % Ohmic resistance coefficient 2              [Ohm*m^2/degC]
    te      = 0.185;        % Overvoltage coefficient                      [V]
    t1      = 1.6e-2;       % Overvoltage parameter 1                     [m^2/A]
    t2      = -1.302;       % Overvoltage parameter 2                     [m^2*degC/A]
    t3      = 4.21e2;       % Overvoltage parameter 3                     [m^2*degC^2/A]
    A       = 0.25;         % Electrolyzer cell active area               [m^2]
    F       = 96485.3;      % Faraday constant                            [C/mol]
    f1      = 2.5e4;        % Current efficiency parameter 1              [A^2/m^4]
    f2      = 0.96;         % Current efficiency parameter 2              [-]

    % Guard against zero or negative current
    iae     = max(iae, 1e-6);

    % =====================================================================
    % STEP 1: Per-cell ohmic resistance term              [Part of Eq. 15]
    % R_ohm = (r1 + r2*Tae) / A
    % Units: [Ohm*m^2] / [m^2] = [Ohm]
    % =====================================================================
    R_ohm   = (r1 + r2 * Tae) / A;

    % =====================================================================
    % STEP 2: Per-cell overvoltage (Tafel) term          [Part of Eq. 15]
    % arg = (t1 + t2/Tae + t3/Tae^2) / A * iae + 1
    % V_over = te * ln(arg)
    % =====================================================================
    t_param = t1 + t2/Tae + t3/Tae^2;     % Combined temp-dependent term [m^2/A]
    arg     = (t_param / A) .* iae + 1;    % Argument of logarithm        [-]
    V_over  = te .* log(arg);              % Overvoltage per cell         [V]

    % =====================================================================
    % STEP 3: AE stack voltage                                    [Eq. 15]
    % vae = Nae * { urev + R_ohm*iae + V_over }
    % =====================================================================
    vae     = Nae .* (urev + R_ohm .* iae + V_over);

    % =====================================================================
    % STEP 4: AE stack power consumed                             [Eq. 16]
    % Pae = vae * iae
    % =====================================================================
    Pae     = vae .* iae;

    % =====================================================================
    % STEP 5: Faraday (current) efficiency                   [Below Eq. 17]
    % etaF = [(iae/A)^2 / (f1 + (iae/A)^2)] * f2
    % J = iae/A  [A/m^2]
    % =====================================================================
    J       = iae / A;                     % Current density              [A/m^2]
    etaF    = (J.^2 ./ (f1 + J.^2)) * f2; % Current efficiency           [-]

    % =====================================================================
    % STEP 6: Hydrogen production rate                            [Eq. 17]
    % NH2 = etaF * Nae * iae / (2*F)       [mol/s]
    % =====================================================================
    NH2     = etaF .* Nae .* iae ./ (2 * F);

end