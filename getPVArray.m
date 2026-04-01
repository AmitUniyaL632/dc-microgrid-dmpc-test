% =========================================================================
% pv_array_model.m
% PV Array Model - Single Diode with Rs and Rsh
% Implements Equations 2.2 - 2.8 from reference paper
%
% Inputs:
%   Vpv  : PV array terminal voltage (V) - scalar or vector
%   G    : Solar irradiance (W/m^2)
%   T    : Cell temperature (degrees C)
%   Ns   : Number of PV modules in series per string
%   Nsh  : Number of parallel strings (Nsh = Np in some notations)
%
% Outputs:
%   Ipv  : PV array output current (A)
%   Ppv  : PV array output power (W)
% =========================================================================

function [Ipv, Ppv] = getPVArray(Vpv, G, T, Ns, Nsh)

    % =====================================================================
    % MODULE PARAMETERS (update from datasheet if different)
    % These correspond to STC: G = 1000 W/m^2, Tn = 25 C
    % =====================================================================
    Vocn    = 47.6;      % Module open-circuit voltage at STC (V)
    Iscn    = 9.99;      % Module short-circuit current at STC (A)
    Ncell   = 72;        % Number of cells per module

    % Series resistance and shunt resistance (module level)
    % These should ideally come from the datasheet or curve fitting.
    % Typical estimates used here - update if datasheet values are available.
    Rs      = 0.20;      % Series resistance per module (Ohm)   [Eq. 2.2]
    Rsh     = 250.0;     % Shunt resistance per module (Ohm)    [Eq. 2.2]

    % Diode and material parameters
    a       = 1.30;      % Diode ideality factor (dimensionless) [Eq. 2.2]
    ki      = 0.005;     % Temperature coefficient of Isc (A/C)  [Eq. 2.3]
                         % Typical: ~0.065% of Isc per degree C
                         % Update from datasheet (usually listed as mA/C or %/C)
    Egap    = 1.12;      % Band gap energy of silicon (eV)        [Eq. 2.4]

    % Physical constants
    q       = 1.602e-19; % Electron charge (C)
    k       = 1.38e-23;  % Boltzmann constant (J/K)

    % Nominal temperature
    Tn      = 25;        % Nominal temperature STC (degrees C)
    Tn_K    = Tn + 273.15; % Nominal temperature in Kelvin

    % =====================================================================
    % ARRAY-LEVEL EQUIVALENT RESISTANCES         [Eq. 2.8]
    % Rseq  = (Ns / Nsh) * Rs
    % Rsheq = (Ns / Nsh) * Rsh
    % =====================================================================
    Rseq    = (Ns / Nsh) * Rs;
    Rsheq   = (Ns / Nsh) * Rsh;

    % =====================================================================
    % TEMPERATURE CONVERSION
    % =====================================================================
    T_K     = T + 273.15; % Operating temperature in Kelvin

    % =====================================================================
    % THERMAL VOLTAGE (per cell)                 [Eq. 2.6]
    % Vt = k*T/q  (single cell thermal voltage at operating temperature)
    % Nominal thermal voltage at STC
    % =====================================================================
    Vt      = k * T_K / q;       % Operating thermal voltage (V/cell)
    Vt_n    = k * Tn_K / q;      % Nominal thermal voltage at STC (V/cell)

    % =====================================================================
    % PHOTOCURRENT (module level)                [Eq. 2.3]
    % Iph = (G/1000) * (Isc + ki * (T - Tn))
    % =====================================================================
    Iph     = (G / 1000) * (Iscn + ki * (T - Tn));

    % =====================================================================
    % NOMINAL SATURATION CURRENT at STC          [Eq. 2.5]
    % Ion = Isc / (exp(Vocn / (a * Ncell * Vt_n)) - 1)
    % =====================================================================
    Ion     = Iscn / (exp(Vocn / (a * Ncell * Vt_n)) - 1);

    % =====================================================================
    % SATURATION CURRENT at operating temperature [Eq. 2.4]
    % Io = Ion * (T/Tn)^3 * exp((q*Egap)/(k*a) * (1/Tn - 1/T))
    % =====================================================================
    Io      = Ion * (T_K / Tn_K)^3 * ...
              exp( (q * Egap) / (k * a) * (1/Tn_K - 1/T_K) );

    % =====================================================================
    % SOLVE IMPLICIT ARRAY EQUATION via Newton-Raphson [Eq. 2.7]
    %
    % F(Ipv) = Ipv
    %          - Nsh*Iph
    %          + Nsh*Io * (exp((Vpv + Ipv*Rseq) / (a*Ns*Ncell*Vt)) - 1)
    %          + (Vpv + Ipv*Rseq) / Rsheq
    %        = 0
    %
    % dF/dIpv = 1
    %           + Nsh*Io * Rseq/(a*Ns*Ncell*Vt)
    %             * exp((Vpv + Ipv*Rseq) / (a*Ns*Ncell*Vt))
    %           + Rseq / Rsheq
    % =====================================================================

    % Denominator of diode exponent for the full array
    Vt_array = a * Ns * Ncell * Vt;  % [V] - thermal voltage scaled for array

    % Newton-Raphson settings
    maxIter  = 50;
    tol      = 1e-9;

    % Initial guess: ideal current (no Rs, no Rsh losses)
    Ipv = Nsh * Iph - Nsh * Io * (exp(Vpv / Vt_array) - 1);
    Ipv = max(0, Ipv);  % Physical lower bound

    for iter = 1:maxIter
        exp_term = exp((Vpv + Ipv .* Rseq) / Vt_array);

        % Residual F(Ipv)
        F  =  Ipv ...
            - Nsh .* Iph ...
            + Nsh .* Io .* (exp_term - 1) ...
            + (Vpv + Ipv .* Rseq) ./ Rsheq;

        % Jacobian dF/dIpv
        dF = 1 ...
           + Nsh .* Io .* (Rseq / Vt_array) .* exp_term ...
           + Rseq / Rsheq;

        % Newton update
        delta   = F ./ dF;
        Ipv     = Ipv - delta;

        % Clamp after each iteration
        Ipv     = max(0, Ipv);

        % Check convergence
        if max(abs(delta)) < tol
            break;
        end
    end

    % =====================================================================
    % POWER OUTPUT
    % =====================================================================
    Ppv = Vpv .* Ipv;

end


