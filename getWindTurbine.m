% =========================================================================
% getWindTurbine.m
% Wind Turbine Generator (WTG) Model
% Simplified aerodynamic and electrical model for DC Microgrid integration
%
% Inputs:
%   Vwt  : WTG output capacitor voltage (V) (DC side of rectifier)
%   v_w  : Wind speed (m/s)
%
% Outputs:
%   Iwt  : Output current of the WTG rectifier (A)
%   Pwt  : Output power of the WTG (W)
% =========================================================================

function [Iwt, Pwt] = getWindTurbine(Vwt, v_w)

    % =====================================================================
    % WIND TURBINE PARAMETERS
    % =====================================================================
    rho     = 1.225;     % Air density (kg/m^3)
    R       = 3.5;       % Turbine blade radius (m) - approx 10kW nominal
    A       = pi * R^2;  % Swept area (m^2)
    Cp_max  = 0.41;      % Maximum power coefficient (assumed fixed at MPPT)
    eta_gen = 0.90;      % Generator and rectifier combined efficiency

    % =====================================================================
    % MECHANICAL & ELECTRICAL POWER
    % =====================================================================
    % Available mechanical power from wind
    Pm = 0.5 * rho * A * Cp_max * (v_w^3);
    
    % Electrical power available at the DC side of the rectifier
    Pwt_available = Pm * eta_gen;
    
    % =====================================================================
    % DC CURRENT CALCULATION
    % =====================================================================
    % The WTG acts as a current source to the boost converter capacitor.
    % To prevent infinite current at 0V, we clamp the minimum voltage and 
    % enforce a maximum current limit (short circuit current limit).
    
    V_safe = max(Vwt, 1.0);     % Prevent division by zero
    Imax   = 100.0;             % Absolute maximum current limit (A)
    
    % Ideal current to extract available power
    Iwt_ideal = Pwt_available / V_safe;
    
    % Actual current is bounded by Imax
    Iwt = min(Iwt_ideal, Imax);
    
    % Actual power delivered (W)
    Pwt = V_safe * Iwt;
    
end
