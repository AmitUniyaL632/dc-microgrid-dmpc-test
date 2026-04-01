% =========================================================================
% getPVBoostConverter.m
% PV Unidirectional Boost Converter State Equations
% Implements Equation 8 from Zhu et al., Renewable Energy 222 (2024) 119871
%
% Inputs:
%   vpv   : PV capacitor voltage              (V)   [state x1]
%   iL    : Boost inductor current            (A)   [state x2]
%   Vdc   : DC bus voltage                    (V)   [state x5]
%   Ss    : Switch signal                     {0,1} [control input]
%   G     : Solar irradiance                  (W/m^2)
%   T     : Cell temperature                  (degC)
%
% Outputs:
%   dvpv  : d(vpv)/dt                         (V/s)
%   diL   : d(iL)/dt                          (A/s)
%   is    : Output current injected to DC bus (A)
%   Ppv   : PV array instantaneous power      (W)
% =========================================================================

function [dvpv, diL, is, Ppv] = getPVBoostConverter(vpv, iL, Vdc, Ss, G, T)

    % --- Converter parameters (Table 4, Zhu et al.) ---
    Cs  = 2000e-6;      % PV side capacitance     [F]
    Ls  = 01e-3;       % Boost inductance         [H]

    % --- PV array current: nonlinear coupling (Eq. 8 comment + Eq. 1) ---
    % getPVArray returns [Ipv, Ppv] at the operating voltage vpv
    % vpv must be clamped positive before passing to getPVArray
    vpv_safe        = max(vpv, 0);
    [ipv, Ppv]      = getPVArray(vpv_safe, G, T, 5, 20);

    % --- State equations (Eq. 8) ---
    % Cs * dvpv/dt = ipv(vpv) - iL
    dvpv    = (ipv - iL) / Cs;

    % Ls * diL/dt  = vpv - (1 - Ss) * Vdc
    diL     = (vpv - (1 - Ss) * Vdc) / Ls;

    % --- Output current injected to DC bus ---
    % is = (1 - Ss) * iL
    is      = (1 - Ss) * iL;

end