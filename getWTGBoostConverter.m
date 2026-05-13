% =========================================================================
% getWTGBoostConverter.m
% Wind Turbine Generator Unidirectional Boost Converter State Equations
%
% Inputs:
%   vwt   : WTG capacitor voltage             (V)   [state x_w1]
%   iLw   : Boost inductor current            (A)   [state x_w2]
%   Vdc   : DC bus voltage                    (V)   
%   Sw    : Switch signal                     {0,1} [control input]
%   v_w   : Wind speed                        (m/s)
%
% Outputs:
%   dvwt  : d(vwt)/dt                         (V/s)
%   diLw  : d(iLw)/dt                         (A/s)
%   iw    : Output current injected to DC bus (A)
%   Pwt   : WTG instantaneous power           (W)
% =========================================================================

function [dvwt, diLw, iw, Pwt] = getWTGBoostConverter(vwt, iLw, Vdc, Sw, v_w)

    % --- Converter parameters ---
    Cw  = 2000e-6;      % WTG side capacitance     [F]
    Lw  = 1e-3;         % Boost inductance         [H]

    % --- WTG array current ---
    % vwt must be clamped positive before passing to getWindTurbine
    vwt_safe        = max(vwt, 0);
    [iwt, Pwt]      = getWindTurbine(vwt_safe, v_w);

    % --- State equations ---
    % Cw * dvwt/dt = iwt(vwt) - iLw
    dvwt    = (iwt - iLw) / Cw;

    % Lw * diLw/dt  = vwt - (1 - Sw) * Vdc
    diLw    = (vwt - (1 - Sw) * Vdc) / Lw;

    % --- Output current injected to DC bus ---
    % iw = (1 - Sw) * iLw
    iw      = (1 - Sw) * iLw;

end
