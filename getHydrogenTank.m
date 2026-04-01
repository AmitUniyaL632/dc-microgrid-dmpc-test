% =========================================================================
% getHydrogenTank.m
% Compressed Hydrogen Tank Model
%
% This function models a compressed hydrogen storage tank using the ideal
% gas law to determine the pressure based on the net flow of hydrogen.
%
% Inputs:
%   n_H2_prev : Previous amount of hydrogen in the tank (mol)
%   NH2       : Hydrogen production rate from Electrolyzer (mol/s)
%   qH2       : Hydrogen consumption rate by PEMFC (mol/s)
%   Ts        : Sampling time (s)
%
% Outputs:
%   P_tank    : Current tank pressure (Pa)
%   n_H2_new  : Updated amount of hydrogen in the tank (mol)
% =========================================================================

function [P_tank, n_H2_new] = getHydrogenTank(n_H2_prev, NH2, qH2, Ts)
    % Tank Parameters
    V_tank = 0.1;           % Volume of the tank (m^3)
    T_tank = 298.15;        % Operating temperature (K) (approx 25 C)
    R      = 8.31446;       % Universal gas constant (J/(mol*K))

    % Mass Balance (Euler Integration)
    n_H2_new = max(n_H2_prev + (NH2 - qH2) * Ts, 0);

    % Pressure Calculation (Ideal Gas Law)
    P_tank = (n_H2_new * R * T_tank) / V_tank;
end