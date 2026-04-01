% =========================================================================
% getBattery.m
% Battery Energy Storage System (BESS) Model
%
% This function models a basic Battery using an internal resistance model
% and Coulomb counting for State of Charge (SOC) tracking.
%
% Inputs:
%   I_bat    : Battery current (A) - Positive for discharge, negative for charge
%   SOC_prev : Previous State of Charge (fraction, 0 to 1)
%   Ts       : Sampling time (s)
%
% Outputs:
%   V_bat    : Battery terminal voltage (V)
%   SOC_new  : Updated State of Charge (fraction, 0 to 1)
%   P_bat    : Battery output power (W)
% =========================================================================
function [V_bat, SOC_new, P_bat] = getBattery(I_bat, SOC_prev, Ts)

    % --- Battery Parameters ---
    V_nom    = 240;            % Nominal voltage (V)
    R_int    = 0.15;           % Internal resistance (Ohms)
    C_Ah     = 100;            % Battery capacity in Ampere-hours (Ah)
    C_As     = C_Ah * 3600;    % Battery capacity in Ampere-seconds (Coulombs)

    % --- SOC Calculation (Coulomb Counting) ---
    SOC_new = max(0, min(1, SOC_prev - (I_bat * Ts) / C_As));

    % --- Terminal Voltage Calculation ---
    V_bat = V_nom - I_bat * R_int;

    % --- Power Calculation ---
    P_bat = V_bat * I_bat;
end