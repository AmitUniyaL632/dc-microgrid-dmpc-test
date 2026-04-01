% =========================================================================
% getBattery.m
% Battery Energy Storage System (BESS) Model
%
% This function models a basic Battery using an internal resistance model
% and Coulomb counting for State of Charge (SOC) tracking.
%
% Inputs:
%   I_bat    : Battery current (A) - Positive for discharge, negative for charge
%   SOC      : Current State of Charge (fraction, 0 to 1)
%
% Outputs:
%   V_bat    : Battery terminal voltage (V)
%   dSOC     : Derivative of State of Charge (1/s)
%   P_bat    : Battery output power (W)
% =========================================================================
function [V_bat, dSOC, P_bat] = getBattery(I_bat, SOC)

    % --- Battery Parameters ---
    V_nom    = 240;            % Nominal voltage (V)
    R_int    = 0.15;           % Internal resistance (Ohms)
    C_Ah     = 100;            % Battery capacity in Ampere-hours (Ah)
    C_As     = C_Ah * 3600;    % Battery capacity in Ampere-seconds (Coulombs)

    % --- SOC Derivative ---
    dSOC = -I_bat / C_As;

    % --- Terminal Voltage Calculation ---
    V_bat = V_nom - I_bat * R_int;

    % --- Power Calculation ---
    P_bat = V_bat * I_bat;
end