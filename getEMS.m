% =========================================================================
% getEMS.m
% Energy Management System
% Implements Equation 32 from Zhu et al., Renewable Energy 222 (2024) 119871
%
% Inputs:
%   Pmax  : PV maximum available power    [W]
%   Pload : Load power demand             [W]
%   SOC   : Battery State of Charge       [0 to 1]
%
% Outputs:
%   zeta_a : AES  operational mode  (1=ON, 0=OFF)
%   zeta_p : PEMFCS operational mode (1=ON, 0=OFF)
% =========================================================================
function [zeta_a, zeta_p] = getEMS(Pmax, Pload, SOC)

    P_bat_max = 5000; % 5 kW power limit for the battery
    SOC_min   = 0.20; % 20% minimum SOC
    SOC_max   = 0.90; % 90% maximum SOC
    P_net     = Pmax - Pload; % Available energy balance

    % Default state: Both H2 systems idle
    zeta_a = 0;
    zeta_p = 0;

    % Evaluate surplus conditions
    if P_net > P_bat_max || (P_net > 0 && SOC >= SOC_max)
        % AE absorbs if surplus > 5 kW OR if battery is full
        zeta_a = 1;
    end

    % Evaluate deficit conditions
    if P_net < -P_bat_max || (P_net < 0 && SOC <= SOC_min)
        % FC supplies if deficit > 5 kW OR if battery is empty
        zeta_p = 1;
    end

end