% =========================================================================
% getEMS.m
% Energy Management System
% Implements Equation 32 from Zhu et al., Renewable Energy 222 (2024) 119871
%
% Inputs:
%   Pmax  : PV maximum available power    [W]
%   Pload : Load power demand             [W]
%
% Outputs:
%   zeta_a : AES  operational mode  (1=ON, 0=OFF)
%   zeta_p : PEMFCS operational mode (1=ON, 0=OFF)
% =========================================================================
function [zeta_a, zeta_p] = getEMS(Pmax, Pload)

    P_bat_max = 5000; % 5 kW power limit for the battery
    P_net     = Pmax - Pload; % Available energy balance

    if P_net > P_bat_max
        % Surplus is > 5 kW: Battery charges at max, AE absorbs the rest
        zeta_a = 1;
        zeta_p = 0;

    elseif P_net >= -P_bat_max && P_net <= P_bat_max
        % Surplus/Deficit is within battery capability: Both H2 systems idle
        zeta_a = 0;
        zeta_p = 0;

    else % P_net < -P_bat_max
        % Deficit is > 5 kW: Battery discharges at max, FC supplies the rest
        zeta_a = 0;
        zeta_p = 1;
    end

end