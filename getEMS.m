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

    if Pmax > Pload
        % PV surplus: AE absorbs excess, FC idle
        zeta_a = 1;
        zeta_p = 0;

    elseif Pmax < Pload
        % PV deficit: FC supplies deficit, AE idle
        zeta_a = 0;
        zeta_p = 1;

    else
        % Perfect balance: both idle
        zeta_a = 0;
        zeta_p = 0;
    end

end