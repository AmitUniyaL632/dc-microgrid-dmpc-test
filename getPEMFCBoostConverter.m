% =========================================================================
% getPEMFCBoostConverter.m
% PEMFC Unidirectional Boost Converter State Equations
% Implements Equation 28 from Zhu et al., Renewable Energy 222 (2024) 119871
%
% Inputs:
%   ipe   : PEMFC inductor/FC current         (A)   [state x4]
%   Vdc   : DC bus voltage                    (V)   [state x5]
%   Spe   : Switch signal                     {0,1} [control input]
%
% Outputs:
%   dipe  : d(ipe)/dt                         (A/s)
%   ip    : Output current injected to DC bus (A)
%   vpe   : PEMFC stack voltage               (V)
%   Ppe   : PEMFC output power                (W)
%   qH2   : Hydrogen consumption rate         (mol/s)
% =========================================================================

function [dipe, ip, vpe, Ppe, qH2] = getPEMFCBoostConverter(ipe, Vdc, Spe)

    % --- Converter parameter (Table 4, Zhu et al.) ---
    Lpe     = 01e-3;     % Boost converter inductance  [H]

    % --- PEMFC stack voltage: nonlinear coupling (Eq. 20-26) ---
    % Clamp current positive; getPEMFC handles its own internal guard
    ipe_safe        = max(ipe, 0);
    [vpe, Ppe, qH2] = getPEMFC(ipe_safe);

    % --- State equation (Eq. 28) ---
    % Lpe * dipe/dt = vpe(ipe) - (1 - Spe) * Vdc
    dipe    = (vpe - (1 - Spe) * Vdc) / Lpe;

    % --- Output current injected to DC bus ---
    % ip = (1 - Spe) * ipe
    ip      = (1 - Spe) * ipe_safe;

end