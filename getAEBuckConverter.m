% =========================================================================
% getAEBuckConverter.m
% Alkaline Electrolyzer Unidirectional Buck Converter State Equations
% Implements Equation 13 from Zhu et al., Renewable Energy 222 (2024) 119871
%
% Inputs:
%   iae   : AE inductor/AE current            (A)   [state x3]
%   Vdc   : DC bus voltage                    (V)   [state x5]
%   Sae   : Switch signal                     {0,1} [control input]
%
% Outputs:
%   diae  : d(iae)/dt                         (A/s)
%   ia    : Current drawn from DC bus         (A)
%   vae   : AE stack voltage                  (V)
%   Pae   : AE power consumed                 (W)
%   NH2   : Hydrogen production rate          (mol/s)
% =========================================================================

function [diae, ia, vae, Pae, NH2] = getAEBuckConverter(iae, Vdc, Sae)

    % --- Converter parameter (Table 4, Zhu et al.) ---
    Lae     = 01e-3;     % Buck converter inductance   [H]

    % --- AE stack voltage: nonlinear coupling (Eq. 15) ---
    % Clamp current positive; getAE handles its own internal guard
    iae_safe            = max(iae, 0);
    [vae, Pae, NH2, ~]  = getAE(iae_safe);

    % --- State equation (Eq. 13) ---
    % Lae * diae/dt = Sae * Vdc - vae(iae)
    diae    = (Sae * Vdc - vae) / Lae;

    % --- Current drawn from DC bus ---
    % ia = Sae * iae
    ia      = Sae * iae_safe;

end