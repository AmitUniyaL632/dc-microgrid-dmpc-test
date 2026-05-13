% =========================================================================
% getDCBus.m
% DC Bus Voltage Dynamics
% Implements Equation 10 from Zhu et al., Renewable Energy 222 (2024) 119871
%
% Inputs:
%   Vdc   : DC bus voltage                    (V)   [state x5]
%   is    : PV converter output current       (A)   [from getPVBoostConverter]
%   ip    : PEMFC converter output current    (A)   [from getPEMFCBoostConverter]
%   ia    : AE converter input current        (A)   [from getAEBuckConverter]
%   ib_bus: Battery converter output current  (A)   [from getBidirectionalConverter]
%   Pload : Load power demand                 (W)   [disturbance]
%   iw    : WTG converter output current      (A)   [from getWTGBoostConverter]
%
% Output:
%   dVdc  : d(Vdc)/dt                         (V/s)
%   il    : Load current                      (A)
% =========================================================================

function [dVdc, il] = getDCBus(Vdc, is, ip, ia, varargin)

    if nargin == 5
        ib_bus = 0;
        Pload  = varargin{1};
        iw     = 0;
    elseif nargin == 6
        ib_bus = varargin{1};
        Pload  = varargin{2};
        iw     = 0;
    else
        ib_bus = varargin{1};
        Pload  = varargin{2};
        iw     = varargin{3};
    end

    % --- DC bus parameter (Table 4, Zhu et al.) ---
    Cdc     = 30e-3;    % DC bus capacitance          [F]

    % --- Load current (constant power load model) ---
    % il = Pload / Vdc
    Vdc_safe    = max(Vdc, 1);      % Prevent division by zero
    il          = Pload / Vdc_safe;

    % --- DC bus state equation (Eq. 10) ---
    % Cdc * dVdc/dt = is + ip + ib_bus + iw - ia - il
    dVdc    = (is + ip + ib_bus + iw - ia - il) / Cdc;

end
%%
% 
% ---
% 
% **How these four files connect:**
% ```
% G, T, vpv ──► getPVBoostConverter ──► dvpv, diL, is, Ppv ──┐
%                                                               │
% iae, Vdc ───► getAEBuckConverter  ──► diae, ia, vae, Pae ───┤──► getDCBus ──► dVdc
%                                                               │
% ipe, Vdc ───► getPEMFCBoostConverter ► dipe, ip, vpe, Ppe ──┘
