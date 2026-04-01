% =========================================================================
% getMicrogridModel.m
% Complete Nonlinear State-Space Model of PV/Hydrogen DC Microgrid
%
% Assembles all subsystem state equations into one function of the form:
%   xdot = f(x, u, w)
% for use in MPC prediction, open-loop simulation, and validation.
%
% Reference: Zhu et al., Renewable Energy 222 (2024) 119871
%            Equations 8, 10, 13, 28
%
% State vector x (5x1):
%   x(1) = vpv   : PV capacitor voltage          [V]
%   x(2) = iL    : PV boost inductor current      [A]
%   x(3) = iae   : AE buck inductor/AE current    [A]
%   x(4) = ipe   : PEMFC boost inductor/FC current [A]
%   x(5) = Vdc   : DC bus voltage                 [V]
%
% Control vector u (3x1):
%   u(1) = Ss    : PV boost switch signal         {0,1}
%   u(2) = Sae   : AE buck switch signal          {0,1}
%   u(3) = Spe   : PEMFC boost switch signal      {0,1}
%
% Disturbance vector w (3x1):
%   w(1) = G     : Solar irradiance               [W/m^2]
%   w(2) = T     : PV cell temperature            [degC]
%   w(3) = Pload : Load power demand              [W]
%
% Outputs:
%   xdot  : State derivative vector (5x1)         [mixed units/s]
%   aux   : Auxiliary signals struct for monitoring and MPC cost function
%             .Ppv   : PV array power             [W]
%             .Pae   : AE power consumed          [W]
%             .Ppe   : PEMFC power output         [W]
%             .NH2   : H2 production rate (AE)    [mol/s]
%             .qH2   : H2 consumption rate (FC)   [mol/s]
%             .vae   : AE stack voltage            [V]
%             .vpe   : PEMFC stack voltage         [V]
%             .is    : PV converter output current [A]
%             .ia    : AE converter input current  [A]
%             .ip    : PEMFC converter output curr [A]
%             .il    : Load current                [A]
%             .Pbal  : Power balance error (gen-load) [W]
% =========================================================================

function [xdot, aux] = getMicrogridModel(x, u, w)

    % =====================================================================
    % UNPACK STATE VECTOR
    % =====================================================================
    vpv     = x(1);
    iL      = x(2);
    iae     = x(3);
    ipe     = x(4);
    Vdc     = x(5);

    % =====================================================================
    % UNPACK CONTROL VECTOR
    % =====================================================================
    Ss      = u(1);
    Sae     = u(2);
    Spe     = u(3);

    % =====================================================================
    % UNPACK DISTURBANCE VECTOR
    % =====================================================================
    G       = w(1);
    T       = w(2);
    Pload   = w(3);

    % =====================================================================
    % SUBSYSTEM 1: PV Boost Converter                            [Eq. 8]
    % States updated: x(1)=vpv, x(2)=iL
    % Also produces: is (current injected to DC bus), Ppv
    % =====================================================================
    [dvpv, diL, is, Ppv] = getPVBoostConverter(vpv, iL, Vdc, Ss, G, T);

    % =====================================================================
    % SUBSYSTEM 2: AE Buck Converter                             [Eq. 13]
    % State updated: x(3)=iae
    % Also produces: ia (current drawn from DC bus), vae, Pae, NH2
    % =====================================================================
    [diae, ia, vae, Pae, NH2] = getAEBuckConverter(iae, Vdc, Sae);

    % =====================================================================
    % SUBSYSTEM 3: PEMFC Boost Converter                         [Eq. 28]
    % State updated: x(4)=ipe
    % Also produces: ip (current injected to DC bus), vpe, Ppe, qH2
    % =====================================================================
    [dipe, ip, vpe, Ppe, qH2] = getPEMFCBoostConverter(ipe, Vdc, Spe);

    % =====================================================================
    % SUBSYSTEM 4: DC Bus                                        [Eq. 10]
    % State updated: x(5)=Vdc
    % Receives is, ip from generators and ia, il as sinks
    % =====================================================================
    [dVdc, il] = getDCBus(Vdc, is, ip, ia, Pload);

    % =====================================================================
    % ASSEMBLE STATE DERIVATIVE VECTOR
    % =====================================================================
    xdot    = [dvpv;    % dx(1)/dt
               diL;     % dx(2)/dt
               diae;    % dx(3)/dt
               dipe;    % dx(4)/dt
               dVdc];   % dx(5)/dt

    % =====================================================================
    % AUXILIARY OUTPUTS  (for MPC cost function and monitoring)
    % =====================================================================
    aux.Ppv     = Ppv;
    aux.Pae     = Pae;
    aux.Ppe     = Ppe;
    aux.NH2     = NH2;
    aux.qH2     = qH2;
    aux.vae     = vae;
    aux.vpe     = vpe;
    aux.is      = is;
    aux.ia      = ia;
    aux.ip      = ip;
    aux.il      = il;
    aux.Pbal    = (Ppv + Ppe) - (Pae + Pload);  % Power balance error [W]

end