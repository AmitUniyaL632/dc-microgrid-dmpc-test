% =========================================================================
% getDEMPC_PEMFCS.m
% Local Distributed Economic MPC for the PEMFC Subsystem
% Implements Equations 41-43 from Zhu et al., Renewable Energy 222 (2024)
%
% Inputs:
%   xp        : PEMFCS local states [ipe; Vdc]            (2x1)
%   d_prev    : Previous duty cycle                       [0-1]
%   Ppv_comm  : PV power from previous timestep           [W]  (communication)
%   Pae_comm  : AE power from previous timestep           [W]  (communication)
%   is_comm   : PV bus current from previous timestep     [A]  (communication)
%   ia_comm   : AE bus current from previous timestep     [A]  (communication)
%   Pload     : Load power demand                         [W]
%   Vdc_ref   : DC bus voltage reference this timestep    [V]
%
% Outputs:
%   d_pe_opt  : Optimal duty cycle                        [0-1]
%   Ppe_comm  : PEMFC power for communication to others   [W]
%   ip_comm   : PEMFC bus current for communication       [A]
% =========================================================================
function [d_pe_opt, Ppe_comm, ip_comm] = getDEMPC_PEMFCS(xp, d_prev, ...
    Ppv_comm, Pae_comm, is_comm, ia_comm, Pload, Vdc_ref)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;
    Np      = 2;
    
    % Weights normalized to match CMPC stability
    wp1     = 400;      % Weight: power balance (deficit tracking)
    wp2     = 20000;    % Weight: Vdc regulation

    % ---- Generate Dynamic Adaptive Grid ----
    d_pe_grid = max(0.05, min(0.95, d_prev + linspace(-0.045, 0.045, 7)'));
    n_dpe    = length(d_pe_grid);
    n_seq    = n_dpe^Np;

    seq_table = zeros(n_seq, Np);
    for j = 1:n_seq
        idx1 = mod(j-1, n_dpe) + 1;
        idx2 = floor((j-1) / n_dpe) + 1;
        seq_table(j,:) = [d_pe_grid(idx1), d_pe_grid(idx2)];
    end

    costs   = Inf(n_seq, 1);

    % ---- Evaluate cost for each candidate sequence ----
    for j = 1:n_seq
        xp_pred     = xp;
        total_cost  = 0;

        for l = 1:Np
            d_pe_l   = seq_table(j, l);

            % --- One-step forward prediction (local PEMFCS model) ---
            xp_pred = pemfcs_predict(xp_pred, d_pe_l, is_comm, ia_comm, Pload, Ts);

            % --- PEMFC power at predicted ipe ---
            ipe_l           = max(xp_pred(1), 1e-6);
            [~, Ppe_l, ~]   = getPEMFC(ipe_l);

            % --- Cost terms (Eq. 41) ---
            % Droop target for Deficit Power
            K_vdc    = 400; % [W/V] Proportional gain
            Ptarget  = max(Pload + Pae_comm - Ppv_comm - K_vdc * (xp(2) - Vdc_ref), 0);

            lp1 = ((Ppe_l - Ptarget) / max(Ptarget, 1))^2;
            lp2 = ((xp_pred(2) - Vdc_ref) / Vdc_ref)^2;

            total_cost  = total_cost + wp1*lp1 + wp2*lp2;
        end

        costs(j) = total_cost;
    end

    % ---- Select optimal sequence ----
    [~, j_opt]  = min(costs);
    d_pe_opt    = seq_table(j_opt, 1);

    % ---- Compute communication variables from current measured state ----
    ipe_now         = max(xp(1), 1e-6);
    [~, Ppe_comm, ~] = getPEMFC(ipe_now);
    ip_comm         = (1 - d_pe_opt) * max(xp(1), 0);

end

% =========================================================================
% Local prediction model for PEMFCS — Forward Euler (4 sub-steps)
% xp = [ipe; Vdc]
% Other subsystems' currents frozen at previous values (is_comm, ia_comm)
% =========================================================================
function xp_next = pemfcs_predict(xp, d_pe, is_comm, ia_comm, Pload, Ts)

    Lpe = 1e-3;         % FC inductance   [H]
    Cdc = 30e-3;        % Bus capacitance [F]

    Ts_sub = Ts / 4;
    xp_c   = xp;

    for sub = 1:4
        ipe = max(xp_c(1), 0);
        Vdc = max(xp_c(2), 1);

        [vpe, ~, ~]     = getPEMFC(max(ipe, 1e-6));
        il              = Pload / Vdc;
        ip              = (1 - d_pe) * ipe;

        dipe    = (vpe - (1 - d_pe) * Vdc) / Lpe;
        dVdc    = (is_comm + ip - ia_comm - il) / Cdc;

        xp_c        = xp_c + Ts_sub * [dipe; dVdc];
        xp_c(1)     = max(xp_c(1), 0);
        xp_c(2)     = max(xp_c(2), 1);
    end

    xp_next = xp_c;
end