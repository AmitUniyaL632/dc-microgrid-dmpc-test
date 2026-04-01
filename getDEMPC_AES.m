% =========================================================================
% getDEMPC_AES.m
% Local Distributed Economic MPC for the Alkaline Electrolyzer Subsystem
% Implements Equations 38-40 from Zhu et al., Renewable Energy 222 (2024)
%
% Inputs:
%   xa        : AES local states [iae; Vdc]               (2x1)
%   d_prev    : Previous duty cycle                       [0-1]
%   Ppv_comm  : PV power from previous timestep           [W]  (communication)
%   Ppe_comm  : PEMFC power from previous timestep        [W]  (communication)
%   is_comm   : PV bus current from previous timestep     [A]  (communication)
%   ip_comm   : PEMFC bus current from previous timestep  [A]  (communication)
%   Pload     : Load power demand                         [W]
%   Vdc_ref   : DC bus voltage reference this timestep    [V]
%
% Outputs:
%   d_ae_opt  : Optimal duty cycle                        [0-1]
%   Pae_comm  : AE power for communication to others      [W]
%   ia_comm   : AE bus input current for communication    [A]
%   NH2_comm  : H2 production rate for monitoring         [mol/s]
% =========================================================================
function [d_ae_opt, Pae_comm, ia_comm, NH2_comm] = getDEMPC_AES(xa, d_prev, ...
    Ppv_comm, Ppe_comm, is_comm, ip_comm, Pload, Vdc_ref)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;
    Np      = 2;
    
    % Weights normalized to match CMPC stability
    wa1     = 400;      % Weight: power balance (surplus tracking)
    wa2     = 20000;    % Weight: Vdc regulation

    % ---- Generate Dynamic Adaptive Grid ----
    d_ae_grid = max(0.05, min(0.65, d_prev + linspace(-0.045, 0.045, 7)'));
    n_dae    = length(d_ae_grid);
    n_seq    = n_dae^Np;             % 7^2 = 49 combinations

    seq_table = zeros(n_seq, Np);
    for j = 1:n_seq
        idx1 = mod(j-1, n_dae) + 1;
        idx2 = floor((j-1) / n_dae) + 1;
        seq_table(j,:) = [d_ae_grid(idx1), d_ae_grid(idx2)];
    end

    costs   = Inf(n_seq, 1);

    % ---- Evaluate cost for each candidate sequence ----
    for j = 1:n_seq
        xa_pred     = xa;
        total_cost  = 0;

        for l = 1:Np
            d_ae_l   = seq_table(j, l);

            % --- One-step forward prediction (local AES model) ---
            xa_pred = aes_predict(xa_pred, d_ae_l, is_comm, ip_comm, Pload, Ts);

            % --- AE power and H2 rate at predicted iae ---
            iae_l           = max(xa_pred(1), 1e-6);
            [~, Pae_l, NH2_l, ~] = getAE(iae_l);

            % --- Cost terms (Eq. 38) ---
            % Droop target for Surplus Power
            K_vdc    = 400; % [W/V] Proportional gain
            Ptarget  = max(Ppv_comm + Ppe_comm - Pload + K_vdc * (xa(2) - Vdc_ref), 0);

            la1 = ((Pae_l - Ptarget) / max(Ptarget, 1))^2;
            la2 = ((xa_pred(2) - Vdc_ref) / Vdc_ref)^2;

            total_cost  = total_cost + wa1*la1 + wa2*la2;
        end

        costs(j) = total_cost;
    end

    % ---- Select optimal sequence ----
    [~, j_opt]  = min(costs);
    d_ae_opt    = seq_table(j_opt, 1);

    % ---- Compute communication variables from current measured state ----
    iae_now             = max(xa(1), 1e-6);
    [~, Pae_comm, NH2_comm, ~] = getAE(iae_now);
    ia_comm             = d_ae_opt * max(xa(1), 0);

end

% =========================================================================
% Local prediction model for AES — Forward Euler  (Eq. 13 + Eq. 10)
% xa = [iae; Vdc]
% Other subsystems' currents frozen at previous values (is_comm, ip_comm)
% =========================================================================
function xa_next = aes_predict(xa, d_ae, is_comm, ip_comm, Pload, Ts)

    Lae = 1e-3;         % AE inductance   [H]
    Cdc = 30e-3;        % Bus capacitance [F]

    Ts_sub = Ts / 4;
    xa_c   = xa;

    for sub = 1:4
        iae = max(xa_c(1), 0);
        Vdc = max(xa_c(2), 1);

        [vae, ~, ~, ~]  = getAE(max(iae, 1e-6));
        il              = Pload / Vdc;
        ia              = d_ae * iae;

        diae    = (d_ae * Vdc - vae) / Lae;
        dVdc    = (is_comm + ip_comm - ia - il) / Cdc;

        xa_c        = xa_c + Ts_sub * [diae; dVdc];
        xa_c(1)     = max(xa_c(1), 0);
        xa_c(2)     = max(xa_c(2), 1);
    end

    xa_next = xa_c;
end