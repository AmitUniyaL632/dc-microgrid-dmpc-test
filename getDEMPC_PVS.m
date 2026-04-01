% =========================================================================
% getDEMPC_PVS.m
% Local Distributed Economic MPC for the PV Subsystem (PVS)
% Implements Equations 33-37 from Zhu et al., Renewable Energy 222 (2024)
%
% MINLP: Enumerates all 2^Np switching sequences, selects minimum cost.
%
% Inputs:
%   xs        : PVS local states [vpv; iL; Vdc]           (3x1)
%   d_prev    : Previous duty cycle                       [0-1]
%   Pmax      : PV maximum power at current conditions    [W]
%   Ppe_comm  : PEMFC power from previous timestep        [W]  (communication)
%   Pae_comm  : AE power from previous timestep           [W]  (communication)
%   ip_comm   : PEMFC bus current from previous timestep  [A]  (communication)
%   ia_comm   : AE bus current from previous timestep     [A]  (communication)
%   Pload     : Load power demand                         [W]
%   Vdc_ref   : DC bus voltage reference this timestep    [V]  (Eq. 34)
%   G         : Solar irradiance                          [W/m^2]
%   T         : PV cell temperature                       [degC]
%
% Outputs:
%   d_s_opt   : Optimal duty cycle                        [0-1]
%   Ppv_comm  : PV power for communication to others      [W]
%   is_comm   : PV bus output current for communication   [A]
% =========================================================================
function [d_s_opt, Ppv_comm, is_comm] = getDEMPC_PVS(xs, d_prev, Pmax, ...
    Ppe_comm, Pae_comm, ip_comm, ia_comm, Pload, Vdc_ref, G, T)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;                % Sampling interval           [s]
    Np      = 2;                    % Prediction horizon          [-]
    
    % Weights normalized to match CMPC stability
    ws1     = 0;                    % Power balance (delegated entirely to AE/FC)
    ws2     = 800;                  % Weight: MPPT
    ws3     = 20000;                % Weight: Vdc regulation
    ws4     = 400;                  % Weight: Inductor current tracking

    % ---- PV array config ----
    Ns_pv   = 5;                    % Series modules
    Np_pv   = 20;                   % Parallel strings
    Voc_mod = 47.6;                 % Module open-circuit voltage [V]
    vpv_max = Ns_pv * Voc_mod;     % Max PV voltage constraint   [V] (Eq. 36e)

    % --- Inductor current reference ---
    V_sw        = linspace(0.1, Ns_pv * Voc_mod, 500);
    [~, Pp_sw]  = getPVArray(V_sw, G, T, Ns_pv, Np_pv);
    [~, mi]     = max(Pp_sw);
    Vmpp        = V_sw(mi);
    [Impp, ~]   = getPVArray(Vmpp, G, T, Ns_pv, Np_pv);
    iL_ref      = Impp;

    % ---- Generate Dynamic Adaptive Grid ----
    d_s_grid = max(0.05, min(0.65, d_prev + linspace(-0.045, 0.045, 7)'));
    n_ds     = length(d_s_grid);
    n_seq    = n_ds^Np;             % 7^2 = 49 combinations

    seq_table = zeros(n_seq, Np);
    for j = 1:n_seq
        idx1 = mod(j-1, n_ds) + 1;
        idx2 = floor((j-1) / n_ds) + 1;
        seq_table(j,:) = [d_s_grid(idx1), d_s_grid(idx2)];
    end

    costs   = Inf(n_seq, 1);

    % ---- Evaluate cost for each candidate sequence ----
    for j = 1:n_seq
        xs_pred     = xs;
        total_cost  = 0;
        feasible    = true;

        for l = 1:Np
            d_s_l   = seq_table(j, l);

            % --- One-step forward prediction (local PVS model) ---
            xs_pred = pvs_predict(xs_pred, d_s_l, ip_comm, ia_comm, ...
                                  Pload, G, T, Ts, Ns_pv, Np_pv);

            % --- Constraint check: 0 <= vpv <= vpv_max  (Gamma_cs, Eq. 37) ---
            if xs_pred(1) < 0 || xs_pred(1) > vpv_max
                feasible = false;
                break;
            end

            % --- PV power at predicted state ---
            vpv_l       = max(xs_pred(1), 0);
            [ipv_l, ~]  = getPVArray(vpv_l, G, T, Ns_pv, Np_pv);
            Ppv_l       = vpv_l * ipv_l;

            % --- Cost terms (Eq. 33) ---
            % Normalized identically to CMPC
            ls1 = 0; % PV does not explicitly load-follow
            ls2 = ((Ppv_l - Pmax) / max(Pmax, 1))^2;
            ls3 = ((xs_pred(3) - Vdc_ref) / Vdc_ref)^2;
            ls4 = ((xs_pred(2) - iL_ref) / max(iL_ref, 1))^2;

            total_cost  = total_cost + ws1*ls1 + ws2*ls2 + ws3*ls3 + ws4*ls4;
        end

        if feasible
            costs(j) = total_cost;
        end
    end

    % ---- Select optimal sequence (Algorithm 1, Step 3) ----
    [~, j_opt]  = min(costs);
    d_s_opt     = seq_table(j_opt, 1);

    % ---- Compute communication variables from current measured state ----
    vpv_now         = max(xs(1), 0);
    [ipv_now, ~]    = getPVArray(vpv_now, G, T, Ns_pv, Np_pv);
    Ppv_comm        = vpv_now * ipv_now;
    is_comm         = (1 - d_s_opt) * max(xs(2), 0);

end

% =========================================================================
% Local prediction model for PVS — Forward Euler (4 sub-steps)
% xs = [vpv; iL; Vdc]
% Other subsystems' currents frozen at previous values (ip_comm, ia_comm)
% =========================================================================
function xs_next = pvs_predict(xs, d_s, ip_comm, ia_comm, Pload, G, T, Ts, Ns_pv, Np_pv)

    Cs  = 2000e-6;      % PV capacitance  [F]
    Ls  = 1e-3;         % PV inductance   [H]
    Cdc = 30e-3;        % Bus capacitance [F]
    
    Ts_sub = Ts / 4;
    xs_c   = xs;

    for sub = 1:4
        vpv = max(xs_c(1), 0);
        iL  = max(xs_c(2), 0);
        Vdc = max(xs_c(3), 1);

        [ipv, ~]    = getPVArray(vpv, G, T, Ns_pv, Np_pv);
        il          = Pload / Vdc;
        is          = (1 - d_s) * iL;

        dvpv    = (ipv - iL) / Cs;
        diL     = (vpv - (1 - d_s) * Vdc) / Ls;
        dVdc    = (is + ip_comm - ia_comm - il) / Cdc;

        xs_c        = xs_c + Ts_sub * [dvpv; diL; dVdc];
        xs_c(1)     = max(xs_c(1), 0);
        xs_c(2)     = max(xs_c(2), 0);
        xs_c(3)     = max(xs_c(3), 1);
    end
    
    xs_next = xs_c;
end