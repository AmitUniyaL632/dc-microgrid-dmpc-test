% =========================================================================
% getDEMPC_WTG.m
% Local Distributed Economic MPC for the Wind Turbine Generator (WTG)
%
% MINLP: Enumerates all 2^Np switching sequences, selects minimum cost.
%
% Inputs:
%   xw        : WTG local states [vwt; iLw; Vdc]          (3x1)
%   d_prev    : Previous duty cycle                       [0-1]
%   Pmax      : WTG maximum power at current conditions   [W]
%   Ppv_comm  : PV power from previous timestep           [W]  (communication)
%   Ppe_comm  : PEMFC power from previous timestep        [W]  (communication)
%   Pae_comm  : AE power from previous timestep           [W]  (communication)
%   is_comm   : PV bus current from previous timestep     [A]  (communication)
%   ip_comm   : PEMFC bus current from previous timestep  [A]  (communication)
%   ia_comm   : AE bus current from previous timestep     [A]  (communication)
%   Pload     : Load power demand                         [W]
%   Vdc_ref   : DC bus voltage reference this timestep    [V]
%   v_w       : Wind speed                                [m/s]
%
% Outputs:
%   d_w_opt   : Optimal duty cycle                        [0-1]
%   Pwt_comm  : WTG power for communication to others     [W]
%   iw_comm   : WTG bus output current for communication  [A]
% =========================================================================
function [d_w_opt, Pwt_comm, iw_comm] = getDEMPC_WTG(xw, d_prev, Pmax, ...
    Ppv_comm, Ppe_comm, Pae_comm, is_comm, ip_comm, ia_comm, Pload, Vdc_ref, v_w)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;                % Sampling interval           [s]
    Np      = 2;                    % Prediction horizon          [-]
    
    % Weights normalized
    ws1     = 0;                    % Power balance
    ws2     = 800;                  % Weight: MPPT
    ws3     = 20000;                % Weight: Vdc regulation
    ws4     = 400;                  % Weight: Inductor current tracking

    % ---- Inductor current reference for WTG ----
    % Approximate MPPT reference current.
    % We assume the WTG produces Pmax, so Impp ~ Pmax / Vmpp
    % Since WTG is current source at MPP, optimal current is Pmax / vwt_optimal.
    % We approximate it as Pmax / (0.5 * Vdc_ref) or use current state as base.
    vwt_safe = max(xw(1), 1.0);
    [iwt_mpp, Pwt_mpp] = getWindTurbine(vwt_safe, v_w);
    iL_ref = iwt_mpp; % Simply use current output from WTG as reference

    vwt_max = 800; % Max WTG voltage constraint [V]

    % ---- Generate Dynamic Adaptive Grid ----
    d_w_grid = max(0.05, min(0.65, d_prev + linspace(-0.045, 0.045, 7)'));
    n_dw     = length(d_w_grid);
    n_seq    = n_dw^Np;

    seq_table = zeros(n_seq, Np);
    for j = 1:n_seq
        idx1 = mod(j-1, n_dw) + 1;
        idx2 = floor((j-1) / n_dw) + 1;
        seq_table(j,:) = [d_w_grid(idx1), d_w_grid(idx2)];
    end

    costs   = Inf(n_seq, 1);

    % ---- Evaluate cost for each candidate sequence ----
    for j = 1:n_seq
        xw_pred     = xw;
        total_cost  = 0;
        feasible    = true;

        for l = 1:Np
            d_w_l   = seq_table(j, l);

            % --- One-step forward prediction (local WTG model) ---
            xw_pred = wtg_predict(xw_pred, d_w_l, is_comm, ip_comm, ia_comm, ...
                                  Pload, v_w, Ts);

            % --- Constraint check ---
            if xw_pred(1) < 0 || xw_pred(1) > vwt_max
                feasible = false;
                break;
            end

            % --- WTG power at predicted state ---
            vwt_l       = max(xw_pred(1), 1.0);
            [iwt_l, Pwt_l]  = getWindTurbine(vwt_l, v_w);

            % --- Cost terms ---
            ls1 = 0; % WTG does not explicitly load-follow
            ls2 = ((Pwt_l - Pmax) / max(Pmax, 1))^2;
            ls3 = ((xw_pred(3) - Vdc_ref) / Vdc_ref)^2;
            ls4 = ((xw_pred(2) - iL_ref) / max(iL_ref, 1))^2;

            total_cost  = total_cost + ws1*ls1 + ws2*ls2 + ws3*ls3 + ws4*ls4;
        end

        if feasible
            costs(j) = total_cost;
        end
    end

    % ---- Select optimal sequence ----
    [~, j_opt]  = min(costs);
    d_w_opt     = seq_table(j_opt, 1);

    % ---- Compute communication variables from current measured state ----
    vwt_now         = max(xw(1), 1.0);
    [iwt_now, Pwt_now] = getWindTurbine(vwt_now, v_w);
    Pwt_comm        = Pwt_now;
    iw_comm         = (1 - d_w_opt) * max(xw(2), 0);

end

% =========================================================================
% Local prediction model for WTG — Forward Euler (4 sub-steps)
% xw = [vwt; iLw; Vdc]
% =========================================================================
function xw_next = wtg_predict(xw, d_w, is_comm, ip_comm, ia_comm, Pload, v_w, Ts)

    Cw  = 2000e-6;      % WTG capacitance [F]
    Lw  = 1e-3;         % WTG inductance  [H]
    Cdc = 30e-3;        % Bus capacitance [F]
    
    Ts_sub = Ts / 4;
    xw_c   = xw;

    for sub = 1:4
        vwt = max(xw_c(1), 1.0);
        iLw = max(xw_c(2), 0);
        Vdc = max(xw_c(3), 1.0);

        [iwt, ~]    = getWindTurbine(vwt, v_w);
        il          = Pload / Vdc;
        iw          = (1 - d_w) * iLw;

        dvwt    = (iwt - iLw) / Cw;
        diLw    = (vwt - (1 - d_w) * Vdc) / Lw;
        dVdc    = (iw + is_comm + ip_comm - ia_comm - il) / Cdc;

        xw_c        = xw_c + Ts_sub * [dvwt; diLw; dVdc];
        xw_c(1)     = max(xw_c(1), 0);
        xw_c(2)     = max(xw_c(2), 0);
        xw_c(3)     = max(xw_c(3), 1.0);
    end
    
    xw_next = xw_c;
end
