% =========================================================================
% getDEMPC_PVS_GWO.m
% Local DEMPC for the PV Subsystem using Grey Wolf Optimizer.
%
% Replaces the grid search from the original getDEMPC_PVS.m with a
% metaheuristic optimization to find the optimal duty cycle sequence.
% =========================================================================
function [d_s_opt, Ppv_comm, is_comm] = getDEMPC_PVS_GWO(xs, Pmax, ...
    Ppe_comm, Pae_comm, ip_comm, ia_comm, Pload, Vdc_ref, G, T)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;
    Np      = 2;
    ws1     = 0;
    ws2     = 800;
    ws3     = 20000;
    ws4     = 400;

    % ---- PV array config ----
    Ns_pv   = 5;
    Np_pv   = 20;
    Voc_mod = 47.6;
    vpv_max = Ns_pv * Voc_mod;

    % --- Inductor current reference ---
    V_sw        = linspace(0.1, Ns_pv * Voc_mod, 500);
    [~, Pp_sw]  = getPVArray(V_sw, G, T, Ns_pv, Np_pv);
    [~, mi]     = max(Pp_sw);
    Vmpp        = V_sw(mi);
    [Impp, ~]   = getPVArray(Vmpp, G, T, Ns_pv, Np_pv);
    iL_ref      = Impp;

    % ---- GWO parameters ----
    maxIter     = 10; % Keep low for speed
    wolfCount   = 8;  % Keep low for speed
    lowerBound  = 0.05;
    upperBound  = 0.65;

    % ---- Define Cost Function for GWO ----
    costFunction = @(d_seq) calculate_cost(d_seq);

    % ---- Run GWO to find optimal sequence ----
    [best_d_seq, ~] = optimizeDutyCycleGWO(costFunction, Np, lowerBound, upperBound, maxIter, wolfCount);
    d_s_opt = best_d_seq(1);

    % ---- Compute communication variables ----
    vpv_now         = max(xs(1), 0);
    [ipv_now, ~]    = getPVArray(vpv_now, G, T, Ns_pv, Np_pv);
    Ppv_comm        = vpv_now * ipv_now;
    is_comm         = (1 - d_s_opt) * max(xs(2), 0);

    % =====================================================================
    % NESTED COST FUNCTION
    % =====================================================================
    function total_cost = calculate_cost(d_s_sequence)
        xs_pred     = xs;
        total_cost  = 0;

        for l = 1:Np
            d_s_l   = d_s_sequence(l);

            % --- One-step forward prediction (local PVS model) ---
            xs_pred = pvs_predict(xs_pred, d_s_l, ip_comm, ia_comm, ...
                                  Pload, G, T, Ts, Ns_pv, Np_pv);

            % --- Constraint check ---
            if xs_pred(1) < 0 || xs_pred(1) > vpv_max
                total_cost = 1e9; % High penalty for infeasible solution
                return;
            end

            % --- PV power at predicted state ---
            vpv_l       = max(xs_pred(1), 0);
            [ipv_l, ~]  = getPVArray(vpv_l, G, T, Ns_pv, Np_pv);
            Ppv_l       = vpv_l * ipv_l;

            % --- Cost terms (Eq. 33) ---
            ls1 = 0;
            ls2 = ((Ppv_l - Pmax) / max(Pmax, 1))^2;
            ls3 = ((xs_pred(3) - Vdc_ref) / Vdc_ref)^2;
            ls4 = ((xs_pred(2) - iL_ref) / max(iL_ref, 1))^2;

            total_cost  = total_cost + ws1*ls1 + ws2*ls2 + ws3*ls3 + ws4*ls4;
        end
    end
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