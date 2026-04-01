% =========================================================================
% getDEMPC_PEMFCS_GWO.m
% Local DEMPC for the PEMFC Subsystem using Grey Wolf Optimizer.
% =========================================================================
function [d_pe_opt, Ppe_comm, ip_comm] = getDEMPC_PEMFCS_GWO(xp, ...
    Ppv_comm, Pae_comm, is_comm, ia_comm, Pload, Vdc_ref)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;
    Np      = 2;
    wp1     = 400;
    wp2     = 20000;

    % ---- GWO parameters ----
    maxIter     = 10;
    wolfCount   = 8;
    lowerBound  = 0.05;
    upperBound  = 0.95;

    % ---- Define Cost Function for GWO ----
    costFunction = @(d_seq) calculate_cost(d_seq);

    % ---- Run GWO to find optimal sequence ----
    [best_d_seq, ~] = optimizeDutyCycleGWO(costFunction, Np, lowerBound, upperBound, maxIter, wolfCount);
    d_pe_opt = best_d_seq(1);

    % ---- Compute communication variables ----
    ipe_now         = max(xp(1), 1e-6);
    [~, Ppe_comm, ~] = getPEMFC(ipe_now);
    ip_comm         = (1 - d_pe_opt) * max(xp(1), 0);

    % =====================================================================
    % NESTED COST FUNCTION
    % =====================================================================
    function total_cost = calculate_cost(d_pe_sequence)
        xp_pred     = xp;
        total_cost  = 0;

        for l = 1:Np
            d_pe_l   = d_pe_sequence(l);

            % --- One-step forward prediction ---
            xp_pred = pemfcs_predict(xp_pred, d_pe_l, is_comm, ia_comm, Pload, Ts);

            % --- PEMFC power at predicted ipe ---
            ipe_l           = max(xp_pred(1), 1e-6);
            [~, Ppe_l, ~]   = getPEMFC(ipe_l);

            % --- Cost terms (Eq. 41) ---
            K_vdc    = 400;
            Ptarget  = max(Pload + Pae_comm - Ppv_comm - K_vdc * (xp(2) - Vdc_ref), 0);

            lp1 = ((Ppe_l - Ptarget) / max(Ptarget, 1))^2;
            lp2 = ((xp_pred(2) - Vdc_ref) / Vdc_ref)^2;

            total_cost  = total_cost + wp1*lp1 + wp2*lp2;
        end
    end
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