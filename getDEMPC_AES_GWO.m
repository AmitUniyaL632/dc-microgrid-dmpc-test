% =========================================================================
% getDEMPC_AES_GWO.m
% Local DEMPC for the AE Subsystem using Grey Wolf Optimizer.
% =========================================================================
function [d_ae_opt, Pae_comm, ia_comm, NH2_comm] = getDEMPC_AES_GWO(xa, ...
    Ppv_comm, Ppe_comm, is_comm, ip_comm, Pload, Vdc_ref)

    % ---- DEMPC parameters ----
    Ts      = 40e-6;
    Np      = 2;
    wa1     = 400;
    wa2     = 20000;

    % ---- GWO parameters ----
    maxIter     = 10;
    wolfCount   = 8;
    lowerBound  = 0.05;
    upperBound  = 0.65;

    % ---- Define Cost Function for GWO ----
    costFunction = @(d_seq) calculate_cost(d_seq);

    % ---- Run GWO to find optimal sequence ----
    [best_d_seq, ~] = optimizeDutyCycleGWO(costFunction, Np, lowerBound, upperBound, maxIter, wolfCount);
    d_ae_opt = best_d_seq(1);

    % ---- Compute communication variables ----
    iae_now             = max(xa(1), 1e-6);
    [~, Pae_comm, NH2_comm, ~] = getAE(iae_now);
    ia_comm             = d_ae_opt * max(xa(1), 0);

    % =====================================================================
    % NESTED COST FUNCTION
    % =====================================================================
    function total_cost = calculate_cost(d_ae_sequence)
        xa_pred     = xa;
        total_cost  = 0;

        for l = 1:Np
            d_ae_l   = d_ae_sequence(l);

            % --- One-step forward prediction ---
            xa_pred = aes_predict(xa_pred, d_ae_l, is_comm, ip_comm, Pload, Ts);

            % --- AE power at predicted iae ---
            iae_l           = max(xa_pred(1), 1e-6);
            [~, Pae_l, ~, ~] = getAE(iae_l);

            % --- Cost terms (Eq. 38) ---
            K_vdc    = 400;
            Ptarget  = max(Ppv_comm + Ppe_comm - Pload + K_vdc * (xa(2) - Vdc_ref), 0);

            la1 = ((Pae_l - Ptarget) / max(Ptarget, 1))^2;
            la2 = ((xa_pred(2) - Vdc_ref) / Vdc_ref)^2;

            total_cost  = total_cost + wa1*la1 + wa2*la2;
        end
    end
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