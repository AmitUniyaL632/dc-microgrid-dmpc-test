% =========================================================================
% getDEMPC_BESS_GWO.m
% Local DMPC for Battery Energy Storage System (BESS) using GWO
% =========================================================================
function [db_opt, Pbat_comm, ib_comm] = getDEMPC_BESS_GWO(xb, Ppv_comm, Ppe_comm, Pae_comm, is_comm, ip_comm, ia_comm, Pload, Vdc_ref)

    % Extracted states
    ib  = xb(1);
    SOC = xb(2);
    Vdc = xb(3);

    % MPC Parameters
    Ts = 40e-6;
    Np = 3;

    % Cost function handle
    costFunc = @(U_seq) evaluateCostBESS(U_seq, ib, SOC, Vdc, Ppv_comm, Ppe_comm, Pae_comm, is_comm, ip_comm, ia_comm, Pload, Vdc_ref, Ts, Np);

    % GWO Parameters
    dim = Np;
    lowerBound = 0.1;
    upperBound = 0.9;
    maxIter = 20;   % Increased from 10
    wolfCount = 10; % Increased from 5

    % Run GWO
    [bestSeq, ~] = optimizeDutyCycleGWO(costFunc, dim, lowerBound, upperBound, maxIter, wolfCount);
    db_opt = bestSeq(1);

    % Communication outputs
    [Vbat, ~, Pbat_comm] = getBattery(ib, SOC);
    [~, ib_comm] = getBidirectionalConverter(ib, Vbat, Vdc, db_opt);

end

function J = evaluateCostBESS(U_seq, ib, SOC, Vdc, Ppv, Ppe, Pae, is, ip, ia, Pload, Vdc_ref, Ts, Np)
    w_v   = 20000;
    w_du  = 100;
    w_p   = 5000;
    w_lim = 50000;
    w_soc = 50000;
    Pbat_max = 5000; % 5 kW safety limit

    J = 0;
    db_last = U_seq(1);

    for k = 1:Np
        db = U_seq(k);
        [Vbat, dSOC, Pbat] = getBattery(ib, SOC);
        [dib, ib_bus] = getBidirectionalConverter(ib, Vbat, Vdc, db);
        [dVdc, ~] = getDCBus(Vdc, is, ip, ia, ib_bus, Pload);

        ib = ib + Ts * dib;
        SOC = SOC + Ts * dSOC;
        Vdc = Vdc + Ts * dVdc;

        Pbal = Ppv + Ppe + Pbat - Pae - Pload;
        Pbat_penalty = max(0, abs(Pbat) - Pbat_max);

        SOC_penalty = 0;
        if SOC >= 0.90 && Pbat < 0
            SOC_penalty = abs(Pbat); % Penalize charging when full
        elseif SOC <= 0.20 && Pbat > 0
            SOC_penalty = abs(Pbat); % Penalize discharging when empty
        end

        ls_v   = ((Vdc - Vdc_ref) / Vdc_ref)^2;
        ls_du  = (db - db_last)^2;
        ls_p   = (Pbal / max(Pload, 1))^2;
        ls_lim = (Pbat_penalty / Pbat_max)^2;
        ls_soc = (SOC_penalty / Pbat_max)^2;

        J = J + w_v * ls_v + w_du * ls_du + w_p * ls_p + w_lim * ls_lim + w_soc * ls_soc;
        db_last = db;
    end
end