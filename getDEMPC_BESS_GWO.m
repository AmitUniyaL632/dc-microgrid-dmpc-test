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
    maxIter = 10;
    wolfCount = 5;

    % Run GWO
    [bestSeq, ~] = optimizeDutyCycleGWO(costFunc, dim, lowerBound, upperBound, maxIter, wolfCount);
    db_opt = bestSeq(1);

    % Communication outputs
    [Vbat, ~, Pbat_comm] = getBattery(ib, SOC);
    [~, ib_comm] = getBidirectionalConverter(ib, Vbat, Vdc, db_opt);

end

function J = evaluateCostBESS(U_seq, ib, SOC, Vdc, Ppv, Ppe, Pae, is, ip, ia, Pload, Vdc_ref, Ts, Np)
    w_v   = 1.0;
    w_du  = 0.1;
    w_p   = 0.5;
    w_lim = 10.0;
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
        Pbat_penalty = max(0, abs(Pbat) - Pbat_max)^2;

        J = J + w_v * (Vdc - Vdc_ref)^2 + w_du * (db - db_last)^2 + w_p * Pbal^2 + w_lim * Pbat_penalty;
        db_last = db;
    end
end