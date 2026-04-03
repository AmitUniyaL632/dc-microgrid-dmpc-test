% =========================================================================
% getDEMPC_BESS.m
% Local DMPC for Battery Energy Storage System (BESS) using Grid Search
%
% Inputs:
%   xb        : State vector [ib; SOC; Vdc]
%   db_prev   : Previous duty cycle
%   Ppv_comm  : Communicated PV power
%   Ppe_comm  : Communicated PEMFC power
%   Pae_comm  : Communicated AE power
%   is_comm   : Communicated PV current
%   ip_comm   : Communicated PEMFC current
%   ia_comm   : Communicated AE current
%   Pload     : Load power demand
%   Vdc_ref   : Reference DC bus voltage
%
% Outputs:
%   db_opt    : Optimal duty cycle
%   Pbat_comm : Expected Battery power
%   ib_comm   : Expected Battery bus current
% =========================================================================
function [db_opt, Pbat_comm, ib_comm] = getDEMPC_BESS(xb, db_prev, Ppv_comm, Ppe_comm, Pae_comm, is_comm, ip_comm, ia_comm, Pload, Vdc_ref)

    % Extracted states
    ib  = xb(1);
    SOC = xb(2);
    Vdc = xb(3);

    % MPC Parameters
    Ts = 40e-6;
    Np = 3;
    db_set = linspace(0.1, 0.9, 11); % Limits duty cycle to avoid saturation
    Pbat_max = 5000; % 5 kW safety limit

    % Normalized Weights (Matching CMPC scale)
    w_v   = 20000; % Heavily weight Vdc regulation
    w_du  = 100;
    w_p   = 5000;
    w_lim = 50000; % High penalty for exceeding power limit
    w_soc = 50000; % High penalty for violating SOC limits

    J_min = inf;
    db_opt = db_prev;

    for i = 1:length(db_set)
        db = db_set(i);
        ib_pred = ib; SOC_pred = SOC; Vdc_pred = Vdc;
        J_seq = 0; db_last = db_prev;

        for k = 1:Np
            [Vbat, dSOC, Pbat] = getBattery(ib_pred, SOC_pred);
            [dib, ib_bus] = getBidirectionalConverter(ib_pred, Vbat, Vdc_pred, db);
            [dVdc, ~] = getDCBus(Vdc_pred, is_comm, ip_comm, ia_comm, ib_bus, Pload);

            ib_pred = ib_pred + Ts * dib;
            SOC_pred = SOC_pred + Ts * dSOC;
            Vdc_pred = Vdc_pred + Ts * dVdc;

            Pbal = Ppv_comm + Ppe_comm + Pbat - Pae_comm - Pload;
            Pbat_penalty = max(0, abs(Pbat) - Pbat_max);

            SOC_penalty = 0;
            if SOC_pred >= 0.90 && Pbat < 0
                SOC_penalty = abs(Pbat); % Penalize charging when full
            elseif SOC_pred <= 0.20 && Pbat > 0
                SOC_penalty = abs(Pbat); % Penalize discharging when empty
            end

            ls_v   = ((Vdc_pred - Vdc_ref) / Vdc_ref)^2;
            ls_du  = (db - db_last)^2;
            ls_p   = (Pbal / max(Pload, 1))^2;
            ls_lim = (Pbat_penalty / Pbat_max)^2;
            ls_soc = (SOC_penalty / Pbat_max)^2;

            J_seq = J_seq + w_v * ls_v ...
                          + w_du * ls_du ...
                          + w_p * ls_p ...
                          + w_lim * ls_lim ...
                          + w_soc * ls_soc;
            db_last = db;
        end

        if J_seq < J_min
            J_min = J_seq;
            db_opt = db;
        end
    end

    % Communication outputs
    [Vbat, ~, Pbat_comm] = getBattery(ib, SOC);
    [~, ib_comm] = getBidirectionalConverter(ib, Vbat, Vdc, db_opt);

end