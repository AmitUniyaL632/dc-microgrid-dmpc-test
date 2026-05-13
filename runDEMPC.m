% =========================================================================
% runDEMPC.m
% Main DEMPC Simulation Loop for PV/Hydrogen DC Microgrid
%
% Coordinates three local controllers with EMS, following the DEMPC
% application procedure in Section 3.4, Zhu et al. (2024)
%
% Inputs:
%   x_init        : Initial state [vpv; iL; iae; ipe; Vdc; ib; SOC; vwt; iLw]   (5x1)
%   t_sim         : Total simulation time                     [s]
%   G_profile     : Function handle — irradiance G(t)         [W/m^2]
%   T_profile     : Function handle — temperature T(t)        [degC]
%   Pload_profile : Function handle — load power Pload(t)     [W]
%
% Output:
%   results : Struct containing all logged signals
% =========================================================================
function results = runDEMPC(x_init, t_sim, G_profile, T_profile, Pload_profile, v_w_profile)

    % ---- Simulation parameters ----
    Ts      = 40e-6;                    % Sampling interval     [s]   (Table 3)
    Ns_ref  = 10;                       % Reference horizon     [-]   (Table 3)
    Vdc_set = 300;                      % DC bus setpoint       [V]   (Table 4)
    N_steps = round(t_sim / Ts);        % Total simulation steps

    % ---- PV array config (for Pmax sweep) ----
    Ns_pv   = 5;
    Np_pv   = 20;
    Voc_mod = 47.6;
    V_sweep = linspace(0.1, Ns_pv * Voc_mod, 500);  % 500-pt MPP sweep

    % =====================================================================
    % INITIALIZE STATE AND COMMUNICATION VARIABLES
    % =====================================================================
    Vdc_ref = Vdc_set;
    x       = x_init;

    d_prev  = [0.35; 0.22; 0.30; 0.5; 0.5]; % Adaptive grid initial guesses

    % Initialize communication variables from initial conditions
    G0  = G_profile(0);   T0 = T_profile(0);   v_w0 = v_w_profile(0);
    vpv0 = x(1);   iL0 = x(2);   iae0 = x(3);   ipe0 = x(4);
    Vdc0 = x(5);   ib0 = x(6);   SOC0 = x(7);   vwt0 = x(8);   iLw0 = x(9);

    [ipv0, ~]           = getPVArray(max(vpv0,0), G0, T0, Ns_pv, Np_pv);
    Ppv_comm            = max(vpv0, 0) * ipv0;
    is_comm             = (1 - d_prev(1))  * max(iL0,  0);
    [~, Pae_comm, NH2_init, ~] = getAE(max(iae0, 1e-6));
    ia_comm             = d_prev(2) * max(iae0, 0);
    [~, Ppe_comm, ~]    = getPEMFC(max(ipe0, 1e-6));
    ip_comm             = (1 - d_prev(3)) * max(ipe0, 0);
    
    [Vbat0, ~, Pbat_comm] = getBattery(ib0, SOC0);
    [~, ib_comm]          = getBidirectionalConverter(ib0, Vbat0, Vdc0, d_prev(4));
    
    [iwt0, Pwt_now]       = getWindTurbine(max(vwt0, 1.0), v_w0);
    Pwt_comm              = Pwt_now;
    iw_comm               = (1 - d_prev(5)) * max(iLw0, 0);

    % ---- Initialize Hydrogen Tank ----
    V_tank   = 0.1;                                 % Volume [m^3]
    T_tank   = 298.15;                              % Temp [K]
    R_gas    = 8.31446;                             % Gas constant
    P_tank0  = 10 * 1e5;                            % Init pressure (10 bar)
    n_H2     = P_tank0 * V_tank / (R_gas * T_tank); % Init moles

    % =====================================================================
    % PRE-ALLOCATE RESULTS
    % =====================================================================
    t_vec       = (0:N_steps-1)' * Ts;
    X           = zeros(N_steps, 9);
    U           = zeros(N_steps, 5);
    Ppv_log     = zeros(N_steps, 1);
    Pwt_log     = zeros(N_steps, 1);
    Pae_log     = zeros(N_steps, 1);
    Ppe_log     = zeros(N_steps, 1);
    Pmax_log    = zeros(N_steps, 1);
    Pbal_log    = zeros(N_steps, 1);
    NH2_log     = zeros(N_steps, 1);
    qH2_log     = zeros(N_steps, 1);
    Vdc_ref_log = zeros(N_steps, 1);
    zeta_log    = zeros(N_steps, 2);    % [zeta_a, zeta_p]
    Ptank_log   = zeros(N_steps, 1);
    Pbat_log    = zeros(N_steps, 1);
    SOC_log     = zeros(N_steps, 1);

    fprintf('Running DEMPC: %d steps (Ts=%.0f µs, t_sim=%.3f s)...\n', ...
            N_steps, Ts*1e6, t_sim);
    report_every = max(1, round(N_steps/20));

    % =====================================================================
    % MAIN SIMULATION LOOP  (Section 3.4, Zhu et al.)
    % =====================================================================
    for k = 1:N_steps

        t_now   = (k-1) * Ts;

        % --- Current disturbances ---
        G_now   = G_profile(t_now);
        T_now   = T_profile(t_now);
        Pload   = Pload_profile(t_now);
        v_w_now = v_w_profile(t_now);

        % --- Unpack current states ---
        vpv = x(1);   iL  = x(2);
        iae = x(3);   ipe = x(4);
        Vdc = x(5);   ib  = x(6);
        SOC = x(7);   vwt = x(8);   iLw = x(9);

        % =================================================================
        % STEP 1: Compute PV and WTG maximum power Pmax
        % =================================================================
        [~, Pp_sw]  = getPVArray(V_sweep, G_now, T_now, Ns_pv, Np_pv);
        Pmax_pv     = max(Pp_sw);
        [~, Pwt_max]= getWindTurbine(400, v_w_now); % Approx Pmax using arbitrary nominal voltage
        Pmax_tot    = Pmax_pv + Pwt_max;

        % =================================================================
        % STEP 2: EMS — determine operational modes  (Eq. 32)
        % =================================================================
        [zeta_a, zeta_p] = getEMS(Pmax_tot, Pload, SOC);

        % =================================================================
        % STEP 3: Vdc reference — gradual approach  (Eq. 34)
        % =================================================================
        Vdc_ref = Vdc_ref + (1/Ns_ref) * (Vdc_set - Vdc_ref);
        % Vdc = Vdc_ref + (1/Ns_ref) * (Vdc_set - Vdc);

        % =================================================================
        % STEP 4: Local DEMPC for PVS  (always active)
        % =================================================================
        xs  = [vpv; iL; Vdc];
        Pload_eff = Pload - Pbat_comm; % Effective load for legacy controllers
        [d_s_opt, Ppv_comm, is_comm] = getDEMPC_PVS(xs, d_prev(1), Pmax_pv, ...
            Ppe_comm, Pae_comm, ip_comm, ia_comm, Pload_eff, Vdc_ref, G_now, T_now);

        % =================================================================
        % STEP 4.5: Local DEMPC for WTG (always active)
        % =================================================================
        xw  = [vwt; iLw; Vdc];
        [d_w_opt, Pwt_comm, iw_comm] = getDEMPC_WTG(xw, d_prev(5), Pwt_max, ...
            Ppv_comm, Ppe_comm, Pae_comm, is_comm, ip_comm, ia_comm, Pload_eff, Vdc_ref, v_w_now);

        % =================================================================
        % STEP 5: Local DEMPC for BESS (always active)
        % =================================================================
        xb  = [ib; SOC; Vdc];
        Pae_comm_temp = 0;
        Ppe_comm_temp = 0;
        [d_b_opt, Pbat_comm, ib_comm] = getDEMPC_BESS(xb, d_prev(4), ...
            Ppv_comm, Ppe_comm_temp, Pae_comm_temp, is_comm, ip_comm, ia_comm, Pload, Vdc_ref);

        % =================================================================
        % STEP 6: Local DEMPC for AES  (only when EMS activates it)
        % When off: force iae=0, reset communication to zero
        % =================================================================
        if zeta_a == 1
            xa      = [iae; Vdc];
            [d_ae_opt, Pae_comm, ia_comm, ~] = getDEMPC_AES(xa, d_prev(2), ...
                Ppv_comm, Ppe_comm, is_comm, ip_comm, Pload_eff, Vdc_ref);
        else
            d_ae_opt = 0;
            Pae_comm = 0;
            ia_comm  = 0;
        end

        % =================================================================
        % STEP 7: Local DEMPC for PEMFCS  (only when EMS activates it)
        % When off: force ipe=0, reset communication to zero
        % =================================================================
        if zeta_p == 1
            xp      = [ipe; Vdc];
            [d_pe_opt, Ppe_comm, ip_comm] = getDEMPC_PEMFCS(xp, d_prev(3), ...
                Ppv_comm, Pae_comm, is_comm, ia_comm, Pload_eff, Vdc_ref);
        else
            d_pe_opt = 0;
            Ppe_comm = 0;
            ip_comm  = 0;
        end

        % =================================================================
        % STEP 8: Apply optimal control to plant  (forward Euler integration)
        % =================================================================
        u   = [d_s_opt; d_ae_opt; d_pe_opt; d_b_opt; d_w_opt];
        w   = [G_now; T_now; Pload; v_w_now];

        Ts_sub  = Ts / 4;
        for sub = 1:4
            [xdot, aux] = getMicrogridModel(x, u, w);
            x           = x + Ts_sub * xdot;
            x(1)        = max(x(1), 0);
            x(2)        = max(x(2), 0);
            x(3)        = max(x(3), 0);
            x(4)        = max(x(4), 0);
            x(5)        = max(x(5), 1);
            x(7)        = max(min(x(7), 1), 0); % Bound SOC between 0 and 1
            x(8)        = max(x(8), 0);
            x(9)        = max(x(9), 0);
        end

        % When subsystem is off, forcefully zero the state (contactor open)
        if zeta_a == 0
            x(3) = 0;
        end
        if zeta_p == 0
            x(4) = 0;
        end

        % =================================================================
        % STEP 9: Update previous switch signals
        % =================================================================
        d_prev = u;

        % =================================================================
        % STEP 10: Log results
        % =================================================================
        % Update Tank state
        [P_tank, n_H2]  = getHydrogenTank(n_H2, aux.NH2, aux.qH2, Ts);

        X(k,:)          = x';
        U(k,:)          = u';
        Ppv_log(k)      = aux.Ppv;
        Pwt_log(k)      = aux.Pwt;
        Pae_log(k)      = aux.Pae;
        Ppe_log(k)      = aux.Ppe;
        Pbal_log(k)     = aux.Pbal;
        NH2_log(k)      = aux.NH2;
        qH2_log(k)      = aux.qH2;
        Pmax_log(k)     = Pmax_tot;
        Vdc_ref_log(k)  = Vdc_ref;
        zeta_log(k,:)   = [zeta_a, zeta_p];
        Ptank_log(k)    = P_tank;
        Pbat_log(k)     = aux.Pbat;
        SOC_log(k)      = x(7);

        % Progress report
        if mod(k, report_every) == 0
            Perror = aux.Ppv + aux.Ppe + aux.Pbat - Pload - aux.Pae;
            fprintf('  %3d%% | t=%.4fs | Vdc=%.1fV | Ppv=%.2fkW | Pwt=%.2fkW | Pbat=%.2fkW | Pae=%.2fkW | Pfc=%.2fkW | Perror=%.0fW | EMS=[%d,%d]\n', ...
                round(100*k/N_steps), t_now, x(5), aux.Ppv/1000, aux.Pwt/1000, aux.Pbat/1000, aux.Pae/1000, aux.Ppe/1000, Perror, zeta_a, zeta_p);
        end
    end

    % =====================================================================
    % PACKAGE RESULTS
    % =====================================================================
    results.t       = t_vec;
    results.X       = X;           % [vpv, iL, iae, ipe, Vdc, ib, SOC, vwt, iLw]
    results.U       = U;           % [Ss, Sae, Spe, Sb, Sw]
    results.Ppv     = Ppv_log;
    results.Pwt     = Pwt_log;
    results.Pae     = Pae_log;
    results.Ppe     = Ppe_log;
    results.Pmax    = Pmax_log;
    results.Pbal    = Pbal_log;
    results.NH2     = NH2_log;
    results.qH2     = qH2_log;
    results.Vdc_ref = Vdc_ref_log;
    results.zeta    = zeta_log;
    results.Ptank   = Ptank_log;
    results.Pbat    = Pbat_log;
    results.SOC     = SOC_log;
    results.Ts      = Ts;
    results.total_H2_prod = sum(NH2_log) * Ts; % Total H2 produced [mol]
    results.total_H2_cons = sum(qH2_log) * Ts; % Total H2 consumed [mol]

    fprintf('Simulation complete.\n');
end