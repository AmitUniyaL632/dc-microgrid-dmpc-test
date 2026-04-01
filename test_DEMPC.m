% =========================================================================
% test_DEMPC.m
% Closed-loop DEMPC simulation of PV/Hydrogen DC Microgrid
%
% Scenario matches paper Section 4:
%   Irradiance : 1000 W/m^2, ramps to 1100 (0.1-0.2s), drops to 1000 (0.3s)
%   Load       : 30 kW, steps to 45 kW at t = 0.2s
%   Vdc setpoint: 300 V
%
% Expected behaviour:
%   t=0 to 0.2s  : Ppv > Pload → AE ON (absorbing surplus), FC OFF
%   t=0.2s+      : Ppv < Pload → AE OFF, FC ON (supplying deficit)
% =========================================================================
clc; clear; close all;

% =========================================================================
% DISTURBANCE PROFILES
% =========================================================================
% Irradiance: 1000 -> ramp to 1100 -> hold 1100 -> step to 1000 (0.2s intervals)
G_profile = @(t) 1000 .* (t < 0.2) + ...
                 (1000 + 100*(t-0.2)/0.2) .* (t >= 0.2 & t < 0.4) + ...
                 1100 .* (t >= 0.4 & t < 0.6) + ...
                 1000 .* (t >= 0.6);

% Temperature: fixed
T_profile = @(t) 25;

% Load: 20kW -> ramp to 40kW -> hold 40kW -> step to 50kW (0.2s intervals)
Pload_profile = @(t) 20000 .* (t < 0.2) + ...
                     (20000 + 20000*(t-0.2)/0.2) .* (t >= 0.2 & t < 0.4) + ...
                     40000 .* (t >= 0.4 & t < 0.6) + ...
                     50000 .* (t >= 0.6);

% =========================================================================
% INITIAL STATE
% =========================================================================
Ns_pv   = 5;    Np_pv = 20;
Voc_mod = 47.6;
G0      = G_profile(0);
T0      = T_profile(0);
Vdc_nom = 300;

% Find MPP at t=0 for initial vpv and iL
V_sw    = linspace(0.1, Ns_pv*Voc_mod, 500);
[~, Pp] = getPVArray(V_sw, G0, T0, Ns_pv, Np_pv);
[~, mi] = max(Pp);
vpv0    = V_sw(mi);
[Impp0, ~] = getPVArray(vpv0, G0, T0, Ns_pv, Np_pv);

% --- Compute initial AE current for surplus power ---
P_surplus_0 = max(vpv0*Impp0 - Pload_profile(0), 0);
iae_sw      = linspace(0, 500, 5000);
vae_sw      = zeros(size(iae_sw));
for kk = 1:length(iae_sw), [vae_sw(kk),~,~,~] = getAE(max(iae_sw(kk),1e-6)); end
[~, idx_ae] = min(abs(vae_sw .* iae_sw - P_surplus_0));
iae_init    = iae_sw(idx_ae);

ib_init  = 0;
SOC_init = 0.8; % 80% Initial SOC
x_init   = [vpv0; Impp0; iae_init; 0; Vdc_nom; ib_init; SOC_init];

fprintf('=== DEMPC Simulation Setup ===\n');
fprintf('  vpv0  = %.2f V\n',   vpv0);
fprintf('  iL0   = %.2f A\n',   Impp0);
fprintf('  Ppv0  = %.2f kW\n',  vpv0*Impp0/1000);

% =========================================================================
% RUN DEMPC  (start with short run to verify, then extend t_sim)
% =========================================================================
t_sim   = 0.8;      % Extended to 800ms for full 4-stage profile

tic;
results = runDEMPC(x_init, t_sim, G_profile, T_profile, Pload_profile);
elapsed = toc;
fprintf('Wall-clock time: %.1f s\n', elapsed);

% =========================================================================
% PLOTS
% =========================================================================
t   = results.t * 1000;     % ms
Pl  = arrayfun(Pload_profile, results.t);
G_t = arrayfun(G_profile,     results.t);

figure('Color','w','Position',[30 30 1400 900]);
sgtitle('DEMPC — Closed-Loop PV/Hydrogen DC Microgrid', ...
        'FontSize', 14, 'FontWeight', 'bold');

% 1. Irradiance and load
subplot(3,3,1);
yyaxis left;
plot(t, G_t, 'b', 'LineWidth', 2); ylabel('G (W/m^2)');
yyaxis right;
plot(t, Pl/1000, 'r', 'LineWidth', 2); ylabel('P_{load} (kW)');
grid on; xlabel('Time (ms)');
title('Disturbances: Irradiance & Load');

% 2. DC bus voltage
subplot(3,3,2);
plot(t, results.X(:,5), 'k', 'LineWidth', 2); hold on;
plot(t, results.Vdc_ref, 'b--', 'LineWidth', 1.5);
yline(300, 'r:', 'V_{dc}^*=300V', 'LineWidth', 1.2);
grid on; xlabel('Time (ms)'); ylabel('V_{dc} (V)');
title('DC Bus Voltage');
ylim([270 330]);
legend('V_{dc}','V_{dc}^{ref}','Location','best');

% 3. Power balance
subplot(3,3,3);
plot(t, (results.Ppv+results.Ppe+results.Pbat-results.Pae)/1000, 'b', 'LineWidth', 2); hold on;
plot(t, Pl/1000, 'r--', 'LineWidth', 2);
grid on; xlabel('Time (ms)'); ylabel('Power (kW)');
title('Power Supply vs Demand');
legend('Supply','P_{load}','Location','best');

% 4. MPPT tracking
subplot(3,3,4);
plot(t, results.Ppv/1000, 'b', 'LineWidth', 2); hold on;
plot(t, results.Pmax/1000, 'r--', 'LineWidth', 1.5);
grid on; xlabel('Time (ms)'); ylabel('Power (kW)');
title('PV Power vs P_{max}  (MPPT)');
legend('P_{pv}','P_{max}','Location','best');

% 5. PV converter states
subplot(3,3,5);
yyaxis left;
plot(t, results.X(:,1), 'b', 'LineWidth', 2);
ylabel('v_{pv} (V)');
yyaxis right;
plot(t, results.X(:,2), 'r', 'LineWidth', 1.5);
ylabel('i_L (A)');
grid on; xlabel('Time (ms)');
title('PV Boost Converter States');

% 6. H2 and Battery Powers
subplot(3,3,6);
plot(t, results.Pae/1000, 'r',  'LineWidth', 2); hold on;
plot(t, results.Ppe/1000, 'g',  'LineWidth', 2);
plot(t, results.Pbat/1000, 'm',  'LineWidth', 2);
grid on; xlabel('Time (ms)'); ylabel('Power (kW)');
title('Storage & H_2 Powers');
legend('P_{ae} (AE)','P_{pe} (FC)', 'P_{bat}', 'Location','best');

% 7. Switching signals
subplot(3,3,7);
stairs(t, results.U(:,1), 'b',  'LineWidth', 1.5); hold on;
stairs(t, results.U(:,2), 'r',  'LineWidth', 1.5);
stairs(t, results.U(:,3), 'g',  'LineWidth', 1.5);
stairs(t, results.U(:,4), 'm',  'LineWidth', 1.5);
ylim([-0.05, 1.05]); grid on;
xlabel('Time (ms)'); ylabel('Duty Cycle');
title('Control Inputs  [d_s, d_{ae}, d_{pe}]');
legend('d_s','d_{ae}','d_{pe}','d_b','Location','best');

% 8. EMS modes
subplot(3,3,8);
stairs(t, results.zeta(:,1), 'r', 'LineWidth', 2); hold on;
stairs(t, results.zeta(:,2), 'g', 'LineWidth', 2);
ylim([-0.2, 1.4]); grid on;
xlabel('Time (ms)'); ylabel('Mode');
title('EMS Operational Modes');
legend('\zeta_a (AE on)','\zeta_p (FC on)','Location','best');

% 9. Hydrogen rates
subplot(3,3,9);
plot(t, results.NH2*1000, 'g', 'LineWidth', 2); hold on;
plot(t, results.qH2*1000, 'r', 'LineWidth', 2);
grid on; xlabel('Time (ms)'); ylabel('Rate (mmol/s)');
title('H_2 Production (AE) & Consumption (FC)');
legend('N_{H2}','q_{H2}','Location','best');

% 10. Hydrogen Tank Pressure (New Figure)
figure('Color','w','Position',[50 50 600 400]);
plot(t, results.Ptank / 1e5, 'b', 'LineWidth', 2);
grid on;
xlabel('Time (ms)');
ylabel('Pressure (bar)');
title('Hydrogen Tank Pressure');
xlim([0 t(end)]);

% figure('Color','w','Position',[30 30 1400 900]);
% sgtitle('Switching signals', ...
%         'FontSize', 14, 'FontWeight', 'bold');
% subplot(3,1,1);
% stairs(t, results.U(:,1), 'b',  'LineWidth', 1.5);
% ylim([-0.2, 1.2]); grid on;
% xlabel('Time (ms)'); ylabel('Switch Signal');
% title('Control Inputs  [S_s]');
% legend('S_s','Location','best');
% 
% subplot(3,1,2);
% stairs(t, results.U(:,2), 'r',  'LineWidth', 1.5);
% ylim([-0.2, 1.2]); grid on;
% xlabel('Time (ms)'); ylabel('Switch Signal');
% title('Control Inputs  [S_{ae}]');
% legend('S_{ae}','Location','best');
% 
% subplot(3,1,3);
% stairs(t, results.U(:,3), 'g',  'LineWidth', 1.5);
% ylim([-0.2, 1.2]); grid on;
% xlabel('Time (ms)'); ylabel('Switch Signal');
% title('Control Inputs  [S_{pe}]');
% legend('S_{pe}','Location','best');

% =========================================================================
% PERFORMANCE METRICS
% =========================================================================
Vdc_ss  = results.X(:,5);
Pbal    = results.Pbal;

fprintf('\n=== DEMPC Performance Metrics ===\n');
fprintf('Vdc mean          : %.3f V   (target 300 V)\n', mean(Vdc_ss));
fprintf('Vdc std dev       : %.4f V\n',                   std(Vdc_ss));
fprintf('Vdc max deviation : %.4f V\n',                   max(abs(Vdc_ss-300)));
fprintf('Power balance RMSE: %.3f kW\n',                  rms(Pbal)/1000);
fprintf('PV utilization    : %.2f %%\n', ...
    100*mean(results.Ppv(results.Pmax>0)) / mean(results.Pmax(results.Pmax>0)));
fprintf('AE active fraction: %.2f %%\n', 100*mean(results.zeta(:,1)));
fprintf('FC active fraction: %.2f %%\n', 100*mean(results.zeta(:,2)));
fprintf('Total H2 produced : %.6f mol\n', results.total_H2_prod);
fprintf('Total H2 consumed : %.6f mol\n', results.total_H2_cons);
fprintf('Final Tank Press. : %.4f bar\n', results.Ptank(end) / 1e5);