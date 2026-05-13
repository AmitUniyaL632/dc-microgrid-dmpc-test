%% 
% =========================================================================
% test_comparison.m
%
% Compares the performance of the original DMPC (grid search) against
% the new DMPC using Grey Wolf Optimization (GWO).
%
% This script will:
% 1. Define the simulation scenario (initial state, duration, disturbances).
% 2. Run the original 'runDEMPC' simulation.
% 3. Run the new 'runDEMPC_GWO' simulation.
% 4. Plot the results of both simulations on the same axes for comparison.
% =========================================================================

%clear; clc; close all;

% ---- Simulation Scenario ----
t_sim = 0.8; % Extended to 800ms for full 4-stage profile

% ---- Disturbance Profiles (matching test_DEMPC.m) ----
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

% ---- Initial State Calculation (matching test_DEMPC.m) ----
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
[Impp0, Ppv0] = getPVArray(vpv0, G0, T0, Ns_pv, Np_pv);

% Compute initial AE current for surplus power
P_surplus_0 = max(Ppv0 - Pload_profile(0), 0);
iae_sw      = linspace(0, 500, 5000);
[vae_sw, ~, ~, ~] = getAE(max(iae_sw,1e-6));
[~, idx_ae] = min(abs(vae_sw .* iae_sw - P_surplus_0));
iae_init    = iae_sw(idx_ae);

ib_init  = 0;
SOC_init = 0.8; % 80% Initial SOC
x_init   = [vpv0; Impp0; iae_init; 0; Vdc_nom; ib_init; SOC_init];

% =========================================================================
% RUN SIMULATIONS
% =========================================================================

% Run the original DMPC with grid search
fprintf('--- Starting Original DMPC Simulation ---\n');
results_dempc = runDEMPC(x_init, t_sim, G_profile, T_profile, Pload_profile);
fprintf('--- Original DMPC Simulation Complete ---\n\n');

% Run the new DMPC with Grey Wolf Optimizer
fprintf('--- Starting GWO-DMPC Simulation ---\n');
results_gwo = runDEMPC_GWO(x_init, t_sim, G_profile, T_profile, Pload_profile);
fprintf('--- GWO-DMPC Simulation Complete ---\n\n');

% =========================================================================
% PLOT COMPARISON RESULTS
% =========================================================================

% fprintf('Plotting comparison results...\n');

% % Figure 1: DC Bus Voltage
% figure('Name', 'DC Bus Voltage Comparison');
% plot(results_dempc.t, results_dempc.X(:,5), 'b-', 'LineWidth', 1.5);
% hold on;
% plot(results_gwo.t, results_gwo.X(:,5), 'r--', 'LineWidth', 1.5);
% plot(results_dempc.t, results_dempc.Vdc_ref, 'k:', 'LineWidth', 1);
% grid on;
% title('DC Bus Voltage (V_{dc})');
% xlabel('Time (s)');
% ylabel('Voltage (V)');
% legend('DMPC (Grid Search)', 'DMPC (GWO)', 'V_{dc} Reference');
% ylim([270 330]);

% % Figure 2: Power Profiles
% figure('Name', 'Power Profile Comparison');

% subplot(4,1,1);
% plot(results_dempc.t, results_dempc.Ppv/1000, 'b-', results_gwo.t, results_gwo.Ppv/1000, 'r--');
% hold on; plot(results_dempc.t, results_dempc.Pmax/1000, 'g:');
% grid on; title('PV Power (P_{pv})'); ylabel('Power (kW)');
% legend('DMPC', 'GWO', 'P_{max}');

% subplot(4,1,2);
% plot(results_dempc.t, results_dempc.Pae/1000, 'b-', results_gwo.t, results_gwo.Pae/1000, 'r--');
% grid on; title('Electrolyzer Power (P_{ae})'); ylabel('Power (kW)');
% legend('DMPC', 'GWO');

% subplot(4,1,3);
% plot(results_dempc.t, results_dempc.Ppe/1000, 'b-', results_gwo.t, results_gwo.Ppe/1000, 'r--');
% grid on; title('Fuel Cell Power (P_{pe})'); ylabel('Power (kW)');
% legend('DMPC', 'GWO');

% subplot(4,1,4);
% plot(results_dempc.t, results_dempc.Pbat/1000, 'b-', results_gwo.t, results_gwo.Pbat/1000, 'r--');
% grid on; title('Battery Power (P_{bat})'); ylabel('Power (kW)');
% xlabel('Time (s)');
% legend('DMPC', 'GWO');

% % Figure 3: Duty Cycles
% figure('Name', 'Duty Cycle Comparison');

% subplot(4,1,1);
% plot(results_dempc.t, results_dempc.U(:,1), 'b-', results_gwo.t, results_gwo.U(:,1), 'r--');
% grid on; title('PV Duty Cycle (S_s)'); ylabel('Duty Cycle');
% legend('DMPC', 'GWO');

% subplot(4,1,2);
% plot(results_dempc.t, results_dempc.U(:,2), 'b-', results_gwo.t, results_gwo.U(:,2), 'r--');
% grid on; title('AE Duty Cycle (S_{ae})'); ylabel('Duty Cycle');
% legend('DMPC', 'GWO');

% subplot(4,1,3);
% plot(results_dempc.t, results_dempc.U(:,3), 'b-', results_gwo.t, results_gwo.U(:,3), 'r--');
% grid on; title('FC Duty Cycle (S_{pe})'); ylabel('Duty Cycle');
% legend('DMPC', 'GWO');

% subplot(4,1,4);
% plot(results_dempc.t, results_dempc.U(:,4), 'b-', results_gwo.t, results_gwo.U(:,4), 'r--');
% grid on; title('Battery Duty Cycle (S_b)'); ylabel('Duty Cycle');
% xlabel('Time (s)');
% legend('DMPC', 'GWO');

% % Figure 4: Hydrogen Tank Pressure
% figure('Name', 'Hydrogen Tank Pressure Comparison');
% plot(results_dempc.t, results_dempc.Ptank / 1e5, 'b-', 'LineWidth', 1.5);
% hold on;
% plot(results_gwo.t, results_gwo.Ptank / 1e5, 'r--', 'LineWidth', 1.5);
% grid on;
% title('Hydrogen Tank Pressure');
% xlabel('Time (s)');
% ylabel('Pressure (bar)');
% legend('DMPC (Grid Search)', 'DMPC (GWO)');
% xlim([0 t_sim]);

% % Figure 5: Battery SOC Comparison
% figure('Name', 'Battery SOC Comparison');
% plot(results_dempc.t, results_dempc.SOC * 100, 'b-', 'LineWidth', 1.5);
% hold on;
% plot(results_gwo.t, results_gwo.SOC * 100, 'r--', 'LineWidth', 1.5);
% grid on;
% title('Battery State of Charge');
% xlabel('Time (s)');
% ylabel('SOC (%)');
% legend('DMPC (Grid Search)', 'DMPC (GWO)');
% xlim([0 t_sim]);

% Calculate and display performance metrics
Vdc_IAE_dempc = sum(abs(results_dempc.X(:,5) - results_dempc.Vdc_ref)) * results_dempc.Ts;
Vdc_IAE_gwo   = sum(abs(results_gwo.X(:,5) - results_gwo.Vdc_ref)) * results_gwo.Ts;

fprintf('\n--- Performance Metrics ---\n');
fprintf('Vdc Integral Absolute Error (DMPC): %.4f\n', Vdc_IAE_dempc);
fprintf('Vdc Integral Absolute Error (GWO):  %.4f\n', Vdc_IAE_gwo);
fprintf('Total H2 Produced (DMPC): %.6f mol\n', results_dempc.total_H2_prod);
fprintf('Total H2 Produced (GWO):  %.6f mol\n', results_gwo.total_H2_prod);
fprintf('Total H2 Consumed (DMPC): %.6f mol\n', results_dempc.total_H2_cons);
fprintf('Total H2 Consumed (GWO):  %.6f mol\n', results_gwo.total_H2_cons);
fprintf('Final Tank Pressure (DMPC): %.4f bar\n', results_dempc.Ptank(end) / 1e5);
fprintf('Final Tank Pressure (GWO):  %.4f bar\n', results_gwo.Ptank(end) / 1e5);
fprintf('---------------------------\n');