% =========================================================================
% test_GWO_DMPC.m
%
% Runs a closed-loop simulation of the DC Microgrid using the GWO-based
% Distributed Model Predictive Controller (DMPC).
%
% This script will:
% 1. Define the simulation scenario (initial state, duration, disturbances).
% 2. Run the 'runDEMPC_GWO' simulation.
% 3. Plot the key results.
% =========================================================================

% clear; clc; close all;

% ---- Simulation Scenario ----
t_sim = 0.4; % Simulation time in seconds, matching test_DEMPC

% ---- Disturbance Profiles (matching test_DEMPC.m) ----
% Irradiance: ramp from 1000 to 1100 between 0.1-0.2s, step back at 0.3s
G_profile = @(t) 1000 ...
    + 100 * min(max((t - 0.1)/0.1, 0), 1) ...   % Ramp up
    - 100 * double(t >= 0.3);                    % Step down

% Temperature: fixed
T_profile = @(t) 25;

% Load: 30 kW stepping to 45 kW at t=0.2s
Pload_profile = @(t) 30000 + 15000 * double(t >= 0.2);

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

x_init  = [vpv0; Impp0; iae_init; 0; Vdc_nom];

% =========================================================================
% RUN SIMULATION
% =========================================================================

% Run the new DMPC with Grey Wolf Optimizer
fprintf('--- Starting GWO-DMPC Simulation ---\n');
tic;
results_gwo = runDEMPC_GWO(x_init, t_sim, G_profile, T_profile, Pload_profile);
elapsed_time = toc;
fprintf('--- GWO-DMPC Simulation Complete (%.2f seconds) ---\n\n', elapsed_time);

% =========================================================================
% PLOT RESULTS
% =========================================================================

fprintf('Plotting GWO-DMPC results...\n');

% Figure 1: DC Bus Voltage
figure('Name', 'GWO-DMPC: DC Bus Voltage');
plot(results_gwo.t, results_gwo.X(:,5), 'b-', 'LineWidth', 1.5);
hold on;
plot(results_gwo.t, results_gwo.Vdc_ref, 'r--', 'LineWidth', 1.5);
grid on;
title('DC Bus Voltage (V_{dc})');
xlabel('Time (s)');
ylabel('Voltage (V)');
legend('V_{dc} (GWO)', 'V_{dc} Reference');
ylim([270 330]);

% Figure 2: Power Profiles
figure('Name', 'GWO-DMPC: Power Profiles');

subplot(3,1,1);
plot(results_gwo.t, results_gwo.Ppv/1000, 'b-', 'LineWidth', 1.5);
hold on; plot(results_gwo.t, results_gwo.Pmax/1000, 'r--', 'LineWidth', 1.5);
grid on; title('PV Power (P_{pv})'); ylabel('Power (kW)');
legend('P_{pv} (GWO)', 'P_{max}');

subplot(3,1,2);
plot(results_gwo.t, results_gwo.Pae/1000, 'b-', 'LineWidth', 1.5);
grid on; title('Electrolyzer Power (P_{ae})'); ylabel('Power (kW)');
legend('P_{ae} (GWO)');

subplot(3,1,3);
plot(results_gwo.t, results_gwo.Ppe/1000, 'b-', 'LineWidth', 1.5);
grid on; title('Fuel Cell Power (P_{pe})'); ylabel('Power (kW)');
xlabel('Time (s)');
legend('P_{pe} (GWO)');

% Figure 3: Duty Cycles
figure('Name', 'GWO-DMPC: Duty Cycles');

subplot(3,1,1);
stairs(results_gwo.t, results_gwo.U(:,1), 'b-', 'LineWidth', 1.5);
grid on; title('PV Duty Cycle (S_s)'); ylabel('Duty Cycle');
legend('d_s (GWO)');

subplot(3,1,2);
stairs(results_gwo.t, results_gwo.U(:,2), 'b-', 'LineWidth', 1.5);
grid on; title('AE Duty Cycle (S_{ae})'); ylabel('Duty Cycle');
legend('d_{ae} (GWO)');

subplot(3,1,3);
stairs(results_gwo.t, results_gwo.U(:,3), 'b-', 'LineWidth', 1.5);
grid on; title('FC Duty Cycle (S_{pe})'); ylabel('Duty Cycle');
xlabel('Time (s)');
legend('d_{pe} (GWO)');

% Figure 4: Hydrogen Tank Pressure
figure('Name', 'GWO-DMPC: Hydrogen Tank Pressure');
plot(results_gwo.t, results_gwo.Ptank / 1e5, 'b-', 'LineWidth', 1.5);
grid on;
title('Hydrogen Tank Pressure');
xlabel('Time (s)');
ylabel('Pressure (bar)');
xlim([0 results_gwo.t(end)]);

% Calculate and display performance metrics
Vdc_IAE_gwo   = sum(abs(results_gwo.X(:,5) - results_gwo.Vdc_ref)) * results_gwo.Ts;

fprintf('\n--- GWO-DMPC Performance Metrics ---\n');
fprintf('Simulation Wall-Clock Time:       %.2f seconds\n', elapsed_time);
fprintf('Vdc Integral Absolute Error (IAE): %.4f\n', Vdc_IAE_gwo);
fprintf('Total H2 Produced:                 %.6f mol\n', results_gwo.total_H2_prod);
fprintf('Total H2 Consumed:                 %.6f mol\n', results_gwo.total_H2_cons);
fprintf('Final Tank Pressure:               %.4f bar\n', results_gwo.Ptank(end) / 1e5);
fprintf('-------------------------------------\n');