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

subplot(4,1,1);
plot(results_gwo.t, results_gwo.Ppv/1000, 'b-', 'LineWidth', 1.5);
hold on; plot(results_gwo.t, results_gwo.Pmax/1000, 'r--', 'LineWidth', 1.5);
grid on; title('PV Power (P_{pv})'); ylabel('Power (kW)');
legend('P_{pv} (GWO)', 'P_{max}');

subplot(4,1,2);
plot(results_gwo.t, results_gwo.Pae/1000, 'b-', 'LineWidth', 1.5);
grid on; title('Electrolyzer Power (P_{ae})'); ylabel('Power (kW)');
legend('P_{ae} (GWO)');

subplot(4,1,3);
plot(results_gwo.t, results_gwo.Ppe/1000, 'b-', 'LineWidth', 1.5);
grid on; title('Fuel Cell Power (P_{pe})'); ylabel('Power (kW)');
legend('P_{pe} (GWO)');

subplot(4,1,4);
plot(results_gwo.t, results_gwo.Pbat/1000, 'm-', 'LineWidth', 1.5);
grid on; title('Battery Power (P_{bat})'); ylabel('Power (kW)');
xlabel('Time (s)');
legend('P_{bat} (GWO)');

% Figure 3: Duty Cycles
figure('Name', 'GWO-DMPC: Duty Cycles');

subplot(4,1,1);
stairs(results_gwo.t, results_gwo.U(:,1), 'b-', 'LineWidth', 1.5);
grid on; title('PV Duty Cycle (S_s)'); ylabel('Duty Cycle');
legend('d_s (GWO)');

subplot(4,1,2);
stairs(results_gwo.t, results_gwo.U(:,2), 'b-', 'LineWidth', 1.5);
grid on; title('AE Duty Cycle (S_{ae})'); ylabel('Duty Cycle');
legend('d_{ae} (GWO)');

subplot(4,1,3);
stairs(results_gwo.t, results_gwo.U(:,3), 'b-', 'LineWidth', 1.5);
grid on; title('FC Duty Cycle (S_{pe})'); ylabel('Duty Cycle');
legend('d_{pe} (GWO)');

subplot(4,1,4);
stairs(results_gwo.t, results_gwo.U(:,4), 'm-', 'LineWidth', 1.5);
grid on; title('Battery Duty Cycle (S_b)'); ylabel('Duty Cycle');
xlabel('Time (s)');
legend('d_b (GWO)');

% Figure 4: Hydrogen Tank Pressure
figure('Name', 'GWO-DMPC: Hydrogen Tank Pressure');
plot(results_gwo.t, results_gwo.Ptank / 1e5, 'b-', 'LineWidth', 1.5);
grid on;
title('Hydrogen Tank Pressure');
xlabel('Time (s)');
ylabel('Pressure (bar)');
xlim([0 results_gwo.t(end)]);

% Figure 5: Battery SOC
figure('Name', 'GWO-DMPC: Battery SOC');
plot(results_gwo.t, results_gwo.SOC * 100, 'm-', 'LineWidth', 1.5);
grid on;
title('Battery State of Charge');
xlabel('Time (s)');
ylabel('SOC (%)');
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