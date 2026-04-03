% =========================================================================
% test_RL_Environment.m
% Demonstrates the RL Macro-Model running a 10-minute simulation instantly.
% =========================================================================
clear; clc; close all;

% Simulation settings
dt = 1;              % 1-second time step
t_end = 600;         % 10 minutes
N_steps = t_end / dt;

% Initial State
SOC0   = 0.50;       % 50% SOC
Ptank0 = 10 * 1e5;   % 10 bar
state  = [SOC0; Ptank0];

% Logging arrays
log_SOC = zeros(N_steps, 1);
log_Ptnk = zeros(N_steps, 1);
log_Rwd = zeros(N_steps, 1);
log_Act = zeros(N_steps, 1);

fprintf('Running 10-Minute RL Environment Test...\n');
tic;

prev_action = 1;
for k = 1:N_steps
    % Simulated Disturbances (Randomized step load and PV)
    Ppv   = 35000 + 5000 * sin(2*pi*k/300); % PV waving between 30kW and 40kW
    Pload = 30000 + 15000 * (mod(k, 120) > 60); % Load stepping between 30kW and 45kW

    % THE AGENT'S ACTION (Random for now, later replaced by Neural Network)
    % 1 = Idle, 2 = AE ON, 3 = FC ON
    action = randi([1, 3]); 

    % Step the Environment
    [next_state, reward, is_done, info] = stepMicrogridEnv(state, action, prev_action, Ppv, Pload, dt);

    % Log
    log_SOC(k) = state(1);
    log_Ptnk(k)= state(2);
    log_Rwd(k) = reward;
    log_Act(k) = action;

    % Transition
    state = next_state;
    prev_action = action;
    if is_done
        fprintf('Episode terminated early at step %d due to fatal SOC limits!\n', k);
        break;
    end
end

fprintf('Simulation completed in %.4f seconds.\n', toc);

% Plotting the Agent's Experience
figure('Color', 'w', 'Position', [100 100 800 600]);
subplot(3,1,1);
plot(log_SOC * 100, 'm', 'LineWidth', 1.5); title('Battery SOC (%)'); grid on;
subplot(3,1,2);
plot(log_Ptnk / 1e5, 'b', 'LineWidth', 1.5); title('Tank Pressure (Bar)'); grid on;
subplot(3,1,3);
stairs(log_Act, 'k', 'LineWidth', 1.5); title('Agent Actions (1=Idle, 2=AE, 3=FC)'); 
yticks([1 2 3]); ylim([0.5 3.5]); grid on;
xlabel('Time (seconds)');