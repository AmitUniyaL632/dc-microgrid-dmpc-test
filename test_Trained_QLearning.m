% =========================================================================
% test_Trained_QLearning.m
% Evaluates the fully trained Q-Learning Agent on the Macro Environment.
% =========================================================================
clear; clc; close all;

% Load the trained brain (Q-Table)
if ~isfile('Q_Table_EMS.mat')
    error('Could not find Q_Table_EMS.mat. Please run train_QLearning_EMS.m first.');
end
load('Q_Table_EMS.mat', 'Q');

% Simulation settings
dt = 1;              % 1-second time step
t_end = 600;         % 10 minutes
N_steps = t_end / dt;

% Initial State
SOC0   = 0.50;       % 50% SOC
Ptank0 = 10 * 1e5;   % 10 bar
state  = [SOC0; Ptank0];

% Logging arrays
log_SOC  = zeros(N_steps, 1);
log_Ptnk = zeros(N_steps, 1);
log_Act  = zeros(N_steps, 1);
log_Pnet = zeros(N_steps, 1);

fprintf('Running 10-Minute Evaluation with Trained AI...\n');

prev_action = 1;
for k = 1:N_steps
    % Simulated Disturbances (Same as training)
    Ppv   = 35000 + 5000 * sin(2*pi*k/300); 
    Pload = 30000 + 15000 * (mod(k, 120) > 60); 
    Pnet  = Ppv - Pload;
    
    % 1. Convert raw readings into Discrete States (Buckets)
    SOC = state(1);
    if SOC < 0.30; s_soc = 1;
    elseif SOC <= 0.80; s_soc = 2;
    else; s_soc = 3; end
    
    if Pnet < -5000; s_pnet = 1;
    elseif Pnet <= 5000; s_pnet = 2;
    else; s_pnet = 3; end
    
    % 2. THE AI'S ACTION (100% Exploit, 0% Random)
    [~, action] = max(Q(s_soc, s_pnet, :));

    % Step the Environment
    [next_state, ~, is_done, ~] = stepMicrogridEnv(state, action, prev_action, Ppv, Pload, dt);

    % Log
    log_SOC(k)  = state(1);
    log_Ptnk(k) = state(2);
    log_Act(k)  = action;
    log_Pnet(k) = Pnet;

    % Transition
    state = next_state;
    prev_action = action;
    if is_done
        fprintf('Episode terminated early at step %d due to fatal SOC limits!\n', k);
        break;
    end
end

% Plotting the Agent's Performance
figure('Color', 'w', 'Position', [100 100 800 800]);
subplot(4,1,1); plot(log_Pnet / 1000, 'k', 'LineWidth', 1.5); title('Net Power (PV - Load) [kW]'); grid on;
subplot(4,1,2); plot(log_SOC * 100, 'm', 'LineWidth', 1.5); title('Battery SOC (%)'); grid on;
subplot(4,1,3); plot(log_Ptnk / 1e5, 'b', 'LineWidth', 1.5); title('Tank Pressure (Bar)'); grid on;
subplot(4,1,4); stairs(log_Act, 'r', 'LineWidth', 2); title('AI Actions (1=Idle, 2=AE, 3=FC)'); 
yticks([1 2 3]); ylim([0.5 3.5]); grid on;
xlabel('Time (seconds)');