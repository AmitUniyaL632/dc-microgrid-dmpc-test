% =========================================================================
% train_QLearning_EMS.m
% Trains a Tabular Q-Learning Agent to replace the rule-based EMS.
% =========================================================================
clear; clc; close all;

% --- RL Hyperparameters ---
num_episodes = 500;       % Number of 10-minute training runs
alpha        = 0.1;       % Learning rate (how fast it updates memory)
gamma        = 0.95;      % Discount factor (how much it cares about the future)
epsilon      = 1.0;       % Exploration rate (100% random at start)
epsilon_min  = 0.05;      % Minimum exploration rate
epsilon_decay= 0.99;      % Decay rate per episode

% --- Define the Discrete State Space ---
% To use a Q-Table, we must group continuous numbers into simple buckets.
% SOC States: 1 = Low (<30%), 2 = Normal (30-80%), 3 = High (>80%)
% Net Power States: 1 = Deficit (<-5kW), 2 = Balanced (±5kW), 3 = Surplus (>5kW)

num_soc_states  = 3;
num_pnet_states = 3;
num_actions     = 3; % 1=Idle, 2=AE, 3=FC

% Initialize the Q-Table (Cheat Sheet) with zeros
Q = zeros(num_soc_states, num_pnet_states, num_actions);

% Logging
reward_history = zeros(num_episodes, 1);

fprintf('Starting Q-Learning Training for %d episodes...\n', num_episodes);
tic;

for episode = 1:num_episodes
    % Reset Environment for a new 10-minute run
    dt = 1; N_steps = 600;
    % Randomize initial SOC between 15% and 95% so the AI explores all states!
    state = [0.15 + 0.80 * rand(); 10e5]; 
    prev_action = 1;      % Start Idle
    total_reward = 0;
    
    for k = 1:N_steps
        % 1. Get Current Disturbances
        Ppv   = 35000 + 5000 * sin(2*pi*k/300); 
        Pload = 30000 + 15000 * (mod(k, 120) > 60);
        Pnet  = Ppv - Pload;
        
        % 2. Convert raw readings into Discrete States (Buckets)
        SOC = state(1);
        
        if SOC < 0.30; s_soc = 1;
        elseif SOC <= 0.80; s_soc = 2;
        else; s_soc = 3; end
        
        if Pnet < -5000; s_pnet = 1;
        elseif Pnet <= 5000; s_pnet = 2;
        else; s_pnet = 3; end
        
        % 3. Choose Action (Epsilon-Greedy)
        if rand() < epsilon
            % Explore: Pick a random action
            action = randi([1, 3]);
        else
            % Exploit: Pick the best action from the Q-Table
            [~, action] = max(Q(s_soc, s_pnet, :));
        end
        
        % 4. Step the Environment
        [next_state, reward, is_done, ~] = stepMicrogridEnv(state, action, prev_action, Ppv, Pload, dt);
        
        % 5. Get the Next Discrete State
        next_SOC = next_state(1);
        if next_SOC < 0.30; next_s_soc = 1;
        elseif next_SOC <= 0.80; next_s_soc = 2;
        else; next_s_soc = 3; end
        
        % (Pnet for the next step is unknown to the agent, so we assume it stays roughly the same for the Q-update)
        next_s_pnet = s_pnet; 
        
        % 6. Update the Q-Table (The Core Learning Equation)
        best_future_Q = max(Q(next_s_soc, next_s_pnet, :));
        current_Q = Q(s_soc, s_pnet, action);
        
        Q(s_soc, s_pnet, action) = current_Q + alpha * (reward + gamma * best_future_Q - current_Q);
        
        % 7. Transition to next step
        state = next_state;
        prev_action = action;
        total_reward = total_reward + reward;
        
        if is_done
            break; % Grid crashed, end episode early
        end
    end
    
    % Decay exploration rate so it acts smarter over time
    epsilon = max(epsilon_min, epsilon * epsilon_decay);
    reward_history(episode) = total_reward;
    
    % Print progress
    if mod(episode, 50) == 0
        fprintf('Episode %3d | Total Reward: %6.1f | Epsilon: %.2f\n', episode, total_reward, epsilon);
    end
end

fprintf('Training complete in %.2f seconds.\n', toc);

% Plot Learning Curve
figure('Color','w');
plot(1:num_episodes, reward_history, 'b', 'LineWidth', 1.5);
xlabel('Training Episode'); ylabel('Total Reward');
title('AI Learning Curve (Higher is Better)');
grid on;

% Save the trained AI brain
save('Q_Table_EMS.mat', 'Q');