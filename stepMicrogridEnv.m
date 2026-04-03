% =========================================================================
% stepMicrogridEnv.m
% Reinforcement Learning Environment - Macro Power Model
%
% Inputs:
%   state  : [SOC; Ptank (Pa)]
%   action : 1 = EMS Idle, 2 = AE ON, 3 = FC ON
%   prev_action : The action taken in the previous step
%   Ppv    : Current PV Power (W)
%   Pload  : Current Load Power (W)
%   dt     : Time step duration (seconds)
%
% Outputs:
%   next_state : [SOC_new; Ptank_new]
%   reward     : Reward signal for the RL agent
%   is_done    : Boolean flag if episode terminates (e.g. failure)
%   info       : Struct with power logging details
% =========================================================================
function [next_state, reward, is_done, info] = stepMicrogridEnv(state, action, prev_action, Ppv, Pload, dt)

    % Unpack State
    SOC   = state(1);
    Ptank = state(2);

    % System Parameters
    P_bat_max = 5000;                % Battery power limit (W)
    E_bat_max = 240 * 100 * 3600;    % Battery capacity (Joules) -> 240V * 100Ah
    V_tank    = 0.1;                 % Tank volume (m^3)
    T_tank    = 298.15;              % Tank temp (K)
    R_gas     = 8.31446;             % Universal gas constant

    % Calculate initial net power
    P_net = Ppv - Pload;
    P_ae = 0; P_fc = 0;

    % Apply Action (Hydrogen Systems)
    if action == 2 && P_net > 0
        % AE ON: Greedily absorb surplus up to ~20kW limit
        P_ae = min(P_net, 20000);
    elseif action == 3 && P_net < 0
        % FC ON: Greedily supply deficit up to ~15kW limit
        P_fc = min(abs(P_net), 15000);
    end

    % Battery handles the remainder
    P_rem = P_net - P_ae + P_fc;
    P_bat = min(max(-P_rem, -P_bat_max), P_bat_max); % Positive = Discharging

    % Calculate any dropped load or wasted power (Mismatch)
    P_mismatch = abs(P_rem + P_bat);

    % Update States
    SOC_new = SOC - (P_bat * dt) / E_bat_max;
    
    % Simplified Hydrogen Molar Conversion (approx from your models)
    NH2 = (P_ae / 1000) * 0.0006; % ~0.6 mmol per kW
    qH2 = (P_fc / 1000) * 0.0012; % ~1.2 mmol per kW
    
    n_H2 = (Ptank * V_tank) / (R_gas * T_tank);
    n_H2_new = max(n_H2 + (NH2 - qH2) * dt, 0);
    Ptank_new = (n_H2_new * R_gas * T_tank) / V_tank;

    next_state = [SOC_new; Ptank_new];

    % =====================================================================
    % REWARD FUNCTION (The AI's Objective)
    % =====================================================================
    reward = 0;
    reward = reward - (P_mismatch / 1000) * 5; % -5 points for every kW dropped

    if SOC_new < 0.20 || SOC_new > 0.90
        reward = reward - 50; % Penalty for leaving safe SOC bounds
    end

    if action ~= prev_action
        reward = reward - 10; % Penalty for switching states (Chattering)
    end

    is_done = (SOC_new <= 0.01 || SOC_new >= 0.99); % Fatal failure

    % Logging info
    info = struct('Pbat', P_bat, 'Pae', P_ae, 'Pfc', P_fc, 'Pmiss', P_mismatch);
end