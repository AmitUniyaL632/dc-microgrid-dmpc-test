% =========================================================================
% optimizeDutyCycleGWO.m
% Grey Wolf Optimizer for Duty Cycle Optimization
%
% This function finds the optimal duty cycle sequence that minimizes a
% given cost function using the Grey Wolf Optimizer (GWO) metaheuristic.
%
% Inputs:
%   costFunc    : Function handle to the objective/cost function.
%                 The cost function must accept a row vector of duty cycles
%                 and return a scalar cost.
%   dim         : The dimension of the problem (number of duty cycles in a
%                 sequence, i.e., the prediction horizon Np).
%   lowerBound  : Lower bound for each duty cycle (scalar).
%   upperBound  : Upper bound for each duty cycle (scalar).
%   maxIter     : Maximum number of iterations for the optimizer.
%   wolfCount   : Number of search agents (wolves).
%
% Outputs:
%   bestSequence : The optimal sequence of duty cycles found (1 x dim).
%   bestCost     : The minimum cost value corresponding to bestSequence.
% =========================================================================

function [bestSequence, bestCost] = optimizeDutyCycleGWO(costFunc, dim, lowerBound, upperBound, maxIter, wolfCount)

    % Initialize the positions of the alpha, beta, and delta wolves
    Alpha_pos   = zeros(1, dim);
    Alpha_score = inf;
    Beta_pos    = zeros(1, dim);
    Beta_score  = inf;
    Delta_pos   = zeros(1, dim);
    Delta_score = inf;

    % Initialize the wolf pack positions randomly
    Positions = lowerBound + (upperBound - lowerBound) * rand(wolfCount, dim);

    % Main loop
    for iter = 1:maxIter
        for i = 1:wolfCount
            % Return wolves that go outside the search space
            Positions(i,:) = max(Positions(i,:), lowerBound);
            Positions(i,:) = min(Positions(i,:), upperBound);

            % Calculate objective function for each wolf
            fitness = costFunc(Positions(i,:));

            % Update Alpha, Beta, and Delta
            if fitness < Alpha_score, Alpha_score = fitness; Alpha_pos = Positions(i,:); end
            if fitness > Alpha_score && fitness < Beta_score, Beta_score = fitness; Beta_pos = Positions(i,:); end
            if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score, Delta_score = fitness; Delta_pos = Positions(i,:); end
        end

        a = 2 - iter * (2 / maxIter); % Linearly decreasing 'a' from 2 to 0

        % Update the position of the wolves
        for i = 1:wolfCount
            r1 = rand(1, dim); r2 = rand(1, dim); A1 = 2 * a * r1 - a; C1 = 2 * r2;
            D_alpha = abs(C1 .* Alpha_pos - Positions(i,:)); X1 = Alpha_pos - A1 .* D_alpha;

            r1 = rand(1, dim); r2 = rand(1, dim); A2 = 2 * a * r1 - a; C2 = 2 * r2;
            D_beta = abs(C2 .* Beta_pos - Positions(i,:)); X2 = Beta_pos - A2 .* D_beta;

            r1 = rand(1, dim); r2 = rand(1, dim); A3 = 2 * a * r1 - a; C3 = 2 * r2;
            D_delta = abs(C3 .* Delta_pos - Positions(i,:)); X3 = Delta_pos - A3 .* D_delta;

            Positions(i,:) = (X1 + X2 + X3) / 3;
        end
    end

    bestSequence = Alpha_pos;
    bestCost = Alpha_score;

end