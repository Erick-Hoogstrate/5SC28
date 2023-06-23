%% pendulum_learn.m
% *Summary:* Script to learn a controller for the pendulum swingup
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-27
%
%% High-Level Steps
% # Load parameters
% # Create J initial trajectories by applying random controls
% # Controlled learning (train dynamics model, policy learning, policy
% application)

%% Code
% 0. Indicate policy/cost specific parameters
cost_targets = [[0; 170*(pi/180)], [0; pi], [0; 190*(pi/180)]];

% Train the policies after eachother
for policy_i = 1:3
    % 1a. General initialization
    close all
    clearvars -except policy_i learned_policies cost_targets
    
    multi_settings_pendulum;
    basename = ['multi_pendulum_p' num2str(policy_i)];

    % 1b. Policy/cost specific initializaion
    cost.target = cost_targets(:, policy_i);

    % 2. Initial J random rollouts
    for jj = 1:J
      [xx, yy, realCost{jj}, latent{jj}] = ...
        rollout(gaussian(mu0, S0), struct('maxU', policy.maxU), H, plant, cost);
      x = [x; xx]; y = [y; yy];       % augment training sets for dynamics model
      if plotting.verbosity > 0;      % visualization of trajectory
        if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
        multi_draw_rollout_pendulum;
      end
    end
    
    mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
    mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);
    
    % 3. Controlled learning (N iterations)
    for j = 1:N
      trainDynModel;   % train (GP) dynamics model
      learnPolicy;     % learn policy
      applyController; % apply controller to system
      disp(['controlled trial # ' num2str(j)]);
      if plotting.verbosity > 0;      % visualization of trajectory
        if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
        multi_draw_rollout_pendulum;
      end
    end

    % 4. Save the learned policy
    learned_policies(policy_i).policy = policy;
end

%% Visualize learned policies
disp(['Policy performance after ' num2str(j) ' trials']);
% Select initial state
mu0 = [0 0]';                  
S0 = 0.01*eye(2);

% Select target
target_state = 1;    % 1 = 10 degree right, 2 = exactly upright, 3 = 10 degree leftt
policy = learned_policies(target_state).policy;

% Load the final trial of that policy
%load(['multi_pendulum_' num2str(target_state) '_5_H160.mat'])

% Animate results
multi_draw_rollout_pendulum;
