clear all
close all

%% Define landmark positions (this is our map or the world)
landmarks = [-1, -1; ...
             -1,  1; ...
              1, -1; ...
              1,  1]; % 4x2

%% Define the motion the robot actually takes
%% This is the location we want to estimate
%% You don't need to understand this part of the code!
num_time_steps = 500;
T = linspace(5*pi, pi, num_time_steps).';
location = 2.*[T.*cos(T), T.*sin(T)]./max(T); % (num_time_steps)x2
[~, orientation] = gradient(location);  % (num_time_steps)x2
orientation = atan2(orientation(:, 2), orientation(:, 1)); % (num_time_steps)x1

%% Parameters for the particle filter
num_particles = 5000;

%% Initial state of the particle filter;
%% state = (location), i.e. in R^2 -- if you move on to estimating orientation, then D=3
D = 2;
state(:, 1) = 6*rand(num_particles, 1) - 3;     % in [-3, 3]
state(:, 2) = 6*rand(num_particles, 1) - 3;     % in [-3, 3]
%state(:, 3) = 2*pi*rand(num_particles, 1) - pi; % in [-pi, pi]
weights = ones(1, num_particles)/num_particles; % (num_particles)x1

%% Iterate across time
for t = 1:num_time_steps
  %% Extract the true position (we do not use this to determine the robot position)
  true_location = location(t, :); % 1x2
  true_orientation = orientation(t); % 1x1

  %%%%% PREDICT
  %% The robot measures its relative motion since the last time step.
  %% This measurement is subject to noise.
  state = predict_state(state, weights, location, orientation, t); % XXX: YOU NEED TO CHANGE THiS FUNCTION!

  %%%%% MEASURE
  [landmark_idx, distance, angle] = observe_landmark(true_location, true_orientation, landmarks);
  lm = landmarks(landmark_idx, :); % 1x2
  for n=1:num_particles
      weights(n,1) = mvnpdf(distance,norm(state(n,:)-lm), 1); % (num_particles)x1 --- XXX: DO THIS CORRECTLY
  end
  
  weights = weights / sum(weights); % (num_particles)x1

  %% Compute current state mean
  mean_pos = weights.' * state(:, 1:2);
  %mean_ori = angle_mean(state(:, 3), weights);

  %% Plot what's going on
  clf
  if (D == 2)
    plot_robot(state(:, 1:2), NaN, 'r');
  else % D == 3
    plot_robot(state(:, 1:2), state(:, 3), 'r');
  end % if
  hold on
  plot_robot(true_location, true_orientation, [0, 1, 0]);
  plot(landmarks(:, 1), landmarks(:, 2), 'ko', 'markerfacecolor', 'k', 'markersize', 10);
  if (D == 2)
    plot_robot(mean_pos, NaN, 'b');
  else % D == 3
    plot_robot(mean_pos, mean_ori, 'b');
  end % if
  hold off
  axis([-2.2, 2.2, -2.2, 2.2]);
  pause(0.05)
end % for

