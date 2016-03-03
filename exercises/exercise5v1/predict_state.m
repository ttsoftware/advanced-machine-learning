% function new_state = predict_state(state, weights, location, orientation, t)
%   Predict new states from old ones as well as measurements of how much
%   the robot moved since last iteration.
%
%   YOU SHOULD IMPLEMENT THIS FUNCTION!

function new_state = predict_state(state, weights, location, orientation, t)
  [num_particles, D] = size(state);
  
  %% First determine true motion
  if (t == 1)
    delta_location = zeros(1, 2); % 1x2
    delta_orientation = 0; % 1x1
  else
    delta_location = location(t, :) - location(t-1, :); % 1x2
    delta_orientation = subtract_orientation(orientation(t), orientation(t-1)); % 1x1
  end % if
  
  %% Add true motion and noise to particles --- XXX: FILL ME IN
  new_location = zeros(size(state));
  for l=1:num_particles
      z = rand();
      cs = cumsum(weights);
      k = find(cs>= z, 1);
      new_location(l,:) =  sample_gaussian(norm(state(k,:) + delta_location),1, 2)'; % (num_particles)x2
  end
  if (D == 3)
    new_orientation = NaN; % (num_particles)x1
  end % if
  
  %% Join location and possibly orientation to form new states
  if (D == 2)
    new_state = new_location; % (num_particles)x2
  else % D == 3
    new_state = [new_location, new_orientation]; % (num_particles)x3
  end % if
end % function
