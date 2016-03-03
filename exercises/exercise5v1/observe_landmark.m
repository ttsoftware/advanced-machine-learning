% function [landmark_idx, distance, angle] = observe_landmark(true_location, true_orientation, landmarks)
%   Return noisy distance and relative angle from robot to nearest landmark.
%   You do not need to understand what's going on inside this function.

function [landmark_idx, distance, angle] = observe_landmark(true_location, true_orientation, landmarks)
  %% Determine nearest landmark (that's the one we observe)
  all_distances = sqrt(sum(bsxfun(@minus, landmarks, true_location).^2, 2)); % 4x1
  [true_distance, landmark_idx] = min(all_distances);
  lm = landmarks(landmark_idx, :); % 1x2
  
  %% Determine ground truth angle to landmark
  true_global_angle = atan2(lm(2) - true_location(2), lm(1) - true_location(1));
  true_angle = subtract_orientation(true_global_angle, true_orientation);
  
  %% Corrupt measurements by noise
  distance = abs(true_distance + 0.2*randn());
  angle = vmrand(true_angle, 50, 1);
end % function
