% function plot_robot(location, orientation, colors)
%   Plot a robot in a given color.
%
%   The input should have dimensions
%     location:    Nx2
%     orientation: Nx1
%     colors:      Nx3

function plot_robot(location, orientation, color)
  %% Check input
  if (nargin < 3)
    error('plot_robot: not enough input arguments');
  end % if
  
  %% Plot the robots
  hold_status = ishold();
  hold on
  len = 0.3;
  plot(location(:, 1), location(:, 2), 'o', 'color', color, 'markerfacecolor', color);
  if (~isnan(orientation))
    ax = location(:, 1); bx = location(:, 1) + len * cos(orientation); cx = NaN(size(ax));
    ay = location(:, 2); by = location(:, 2) + len * sin(orientation); cy = NaN(size(ay));
    x = [ax(:), bx(:), cx(:)].';
    y = [ay(:), by(:), cy(:)].';
    plot(x(:), y(:), 'color', color);
  end % if
  if (~hold_status)
    hold off
  end % if
end % function