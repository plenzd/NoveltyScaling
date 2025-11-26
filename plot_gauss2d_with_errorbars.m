function plot_gauss2d_with_errorbars(points, sigma_shade)
    if nargin < 2, sigma_shade=2.5, end
    mlmean = mean(points, 1);
    mlcov = cov(points);
    hold on;
    % plot data points
    pointsize=30
    h_points = scatter(points(:,1), points(:,2), pointsize, 'red', 'filled', 'MarkerFaceAlpha', 0.75);
    % get and plot elipse (with or wothout edge line!?)
    [ellipse_x, ellipse_y] = get_conf_ellipse_points(mlmean, mlcov, sigma_shade, 100);
%    h_ellipse = fill(ellipse_x, ellipse_y, 'blue', 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'LineWidth', 0.5);
    h_ellipse = fill(ellipse_x, ellipse_y, 'blue', 'FaceAlpha', 0.3, 'EdgeColor', 'blue', 'LineWidth', 0.5);
    [eigvecs, eigvals] = eig(mlcov);
    eigvals = diag(eigvals);
    [eigvals, iorder] = sort(eigvals, 'descend');
    eigvecs = eigvecs(:, iorder);
    majoraxlen = sqrt(eigvals(1)); %
    minoraxlen = sqrt(eigvals(2));
    majorax = eigvecs(:, 1);
    minorax = eigvecs(:, 2);
    major_start = mlmean - majoraxlen * majorax';
    major_end = mlmean + majoraxlen * majorax';
    
% plotting lines "caps" as error-bars in your plots (along major/minor axes)-- adjust widths colors as it suits you!!
    h_major = plot([major_start(1), major_end(1)], [major_start(2), major_end(2)], 'k-', 'LineWidth', 1);
    minor_start = mlmean - minoraxlen * minorax';
    minor_end = mlmean + minoraxlen * minorax';
    h_minor = plot([minor_start(1), minor_end(1)], [minor_start(2), minor_end(2)], 'k-', 'LineWidth', 1);
    cap_size = 0.1 * max(majoraxlen, minoraxlen);
    major_perp = [-majorax(2), majorax(1)];
    plot([major_start(1) - cap_size*major_perp(1), major_start(1) + cap_size*major_perp(1)], ...
         [major_start(2) - cap_size*major_perp(2), major_start(2) + cap_size*major_perp(2)], ...
         'k-', 'LineWidth', 1.5);
    plot([major_end(1) - cap_size*major_perp(1), major_end(1) + cap_size*major_perp(1)], ...
         [major_end(2) - cap_size*major_perp(2), major_end(2) + cap_size*major_perp(2)], ...
         'k-', 'LineWidth', 1.5);
    minor_perp = [-minorax(2), minorax(1)];  % Perpendicular to minor ax
    plot([minor_start(1) - cap_size*minor_perp(1), minor_start(1) + cap_size*minor_perp(1)], ...
         [minor_start(2) - cap_size*minor_perp(2), minor_start(2) + cap_size*minor_perp(2)], ...
         'k-', 'LineWidth', 1.5);
    plot([minor_end(1) - cap_size*minor_perp(1), minor_end(1) + cap_size*minor_perp(1)], ...
         [minor_end(2) - cap_size*minor_perp(2), minor_end(2) + cap_size*minor_perp(2)], ...
         'k-', 'LineWidth', 1.5);
    h_mean = plot(mlmean(1), mlmean(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'white', 'LineWidth', 2);
    axis equal;
    axis off;
    hold off;
    saveas(gcf, 'fig7subpanel.pdf')
end

function [ellipse_x, ellipse_y] = get_conf_ellipse_points(mean_vec, cov_matrix, n_sigma, n_points)
    
    [eigenvecs, eigenvals] = eig(cov_matrix);
    eigenvals = diag(eigenvals);
    
    [eigenvals, order] = sort(eigenvals, 'descend');
    eigenvecs = eigenvecs(:, order);
    
    angle = atan2(eigenvecs(2,1), eigenvecs(1,1));
    width = 2 * n_sigma * sqrt(eigenvals(1));
    height = 2 * n_sigma * sqrt(eigenvals(2));
    
    theta = linspace(0, 2*pi, n_points);
    ellipse_x_unit = (width/2) * cos(theta);
    ellipse_y_unit = (height/2) * sin(theta);
    cos_angle = cos(angle);
    sin_angle = sin(angle);
    
    ellipse_x = cos_angle * ellipse_x_unit - sin_angle * ellipse_y_unit + mean_vec(1);
    ellipse_y = sin_angle * ellipse_x_unit + cos_angle * ellipse_y_unit + mean_vec(2);
end
