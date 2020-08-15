function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
test1 = X(:,1);
test2 = X(:,2);
cpos = 'r';
spos = '+';
cneg = 'b';
sneg = '0';
s = 150;
pos = find(y == 1);
neg = find(y == 0);

scatter(test1(pos,:),test2(pos,:),s,cpos,spos);
scatter(test1(neg,:),test2(neg,:),s,cneg,sneg);








% =========================================================================



hold off;

end
