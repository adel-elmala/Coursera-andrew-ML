function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X]; # added the bias unit for each example size(5000 * 401)
A1 = X'; # size( 401 * 5000 )
Z2 = Theta1 * A1; # size( 25 ' hidden layer #1 size '  * 5000)
A2 = sigmoid(Z2);
A2 = [ones(1,m) ; A2]; # size( 26 ' hidden layer #1 size '  * 5000)
Z3 = Theta2 * A2; # size( 10 * 5000)
A3 = sigmoid(Z3); # output matrix (each col is an output for each example)

# transform y from vec of labels for each example to matrix of vectors 
u = zeros(m,num_labels); # size(5000 * 10)
for i =1:m
  u(i,:) = vectorize(y(i) , num_labels);
endfor
#theta 2 size ( 10 * 26 )
del3 = A3 - u'; # size (10 * 5000)

del2 = Theta2(:,2:end)' * del3 .* sigmoidGradient(Z2); # size (25 * 5000)

#delta1 = zeros(hidden_layer_size,input_layer_size + 1); # size (25 * 401)
#delta2 = zeros(num_labels, hidden_layer_size + 1); # size (10 * 26 )

delta1 = del2 * A1';

delta2 = del3 * A2';


grad1 = delta1 * (1 / m);
grad2 = delta2 * (1 / m);

grad1 = grad1 + ((lambda / m) .* [zeros(hidden_layer_size,1) Theta1(:,2:end)] );

grad2 = grad2 + ((lambda / m) .* [zeros(num_labels,1) Theta2(:,2:end)] );

grad = [grad1(:) ; grad2(:)];



J = (1 / m) * sum( sum( - ( log(A3) .* u' ) - ( log(1 - A3) .* (1 - u') )  )) ; 


regularizeTerm = (lambda / (2 * m)) * (sum(sum( Theta1(:, 2:end) .^ 2 )) + sum(sum( Theta2(:, 2:end) .^ 2)));

J = J + regularizeTerm ;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
#grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function vec = vectorize(n,k)
  vec = zeros(1,k);
  vec(n) = 1;
  vec;
end

    