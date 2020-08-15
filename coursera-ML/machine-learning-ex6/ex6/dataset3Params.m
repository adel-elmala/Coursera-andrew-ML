function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
try_C      = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
try_sigma  = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
C_sigma = zeros(size(try_C,1) * size(try_sigma,1),2);
val_errors = zeros(size(C_sigma,1),1);

for c = 1:size(try_C,1)
  for sig = 1:size(try_sigma,1)
    C_sigma((((c - 1) * 8) + sig),:) = [try_C(c,1) try_sigma(sig,1) ]; 

  endfor
endfor  

for i = 1 : size(C_sigma,1)
  C_sigi = C_sigma(i,:);
  C = C_sigi(1);
  sigma =  C_sigi(2);
  fprintf('Training Model No. %f ...\n',i);

  trained_model = svmTrain(X,y,C,@(x1,x2)gaussianKernel(x1,x2,sigma)); 
  predictions = svmPredict(trained_model, Xval);
  val_errors(i,1) = mean(double(predictions ~= yval));
  fprintf('Model No. %f , error = %f.\n using C = %f , sigma = %f.\n',i,val_errors(i,1),C,sigma);

endfor

[min , mini] = min(val_errors);
fprintf('Selected Model No. %f ,with error = %f.\n',mini,min);

C_sigmar = C_sigma(mini,:);
C = C_sigmar(1);
sigma = C_sigmar(2);



% =========================================================================

end
