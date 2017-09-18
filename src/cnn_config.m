function [ output ] = cnn_config( input, kernels )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
for i = 1:size(kernels,1)
    output(i,:) = (input - kernels(i,:) + 1) ./ 2
    input = output(i,:);
end

end

