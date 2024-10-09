
% split_dataset.m
% This function splits the preprocessed data into training, validation, and test sets.

function [train_data, val_data, test_data] = split_dataset(data, train_ratio, val_ratio)
    % Calculate the number of samples for each set
    num_samples = size(data, 2);
    train_end = floor(num_samples * train_ratio);
    val_end = floor(num_samples * (train_ratio + val_ratio));

    % Split the data
    train_data = data(:, 1:train_end);
    val_data = data(:, train_end+1:val_end);
    test_data = data(:, val_end+1:end);
end
