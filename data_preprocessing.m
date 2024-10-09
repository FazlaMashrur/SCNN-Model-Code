
% data_preprocessing.m
% This function performs data preprocessing on the ECG dataset, including noise removal and segmentation.

function processed_data = data_preprocessing(raw_data)
    % Apply filtering to remove noise
    Fs = 40.9; % Sampling frequency
    % Define Chebyshev Type II bandpass filter from 0.5 Hz to 48 Hz
    [b, a] = cheby2(4, 20, [0.5, 48] / (Fs / 2), 'bandpass');
    processed_data = filtfilt(b, a, raw_data);

    % Segment the ECG signal into epochs of 60 seconds
    epoch_length = 60 * Fs;  % Number of samples per epoch
    num_epochs = floor(length(processed_data) / epoch_length);
    processed_data = reshape(processed_data(1:num_epochs*epoch_length), epoch_length, num_epochs);
    
    % Additional noise removal
    % Identify noisy epochs based on amplitude thresholds
    median_max = median(max(processed_data, [], 1));
    median_min = median(min(processed_data, [], 1));
    threshold_factor = 2;
    noisy_epochs = any(processed_data > threshold_factor * median_max | processed_data < threshold_factor * median_min, 1);
    processed_data(:, noisy_epochs) = [];
end
