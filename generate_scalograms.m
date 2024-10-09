
% generate_scalograms.m
% This function converts ECG data into scalograms using continuous wavelet transform (CWT).

function scalograms = generate_scalograms(data, Fs)
    % Number of epochs
    num_epochs = size(data, 2);
    % Initialize the output variable
    scalograms = zeros(32, 32, 3, num_epochs);

    for i = 1:num_epochs
        signal = data(:, i);
        % Create a wavelet filter bank
        fb = cwtfilterbank('SignalLength', length(signal), 'SamplingFrequency', Fs, 'Wavelet', 'amor', 'VoicesPerOctave', 48);
        % Compute the CWT
        [wt, ~] = fb.wt(signal);
        % Convert to scalogram image
        scalogram = abs(wt);
        scalogram = imresize(scalogram, [32, 32]);
        % Convert to RGB using MATLAB's colormap
        scalograms(:, :, :, i) = ind2rgb(im2uint8(rescale(scalogram)), jet(256));
    end
end
