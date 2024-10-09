
# SCNN-Based Obstructive Sleep Apnea Detection Using Single-Lead ECG Signals

This repository provides a modular implementation of the scalogram-based convolutional neural network (SCNN) model for 
detecting obstructive sleep apnea using single-lead ECG signals, based on the publication:

**Mashrur, F.R., Islam, M.S., Saha, D.K., & Moni, M.A. (2021). SCNN: Scalogram-based convolutional neural network to 
detect obstructive sleep apnea using single-lead electrocardiogram signals. Computers in Biology and Medicine, 134, 
104532.** https://doi.org/10.1016/j.compbiomed.2021.104532

## Repository Structure

- **data_preprocessing.m**: Prepares and cleans the ECG dataset, removing noisy segments.
- **split_dataset.m**: Splits the data into training, validation, and test sets.
- **generate_scalograms.m**: Generates conventional and hybrid scalograms from ECG signals using continuous wavelet transform.
- **scnn_model.m**: Defines the CNN architecture tailored for scalogram input.
- **train_model.m**: Trains the SCNN model with the training dataset.
- **evaluate_model.m**: Evaluates model performance on validation and test datasets, calculating accuracy, sensitivity, specificity, and F1-score.
- **main.m**: Executes the complete pipeline by sequentially calling the above scripts.

## Setup and Usage

1. **Requirements**: MATLAB 2020a or later.
2. **Data**: Place your ECG dataset files in the `data` directory before running.
3. **Execution**: Run the main script:
   ```matlab
   main
   ```
   This will process the data, generate scalograms, train the SCNN, and display the evaluation metrics.

## Results

The model achieves high accuracy and reliability in detecting obstructive sleep apnea by leveraging time-frequency 
information in ECG signals. Detailed results can be compared to those published in the reference paper.

## Citation

If you use this code in your work, please cite:
```plaintext
@article{Mashrur2021SCNN,
  title = {{SCNN}: Scalogram-based convolutional neural network to detect obstructive sleep apnea using single-lead 
           electrocardiogram signals},
  author = {Mashrur, Fazla Rabbi and Islam, Md. Saiful and Saha, Debasish Kumar and Moni, Mohammad Ali},
  journal = {Computers in Biology and Medicine},
  volume = {134},
  pages = {104532},
  year = {2021},
  publisher = {Elsevier},
  doi = {10.1016/j.compbiomed.2021.104532}
}
```

## License

This code is provided for academic research purposes under the MIT License.
