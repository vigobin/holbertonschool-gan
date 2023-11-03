# Deep Convolutional Generative Adversarial Network (DCGAN)

## Description
Explore and experiment with the training of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images similar to the MNIST dataset.

## Dataset:
The MNIST dataset was used for training since it is well established and a good starting point for the first DCGAN model.

## Github Repo Structure:
 * Configs - contains the yaml file for Weights and Biases experiments.
 * Data - houses the dataset.
 * Experiments - has the notebooks for each experiment and baseline.
 * Logs - images and output files.
 * Models - Baseline and Experiments code.

## Weights and Biases Dashboards:
Images and Charts on Discriminator and Generator losses:
https://api.wandb.ai/links/vgobin/4ht5i9bw


## Takeaways and Observations:
The baseline provided some readable numbers but was still pretty blurry while training time was not bad at around 11s per epoch.
 * Architecture Variations (Experiment 1):
Adjusting the number of epochs to 20 caused the total time to more than double but the image quality was improved and more defined.
 * Hyperparameter Tuning:
The most visible change was adjusting the batch size to 128. This caused the generated images to be visually more defined and streamlined and the charts on W&B also showed the Discriminator gave better loss results. Time definitely increased on processing this update though.
 * Precision Changes:
Trying a precision change to float16 caused much less defined numbers although the model somehow was able to generate some readable numbers.

## Video Overview :
https://www.loom.com/share/09c1ea3dca0e498b88ae06fc1540f84c?sid=1e16d947-6dce-4ef2-a39d-1dd1e40284d7

