# Advanced GAN Experimentation for Complex Image Generation

## Description
Build upon knowledge gained from the DCGAN MNIST project and dive into more complex challenges. The task is to explore and experiment with Generative Adversarial Networks (GANs) in generating images for a more complex dataset.

## Advanced GAN Experimentation for Complex Image Generation

## Dataset:
The CelebFaces Attributes (CelebA) dataset was used on this project since it provided a more complex representation of facial attributes while containing various expressions. It also included colorful representations of those unique identities without providing any names associated with those.

## Github Repo Structure:
 * Configs - contains the yaml file for Weights and Biases experiments.
 * Data - houses the dataset.
 * Experiments - has the notebooks for each experiment and baseline.
 * Logs - images and output files.
 * Models - Baseline and Experiments code.

## Weights and Biases Dashboards:
Images and Charts on Time Taken, Discriminator, and Generator losses:
https://api.wandb.ai/links/vgobin/r7oxll9h

## Takeaways and Observations:
The baseline provided a human shaped face although it was pretty blurry. Time taken was respectable at around 27s per epoch.
 * Architecture Variations (Experiment 1):
Adjusting the number of epochs to 50 caused the total time to more than double but the image quality was improved and the facial expressions really showed up along with more defined features.
 * Hyperparameter Tuning (Experiment2):
Changing the optimizer to RMSProp with a learning rate of 0.0001 resulted in the facial features being less defined than the Adam optimizer. It also took much longer to process at close to 30s per epoch. However, the Discriminator and generator loss was pretty stable showing that the model thought it had a higher confidence.
 * Other Experiments (Experiment 3):
Trying a precision change to int16 didn’t produce any result that was better and I instead used a lower batch size of 16 to experiment with the model. This produced the best time efficiency at less than 15s per epoch while visually showing a respectable image. Although facial features were not fully defined, the discriminator was giving better loss numbers.


## Video Overview:
https://www.loom.com/share/36d0d34d8f0e419491d2bc3bff0a0c9b?sid=79b21325-43c9-4be8-a340-3d8f39a8ad2b
