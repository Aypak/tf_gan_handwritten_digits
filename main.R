# Load libraries
library(keras)
library(tensorflow)
library(tfdatasets)

# Use tensorflow for keras
use_implementation("tensorflow")
# enable eager execuction
tfe_enable_eager_execution(device_policy = "silent")

# load mnist dataset
mnist <- dataset_mnist()
# use %<-% to assign multiple variables from list
c(train_images, train_labels) %<-% mnist$train

# Expand the training set and cast all the values to float 32
train_images <- train_images %>%
  k_expand_dims() %>%
  k_cast(dtype = "float32")

# normalize the values to between -1 and 1
# because generator uses tanh activation (looks like sigmoid but between -1 and 1)
# 127.5 because max value is 256 and min value is 0
train_images <- (train_images - 127.5)/127.5

# Get the number of training observations to be used in each train step
# total number of training obs
buffer_size <- 60000
# number we want in each batch
batch_size <- 256
# number of batches in each epoch (train step)
batches_per_epoch <- (buffer_size / batch_size) %>% round()

# shuffle the dataset
# slice into batches based on the desired batch size set above
# used for the discriminator only
train_dataset <- tensor_slices_dataset(train_images) %>%
  dataset_shuffle(buffer_size) %>%
  dataset_batch(batch_size)

# Source generator and discriminator
source("generator.R")
source("discriminator.R")
generator <- generator()
discriminator <- discriminator()

# Use eager execution in the generator and discriminator calls
generator$call = tf$contrib$eager$defun(generator$call)
discriminator$call = tf$contrib$eager$defun(discriminator$call)
source("loss_fns.R")

# Optimizers
#discriminator_optimizer <- tf$train$AdamOptimizer(1e-4)
discriminator_optimizer <- tf$compat$v1$train$AdamOptimizer(1e-4)
#generator_optimizer <- tf$train$AdamOptimizer(1e-4)
generator_optimizer <- tf$compat$v1$train$AdamOptimizer(1e-4)

# Number of epochs to train
num_epochs <- 20

# Number of examples to generate
num_examples_to_generate <- 25L

# noise to give generator
noise_dim <- 100
random_vector_for_generation <-
  k_random_normal(c(num_examples_to_generate,
                    noise_dim))

# Source the function to save generated images
source("generate_and_save_imgs.R")

# Source the main training function
source("training.R")

# Start training
train(train_dataset, num_epochs, noise_dim)
