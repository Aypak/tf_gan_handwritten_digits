# Loss functions

# discriminator loss
# Does it correctly identify real images as real
# and does it correctly spot fake images as fake
# real output, generated output - logits returned from discriminator
discriminator_loss <- function(real_output, generated_output) {
  real_loss <- tf$compat$v1$losses$sigmoid_cross_entropy(
    multi_class_labels = k_ones_like(real_output),
    logits = real_output)
  generated_loss <- tf$compat$v1$losses$sigmoid_cross_entropy(
    multi_class_labels = k_zeros_like(generated_output),
    logits = generated_output)
  real_loss + generated_loss
}


# generator loss
# how discriminator judged generator's creations
generator_loss <- function(generated_output) {
  tf$compat$v1$losses$sigmoid_cross_entropy(
    tf$ones_like(generated_output),
    generated_output)
}
