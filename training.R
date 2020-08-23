# training loop
train <- function(dataset, epochs, noise_dim) {
  for (epoch in seq_len(num_epochs)) {
    start <- Sys.time()
    total_loss_gen <- 0
    total_loss_disc <- 0
    iter <- make_iterator_one_shot(train_dataset)

    until_out_of_range({
      batch <- iterator_get_next(iter)
      noise <- k_random_normal(c(batch_size, noise_dim))
      with(tf$GradientTape() %as% gen_tape, { with(tf$GradientTape() %as% disc_tape, {
        generated_images <- generator(noise)
        disc_real_output <- discriminator(batch, training = TRUE)
        disc_generated_output <-
          discriminator(generated_images, training = TRUE)
        gen_loss <- generator_loss(disc_generated_output)
        disc_loss <-
          discriminator_loss(disc_real_output, disc_generated_output)
      }) })

      gradients_of_generator <-
        gen_tape$gradient(gen_loss, generator$variables)
      gradients_of_discriminator <-
        disc_tape$gradient(disc_loss, discriminator$variables)

      generator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_generator, generator$variables)
      ))
      discriminator_optimizer$apply_gradients(purrr::transpose(
        list(gradients_of_discriminator, discriminator$variables)
      ))

      total_loss_gen <- total_loss_gen + gen_loss
      total_loss_disc <- total_loss_disc + disc_loss

    })

    cat("Time for epoch ", epoch, ": ", Sys.time() - start, "\n")
    cat("Generator loss: ", total_loss_gen$numpy() / batches_per_epoch, "\n")
    cat("Discriminator loss: ", total_loss_disc$numpy() / batches_per_epoch, "\n\n")

    # Generate pgn file of generated images every 10 epochs
    if (epoch %% 10 == 0)
      generate_and_save_images(generator,
                               epoch,
                               random_vector_for_generation)
  }
}