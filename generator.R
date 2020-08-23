generator <-
  function(name = NULL) {
    keras_model_custom(name = name, function(self) {

      self$fc1 <- layer_dense(units = 7 * 7 * 64, use_bias = FALSE)
      self$batchnorm1 <- layer_batch_normalization()
      self$leaky_relu1 <- layer_activation_leaky_relu()
      self$conv1 <-
        layer_conv_2d_transpose(
          filters = 64,
          kernel_size = c(5, 5),
          strides = c(1, 1),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm2 <- layer_batch_normalization()
      self$leaky_relu2 <- layer_activation_leaky_relu()
      self$conv2 <-
        layer_conv_2d_transpose(
          filters = 32,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE
        )
      self$batchnorm3 <- layer_batch_normalization()
      self$leaky_relu3 <- layer_activation_leaky_relu()
      self$conv3 <-
        layer_conv_2d_transpose(
          filters = 1,
          kernel_size = c(5, 5),
          strides = c(2, 2),
          padding = "same",
          use_bias = FALSE,
          activation = "tanh"
        )

      function(inputs, mask = NULL, training = TRUE) {
        self$fc1(inputs) %>%
          self$batchnorm1(training = training) %>%
          self$leaky_relu1() %>%
          k_reshape(shape = c(-1, 7, 7, 64)) %>%
          self$conv1() %>%
          self$batchnorm2(training = training) %>%
          self$leaky_relu2() %>%
          self$conv2() %>%
          self$batchnorm3(training = training) %>%
          self$leaky_relu3() %>%
          self$conv3()
      }
    })
  }