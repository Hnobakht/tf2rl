import numpy as np
import tensorflow as tf


class VAE(tf.keras.Model):
    """Variational autoencoder using supplied inference and generative networks."""
    def __init__(self, latent_dim, inference_net, generative_net):
        """Attach the inference and generative networks and set the latent dimension.

        Args:
            latent_dim (int): Number of latent coordinates used by the variational autoencoder.
            inference_net (keras.Model): Encoder producing latent means and log variances.
            generative_net (keras.Model): Decoder mapping latent samples to reconstruction logits.

        Returns:
            None: Initializes the instance.
        """
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.inference_net = inference_net
        self.generative_net = generative_net

    @tf.function
    def sample(self, eps=None):
        """Decode supplied latent vectors or 100 standard-normal samples into generated
        observations.

        Args:
            eps (tf.Tensor or None): Latent inputs shaped (batch, latent_dim); None draws 100 standard-
                normal samples.

        Returns:
            tf.Tensor: Decoder probabilities generated from the supplied latent noise, or 100 standard-
                normal latent samples by default.
        """
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        """Return the latent Gaussian mean and log variance for the input batch.

        Args:
            x (array_like): Input samples; leading dimensions index samples and the remaining dimensions
                contain features.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Batched latent means and log variances, each with final
                dimension latent_dim.
        """
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """Sample latent vectors from Gaussian parameters using differentiable
        reparameterization.

        Args:
            mean (tf.Tensor): Latent Gaussian means aligned with logvar and the sampled latent
                coordinates.
            logvar (tf.Tensor): Latent log variances aligned with the latent means.

        Returns:
            tf.Tensor: Differentiable Gaussian latent samples with the same shape as mean.
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """Map latent vectors to decoder logits, optionally applying a sigmoid.

        Args:
            z (tf.Tensor): Latent samples shaped (batch, latent_dim).
            apply_sigmoid (bool): Convert decoder logits to probabilities.

        Returns:
            tf.Tensor: Decoder logits, or probabilities when apply_sigmoid is True, retaining the batch
                axis.
        """
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    @tf.function
    def compute_loss(self, x):
        """Return the Monte Carlo negative evidence lower bound for the input batch.

        Args:
            x (array_like): Input samples; leading dimensions index samples and the remaining dimensions
                contain features.

        Returns:
            tf.Tensor: Scalar negative evidence lower bound, combining reconstruction, prior, and
                posterior log-density terms.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def compute_apply_gradients(self, x):
        """Compute the variational loss and apply one optimizer update.

        Args:
            x (array_like): Input samples; leading dimensions index samples and the remaining dimensions
                contain features.

        Returns:
            None.
        """
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """Return diagonal Gaussian log densities summed along the requested axis.

    Args:
        sample (tf.Tensor): Samples at which to evaluate the diagonal Gaussian log density.
        mean (tf.Tensor): Latent Gaussian means aligned with logvar and the sampled latent
            coordinates.
        logvar (tf.Tensor): Latent log variances aligned with the latent means.
        raxis (int): Axis over which independent log-density terms are summed.

    Returns:
        tf.Tensor: Diagonal-Gaussian log density summed over raxis; that axis is removed from the
            result.
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


if __name__ == "__main__":
    import time
    import glob
    import matplotlib.pyplot as plt
    import PIL
    import imageio

    (train_images, _), (test_images, _) = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

    # Normalizing the images to the range of [0., 1.]
    train_images /= 255.
    test_images /= 255.

    TRAIN_BUF = 60000
    BATCH_SIZE = 100

    TEST_BUF = 10000

    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)

    epochs = 100
    latent_dim = 50
    num_examples_to_generate = 16

    inference_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    generative_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ]
    )

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = VAE(latent_dim, inference_net, generative_net)


    def generate_and_save_images(model, epoch, test_input):
        """Decode test latent vectors and save the image grid for this epoch.

        Args:
            model (VAE): Decoder model exposing sample(test_input) for reconstruction image generation.
            epoch (int): Training epoch used in the generated image filename.
            test_input (array_like): Latent test samples passed to the image-generating model.

        Returns:
            None.
        """
        predictions = model.sample(test_input)
        plt.close()
        plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


    generate_and_save_images(model, 0, random_vector_for_generation)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            model.compute_apply_gradients(train_x)
        end_time = time.time()

        if epoch % 1 == 0:
            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(model.compute_loss(test_x))
            elbo = -loss.result()
            print('Epoch: {}, Test set ELBO: {}, '
                  'time elapse for current epoch {}'.format(epoch,
                                                            elbo,
                                                            end_time - start_time))
            generate_and_save_images(
                model, epoch, random_vector_for_generation)


    def display_image(epoch_no):
        """Open the generated image saved for the requested epoch.

        Args:
            epoch_no (int): Epoch number identifying the saved reconstruction image.

        Returns:
            PIL.Image.Image: Image opened from the epoch-specific reconstruction PNG; the caller owns
                the image resource.
        """
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


    anim_file = 'vae.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob('image*.png')
        filenames = sorted(filenames)
        last = -1
        for i, filename in enumerate(filenames):
            frame = 2 * (i ** 0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
