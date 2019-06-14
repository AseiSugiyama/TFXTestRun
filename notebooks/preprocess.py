import tensorflow as tf
import tensorflow_transform as transform
from tfx.examples.chicago_taxi.trainer import taxi


def _fill_in_missing(x):
  """Replace missing values in a SparseTensor.
  Fills in missing values of `x` with '' or 0, and converts to a dense tensor.
  Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.
  Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
  """
  default_value = '' if x.dtype == tf.string else 0
  return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value),
      axis=1)


def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.
  Args:
    inputs: map from feature keys to raw not-yet-transformed features.
  Returns:
    Map from string feature key to transformed feature operations.
  """
  outputs = {}
  for key in taxi.DENSE_FLOAT_FEATURE_KEYS:
    # Preserve this feature as a dense float, setting nan's to the mean.
    outputs[taxi.transformed_name(key)] = transform.scale_to_z_score(
        _fill_in_missing(inputs[key]))

  for key in taxi.VOCAB_FEATURE_KEYS:
    # Build a vocabulary for this feature.
    outputs[
        taxi.transformed_name(key)] = transform.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=taxi.VOCAB_SIZE,
            num_oov_buckets=taxi.OOV_SIZE)

  for key in taxi.BUCKET_FEATURE_KEYS:
    outputs[taxi.transformed_name(key)] = transform.bucketize(
        _fill_in_missing(inputs[key]), taxi.FEATURE_BUCKET_COUNT)

  for key in taxi.CATEGORICAL_FEATURE_KEYS:
    outputs[taxi.transformed_name(key)] = _fill_in_missing(inputs[key])

  # Was this passenger a big tipper?
  taxi_fare = _fill_in_missing(inputs[taxi.FARE_KEY])
  tips = _fill_in_missing(inputs[taxi.LABEL_KEY])
  outputs[taxi.transformed_name(taxi.LABEL_KEY)] = tf.where(
      tf.is_nan(taxi_fare),
      tf.cast(tf.zeros_like(taxi_fare), tf.int64),
      # Test if the tip was > 20% of the fare.
      tf.cast(
          tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))),
          tf.int64))

  return outputs