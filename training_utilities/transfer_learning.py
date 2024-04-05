from tensorflow_hub import KerasLayer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def build_feature_extractor_model_from_url(model_url, input_shape):
    """
    Creates a TensorFlow model with a pre-trained feature extractor from the provided URL.
    Ensure `input_shape` matches the expected format,
    e.g., for RGB images, use `(224, 224, 3)`.

    Example:
    model = build_feature_extractor_model_from_url("model_url_here", input_shape=(224, 224, 3))

    Parameters:
    - model_url (str): URL to a TensorFlow Hub model.
    - input_shape (tuple): Expected shape of the input data.

    Returns:
    A Sequential model with only the feature extraction layer.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor = KerasLayer(model_url,
                                   trainable=False,  # freeze the already learned patterns
                                   name='feature_extraction_layer',
                                   input_shape=input_shape)

    # Create the output layer
    model = Sequential()
    model.add(feature_extractor)

    return model


def build_complete_model_from_url(model_url, num_classes, input_shape, output_activation="softmax"):
    """
    Constructs a TensorFlow model with a pre-trained feature extractor and an output layer.
    Set `input_shape` appropriately,
    e.g., `(224, 224, 3)` for color images.

    Example:
    model = build_complete_model_from_url("model_url_here", num_classes=10, input_shape=(224, 224, 3))

    Parameters:
    - model_url (str): URL to a TensorFlow Hub model.
    - num_classes (int): Number of classes for the output layer.
    - input_shape (tuple): Expected shape of the input data.
    - output_activation (str): Activation function for the output layer.

    Returns:
    A Sequential model including the feature extraction and output layers.
    """
    # Download the pretrained model and save it as a Keras layer
    feature_extractor = KerasLayer(model_url,
                                   trainable=False,  # freeze the already learned patterns
                                   name='feature_extraction_layer',
                                   input_shape=input_shape)

    # Create the output layer
    model = Sequential()
    model.add(feature_extractor)
    model.add(Dense(units=num_classes, activation=output_activation, name="output_layer"))

    return model


def configure_trainable_layers(model, n_trainable_layers, verbose:bool=False):
    """
    Configures the layers of a given model such that all layers except for the
    last 'n_trainable_layers' are frozen (i.e., set to non-trainable).

    This function sets the entire model as trainable first, then iterates through
    the layers and freezes each one except for the specified number of layers from the end.
    If 'verbose' is True, it prints the 'trainable' status of each layer.

    Parameters:
    - model: A TensorFlow/Keras model to be configured.
    - n_trainable_layers: Integer specifying the number of layers to keep trainable
      at the end of the model.
    - verbose: Boolean, if set to True, the function will print the name and 'trainable'
      status of each layer.

    Returns:
    - None, the modifications are applied in-place on the provided model.
    """
    model.trainable = True
    for layer in model.layers[:-n_trainable_layers]:
        layer.trainable = False
    if verbose:
        for i, layer in enumerate(model.layers):
            print(f"Layer {i+1}: {layer.name}, Trainable: {layer.trainable}")