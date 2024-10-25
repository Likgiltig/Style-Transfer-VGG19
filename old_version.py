import tensorflow as tf
import numpy as np
from PIL import Image
import os

def load_img(path_to_img, max_dim):
    """
    Load and preprocess an image for style transfer.
    
    This function performs the following steps:
    1. Opens the image file
    2. Removes the alpha channel if present
    3. Resizes the image while maintaining aspect ratio, with the longest side equal to max_dim
    4. Converts the image to a numpy array
    5. Normalizes the pixel values to be between 0 and 1
    6. Adds a batch dimension to the array
    
    Args:
    path_to_img (str): Path to the image file
    max_dim (int): Maximum dimension of the output image
    
    Returns:
    numpy.ndarray: Preprocessed image array
    """
    print(f"Loading and preprocessing image: {path_to_img}")
    img = Image.open(path_to_img)
    
    # Remove alpha channel if present
    if img.mode == 'RGBA':
        print("Removing alpha channel...")
        img = img.convert('RGB')
    
    # Calculate the scaling factor
    long = max(img.size)
    scale = max_dim / long
    
    # Calculate new dimensions
    new_width = int(img.size[0] * scale)
    new_height = int(img.size[1] * scale)
    
    img = img.resize((new_width, new_height), Image.LANCZOS)
    img = np.array(img)
    
    # Normalize
    img = img / 255.0
    
    print(f"Image preprocessed. Shape: {img.shape}")
    return img[np.newaxis, ...]
    
    print(f"Image preprocessed. Shape: {img.shape}")
    return img[np.newaxis, ...]

def vgg_layers(layer_names):
    """
    Creates a vgg model that returns a list of intermediate output values.
    
    This function creates a new model that outputs the activations of specified layers
    from the pre-trained VGG19 model.
    
    Args:
    layer_names (list): List of layer names to extract outputs from
    
    Returns:
    tensorflow.keras.Model: Model that outputs specified layer activations
    """
    print("Creating VGG model with specified output layers...")
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """
    Calculate Gram Matrix of a given tensor.
    
    The Gram matrix is calculated by reshaping the input tensor and computing
    the dot product with its transpose. This captures the correlations between
    different features in the input, which is useful for representing style.
    
    Args:
    input_tensor (tensorflow.Tensor): Input tensor
    
    Returns:
    tensorflow.Tensor: Gram matrix of the input tensor
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

class StyleContentModel(tf.keras.models.Model):
    """
    A model that extracts style and content features from an input image.
    
    This model uses the pre-trained VGG19 to extract features from specified
    style and content layers. It also computes the Gram matrix for style features.
    """
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        print("Initializing StyleContentModel...")
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        """
        Process the input through the VGG19 model and extract style and content features.
        
        Args:
        inputs (tensorflow.Tensor): Input image tensor
        
        Returns:
        dict: Dictionary containing style and content features
        """
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}

def clip_0_1(image):
    """
    Clip the pixel values of an image to be between 0 and 1.
    
    Args:
    image (tensorflow.Tensor): Input image tensor
    
    Returns:
    tensorflow.Tensor: Clipped image tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    """
    Calculate the total loss for style transfer.
    
    This function computes both the style loss and the content loss, and combines
    them using the provided weights.
    
    Args:
    outputs (dict): Output features from the style content model
    style_targets (dict): Target style features
    content_targets (dict): Target content features
    style_weight (float): Weight for style loss
    content_weight (float): Weight for content loss
    
    Returns:
    tensorflow.Tensor: Total loss
    """
    print("Calculating style and content loss...")
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image, extractor, optimizer, style_targets, content_targets, style_weight, content_weight):
    """
    Perform a single training step for style transfer.
    
    This function computes the loss and applies gradients to update the image.
    It's decorated with @tf.function for improved performance.
    
    Args:
    image (tensorflow.Variable): The image being optimized
    extractor (StyleContentModel): Model to extract style and content features
    optimizer (tensorflow.optimizers.Optimizer): The optimizer for updating the image
    style_targets (dict): Target style features
    content_targets (dict): Target content features
    style_weight (float): Weight for style loss
    content_weight (float): Weight for content loss
    """
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.
    
    This function takes a tensor representation of an image and converts it
    back to a PIL Image object, ready for saving or display.
    
    Args:
    tensor (tensorflow.Tensor): Input image tensor
    
    Returns:
    PIL.Image: Image object
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def run_style_transfer(content_path, style_path, num_epochs=10, content_weight=1e4, style_weight=1e-2, max_dim=512):
    """
    Run the style transfer algorithm.
    
    This function performs the following steps:
    1. Load and preprocess the content and style images
    2. Set up the style transfer model and optimizer
    3. Iterate for the specified number of epochs, updating the image at each step
    4. Save the result of each epoch
    
    Args:
    content_path (str): Path to the content image
    style_path (str): Path to the style image
    num_epochs (int): Number of epochs to run the optimization
    content_weight (float): Weight for content loss
    style_weight (float): Weight for style loss
    max_dim (int): Maximum dimension of the processed images
    
    Returns:
    tensorflow.Variable: The final stylized image
    """
    print(f"Starting style transfer process for {num_epochs} epochs...")
    content_image = load_img(content_path, max_dim)
    style_image = load_img(style_path, max_dim)

    extractor = StyleContentModel(style_layers, content_layers)

    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    for i in range(num_epochs):
        print(f"Starting epoch {i+1}/{num_epochs}")
        train_step(image, extractor, opt, style_targets, content_targets, style_weight, content_weight)
        
        if i % 1 == 0:
            print(f"\tSaving result for epoch {i+1}")
            img = tensor_to_image(image)
            img.save(f"stylized_image_epoch_{i+1}.png")
        print(f"\tCompleted epoch {i+1}/{num_epochs}")

    print("Style transfer process complete.")
    return image

# Load VGG19 model
print("Loading VGG19 model...")
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Content layer where we will pull our feature maps
content_layers = ['block5_conv2']
# Style layer we are interested in
style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
# Used to calculate content loss
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Paths to your content and style images
content_path = 'content_image.jpg'
style_path = 'style_image.jpg'

# Number of epochs that will be run
epoch_ammount = 10

# Set the image size of the stylized images
image_size = 1024

print("Starting the style transfer process...")
stylized_image = run_style_transfer(content_path, style_path, max_dim=image_size, num_epochs=epoch_ammount)
print("Style transfer complete. Check the current directory for the output images.")
