import tensorflow as tf
import numpy as np
from PIL import Image
import os
import argparse

def parse_arguments():
    """
    Parse command line arguments for the style transfer script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Neural Style Transfer Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    %(prog)s --content path/to/content.jpg --style path/to/style.jpg

  High quality output:
    %(prog)s --content photo.jpg --style art.jpg --epochs 20 --image-size 1024
        """
    )
    
    # Required arguments
    parser.add_argument('--content', required=True,
                        help='Path to the content image')
    parser.add_argument('--style', required=True,
                        help='Path to the style image')
    
    # Optional arguments with defaults
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run (default: 10)')
    parser.add_argument('--content-weight', type=float, default=1e4,
                        help='Weight for content loss (default: 1e4)')
    parser.add_argument('--style-weight', type=float, default=1e-2,
                        help='Weight for style loss (default: 1e-2)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Maximum dimension of the output image (default: 512)')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save output images (default: output)')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save frequency in epochs (default: 1)')
    parser.add_argument('--final-name', default='final_stylized.png',
                        help='Name of the final output image (default: final_stylized.png)')
    
    return parser.parse_args()

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
    if not os.path.exists(path_to_img):
        raise FileNotFoundError(f"Image not found: {path_to_img}")
        
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

def vgg_layers(layer_names):
    """
    Creates a vgg model that returns a list of intermediate output values.
    
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
        """Process the input through the VGG19 model and extract features."""
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
    """Clip the pixel values of an image to be between 0 and 1."""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    """
    Calculate the total loss for style transfer.
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
    """Perform a single training step for style transfer."""
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))

def tensor_to_image(tensor):
    """Convert a tensor to a PIL Image."""
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        tensor = tensor[0]
    return Image.fromarray(tensor)

def run_style_transfer(content_path, style_path, output_dir, save_freq=1, final_name='final_stylized.png',num_epochs=10, content_weight=1e4, style_weight=1e-2, max_dim=512):
    """
    Run the style transfer algorithm with additional output options.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        if (i + 1) % save_freq == 0:
            print(f"\tSaving intermediate result for epoch {i+1}")
            img = tensor_to_image(image)
            img.save(os.path.join(output_dir, f"epoch_{i+1}.png"))
        
        print(f"\tCompleted epoch {i+1}/{num_epochs}")

    # Save final result with custom name
    final_img = tensor_to_image(image)
    final_path = os.path.join(output_dir, final_name)
    final_img.save(final_path)
    print(f"Style transfer complete. Final image saved as: {final_path}")
    
    return image

def main():
    """Main function to run the style transfer script."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Print configuration
    print("\nStyle Transfer Configuration:")
    print(f"Content Image: {args.content}")
    print(f"Style Image: {args.style}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Image Size: {args.image_size}px")
    print(f"Epochs: {args.epochs}")
    print(f"Content Weight: {args.content_weight}")
    print(f"Style Weight: {args.style_weight}")
    print(f"Save Frequency: Every {args.save_freq} epoch(s)")
    print(f"Final Image Name: {args.final_name}\n")

    # Load VGG19 model
    print("Loading VGG19 model...")
    global vgg
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

    # Content and style layers configuration
    global content_layers, style_layers, num_content_layers, num_style_layers
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
    num_content_layers = len(content_layers)
    num_style_layers = len(style_layers)

    # Run style transfer with provided arguments
    try:
        stylized_image = run_style_transfer(
            content_path=args.content,
            style_path=args.style,
            output_dir=args.output_dir,
            save_freq=args.save_freq,
            final_name=args.final_name,
            num_epochs=args.epochs,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            max_dim=args.image_size
        )
        print("\nStyle transfer completed successfully!")
    except Exception as e:
        print(f"\nError during style transfer: {str(e)}")
        raise

if __name__ == "__main__":
    main()
