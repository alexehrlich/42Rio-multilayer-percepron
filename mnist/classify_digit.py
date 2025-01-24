from neural_network.model import Model
import pygame
import numpy as np
from PIL import Image

def main():
    path = './mnist/'

    model = Model.load_model(path + 'mnist.pkl')
    
    # Initialize Pygame
    pygame.init()

    # Set up the drawing window
    window_size = 300  # The size of the drawing window
    screen = pygame.display.set_mode((window_size, window_size + 50))  # Add space for the text
    pygame.display.set_caption("Draw a Number (Press S to Process)")

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)

    # Font setup
    font = pygame.font.Font(None, 36)  # None uses the default font; 36 is the font size

    # Fill the background with black
    screen.fill(BLACK)

    # Main loop variables
    running = True
    drawing = False  # Flag to check if the mouse is pressed
    prediction_text = "Prediction: None"  # Placeholder for prediction text


    # Function to display text
    def display_text(text, position, color=WHITE):
        text_surface = font.render(text, True, color)
        screen.blit(text_surface, position)

    # Main loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True  # Start drawing
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False  # Stop drawing
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    screen.fill(BLACK)
                    prediction_text = "Prediction: None"  # Clear prediction
                if event.key == pygame.K_s:  # Process the drawing when 'S' is pressed
                    # Get the pixel data from the screen
                    raw_data = pygame.image.tostring(screen, "RGB")  # Raw RGB data
                    img = Image.frombytes("RGB", (window_size, window_size), raw_data)
                    img = img.convert("L")  # Convert to grayscale
                    img = img.resize((28, 28))  # Resize to 28x28
                    
                    # Convert to NumPy array for your model
                    arr = np.array(img)
                    arr = arr.flatten().reshape((784, 1))

                    for i, byte in zip(range(1, 785, 1), arr):
                        if byte < 50.0 / 255.0:
                            print(".", end='')
                        elif byte < 100.0 / 255.0:
                            print("o", end='')
                        else:
                            print("@", end='')
                        if i % 28 == 0:
                            print("\n", end='')
                    print("Processed image shape:", arr.shape)

                    # Predict using your model
                    prediction = np.argmax(model.predict(arr / 255.0))
                    prediction_text = f"Prediction: {prediction}"  # Update prediction text
                    print(prediction_text)

        # Draw on the screen
        if drawing:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, WHITE, (mouse_x, mouse_y), 6)  # Adjust brush size as needed

        # Update the display
        screen.fill(BLACK, (0, window_size, window_size, 50))  # Clear the text area
        display_text(prediction_text, (10, window_size + 10))  # Display the prediction text
        pygame.display.flip()

    # Quit Pygame
    pygame.quit()

if __name__ == '__main__':
    main()