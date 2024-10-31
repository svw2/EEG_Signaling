import random
import matplotlib.pyplot as plt
import cv2
image_path = 'test.png' 
image_path2 = 'cap.png'
image = cv2.imread(image_path)
cap = cv2.imread(image_path2)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cap_coordinates = {
#     "C8": (235,196),
#     "FT6": (315,195),
#     "CP6": (159,206),
#     "C4": (229,112),
#     "TP8": (182,282),
#     "T8": (250,271),
#     "F8": (380,256.7),
#     "FC4": (306,114),
#     "F4": (382,125),
#     "CP4": (140,124),
#     "P6": (94,224),
#     "AF8": (436,241),
#     "P5": (121,285)


# }
electrode_coordinates = {
    "C1": (278 ,235),
    "C2": (368, 234),
    "CZ": (323, 234.5),
    "FC1": (280, 191),
    "FCZ": (323,192),
    "FC2": (366.5,190),
    "CP1": (280,278),
    "CPZ":(323,277),
    "CP2": (366.5,278),
    "P1": (285,321.5),
    "PZ": (323, 320.5),
    "P2": (360.5,321),
    "F1": (285,148),
    "FZ": (323,149),
    "F2": (360,148),
    "F3": (248,145),
    "F4": (398, 145),
    "F5": (212,139.5),
    "F7" : (178.5,129.5),
    "F6" : (434.5,139.5),
    #"F8" : (467.5,128.5),
    "FT7" : (151.5,178.5),
    "FC5" : (193.5, 185.5),
    "FC3" : (236.5, 189.5),
    "FC4" : (410,189.3),
    #"FC6" : (452.9, 185.7),
    "FT6" : (495, 178.5),
    #"Fp1" : (269,68),
    "Fpz" : (323,63),
    #"Fp2" : (377,68),
    "AF7" : (219,91),
    "AF3" : (268.5,103),
    #"AFZ" : (323, 105.8),
    "AF4" : (379,102),
    #"AF8" : (427,91),
    #"T7" : (142,234.4),
    "C5" : (187.5,234.4),
    #"C3" : (233,234),
    "C4" : (413.75, 234.5),
    "C6" : (459, 234.25),
    "T8" : (504, 234.5),
    #"TP7" : (151.5,290.4),
    "CP5" : (193.7,282.5),
    "CP3" : (236,279.5),
    "CP4" : (410,279.7),
    "CP6" : (452.9,283.4),
    "TP8" : (494.9,290.4),
    "P7" : (178.4,339.5),
    "P5" : (211.8,330),
    "P3" : (249.4,324),
    "P4" : (398,324),
    "P6" : (434,330),
    "P8" : (467.5,341),
    "PO7" : (219,378),
    "PO5" : (243,371),
    "PO3" : (268.5,366.5),
    "POZ" : (323,363),
    "PO4" : (378,366.5),
    "PO6" : (403.8,370.9),
    "PO8" : (427,378),
    "O1" : (269,400.5),
    "OZ" : (323,406.5),
    "O2" : (377, 400)
}

# Function to plot electrode coordinates
def plot(image_rgb, black_coordinates=None):
    x = []
    y = []
    
    for item in electrode_coordinates.values():
        x.append(item[0])
        y.append(item[1])
    
    plt.imshow(image_rgb)
    
    # Plot all electrode points in red initially
    plt.scatter(x, y, c='red', s=50)  # 's' controls dot size

    # If there are black coordinates, plot those in black
    if black_coordinates:
        black_x = [coord[0] for coord in black_coordinates]
        black_y = [coord[1] for coord in black_coordinates]
        plt.scatter(black_x, black_y, c='black', s=50)  # No outline

    plt.axis('off')  # Hide axes
    plt.show()

# Function to randomly select some coordinates and make them black
def random_black_coordinates(n=10):
    # Get a list of all coordinates
    all_coordinates = list(electrode_coordinates.values())
    
    # Randomly sample 'n' coordinates from the list
    black_coords = random.sample(all_coordinates, n)
    
    return black_coords
# Example of using the random_black_coordinates function
black_coords = random_black_coordinates(n=5)  # Change 'n' to the number of black points you want

# Plot the image with black coordinates
plot(image_rgb, black_coords)
