from PIL import Image, ImageDraw

class pressure_sensors_metadata():
    coords_right_normalized = [[0.2562, 0.8713],
                               [0.2021, 0.7165],
                               [0.3736, 0.5308],
                               [0.5161, 0.7074],
                               [0.3559, 0.7103],
                               [0.6707, 0.6891],
                               [0.7080, 0.4243],
                               [0.6881, 0.1321],
                               [0.8156, 0.6659],
                               [0.4396, 0.2574],
                               [0.4604, 0.1170],
                               [0.5774, 0.8585]]

    coords_left_normalized = [[0.7438, 0.8713],
                              [0.7979, 0.7165],
                              [0.6264, 0.5308],
                              [0.4839, 0.7074],
                              [0.6441, 0.7103],
                              [0.3293, 0.6891],
                              [0.2920, 0.4243],
                              [0.3119, 0.1321],
                              [0.1844, 0.6659],
                              [0.5604, 0.2574],
                              [0.5396, 0.1170],
                              [0.4226, 0.8585]]

    areas_normalized = [0.0804,
                        0.0637,
                        0.0931,
                        0.0750,
                        0.0762,
                        0.0724,
                        0.1490,
                        0.0574,
                        0.0526,
                        0.0897,
                        0.0657,
                        0.1249]

# --- Overlay circles on image and display ---
image_path = 'insole_42_complete_380p.png'
img = Image.open(image_path)
draw = ImageDraw.Draw(img)

# Choose which foot to use
coords = pressure_sensors_metadata.coords_left_normalized  # or coords_left_normalized

w, h = img.size
radius = 5  # pixels
coords.append([0, 0])
for x_norm, y_norm in coords:
    y_norm = 1 - y_norm
    x = int(x_norm * w)
    y = int(y_norm * h)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline="red", width=1, fill="red")

img.show()
img.save("insole_42_complete_380p_with_circles.png", format="png")
