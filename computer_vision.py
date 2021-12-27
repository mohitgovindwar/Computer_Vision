# Digitize your notes with Azure Computer Vision OCR

## Import libraries
"""

pip install azure-cognitiveservices-vision-computervision

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numpy as np

"""## Create variables for your key and endpoint"""

key = 'Your key'
endpoint = 'Your endpoint'

"""## Authenticate the client"""

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

"""## Extract handwritten text"""

def read_handwritten_text(image_name):
    # Open local image file
    image_path = "images/" + image_name
    image = open(image_path, "rb")

    img = Image.open(image_path)

    # Call the API
    read_response = computervision_client.read_in_stream(image, raw=True)

    # Get the operation location (URL with an ID at the end)
    read_operation_location = read_response.headers["Operation-Location"]

    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img)

    # Print the detected text and bounding boxes
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                # Print line
                print(line.text)

                # line.bounding_box contains 4 pairs of (x, y) coordinates
                xy1 = [line.bounding_box[0], line.bounding_box[1]]
                xy2 = [line.bounding_box[2], line.bounding_box[3]]
                xy3 = [line.bounding_box[4], line.bounding_box[5]]
                xy4 = [line.bounding_box[6], line.bounding_box[7]]
                box_coordinates = np.array([xy1, xy2, xy3, xy4])
                
                # Create a Rectangle patch
                box = patches.Polygon(box_coordinates, closed=True, linewidth=2, edgecolor='r', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(box)

                # Print words in line with confidence score
                for word in line.words:
                    print(f"   * {word.text}: {word.confidence * 100:.2f}")
                    
                    # Uncomment the following lines to display the bounding box of each word
                    '''
                    xy1 = [word.bounding_box[0], word.bounding_box[1]]
                    xy2 = [word.bounding_box[2], word.bounding_box[3]]
                    xy3 = [word.bounding_box[4], word.bounding_box[5]]
                    xy4 = [word.bounding_box[6], word.bounding_box[7]]
                    box_coordinates = np.array([xy1, xy2, xy3, xy4])
                
                    # Create a Rectangle patch
                    box = patches.Polygon(box_coordinates, closed=True, linewidth=1, edgecolor='c', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(box)
                    '''
    plt.show()

read_handwritten_text("notes1.jpg")

read_handwritten_text("notes2.jpg")

read_handwritten_text("notes3.jpg")

read_handwritten_text("notes4.jpg")
