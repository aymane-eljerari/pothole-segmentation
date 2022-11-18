import matplotlib.pyplot as plt
from torchvision import transforms


def display_image(tensor):
    img = transforms.ToPILImage()(tensor/255)

    plt.imshow(img) 
    plt.show()
