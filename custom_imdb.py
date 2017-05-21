import os

def get_custom_imdb(imagepath):
    return customimdb(imagepath)

class customimdb(object):
    num_classes = 2
    name = ""

    def __init__(self, imagepath):
        self.image_index = list()
        name = imagepath.split('/')[-1]
        for file in os.listdir(imagepath):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.image_index.append(os.path.join(imagepath, file))

    def image_path_at(self, i):
        impath = self.image_index[i]
        print impath
        return impath

