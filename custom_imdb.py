import os

def get_custom_imdb(imagepath):
    return customimdb(imagepath)

class customimdb(object):

    def __init__(self, imagepath):
        self.image_index = list()
        for file in os.listdir(imagepath):
            if file.endswith('.jpg') or file.endswith('.png'):
                self.image_index.append(os.path.join(imagepath, file))

    def image_path_at(self, i):
        impath = self.image_index[i]
        print impath
        return impath

