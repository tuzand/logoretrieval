import os
import urllib


OUPUTPATH = "/home/andras/data/datasets/logopull"

def siteExists(url):
    r = urllib.urlopen(url)
    if r.getcode() == 404:
        return False
    return True


with open("companies.txt", "r") as fileList:
    if not os.path.exists(os.path.join(OUPUTPATH, "Images")):
        os.makedirs(os.path.join(OUPUTPATH, "Images"))
    if not os.path.exists(os.path.join(OUPUTPATH, "Annotations")):
        os.makedirs(os.path.join(OUPUTPATH, "Annotations"))
    lines = fileList.read().splitlines()
    i = 0
    logos = 0
    for f in lines:
        splittedPath = f.split('\t')
        site = splittedPath[-1]
        logosite = "https://logo.clearbit.com/" + site
        if siteExists(logosite):
            logos += 1
            os.system("wget " + logosite + " -O " + os.path.join(OUPUTPATH, "Images", site + ".jpg") + " >> /dev/null 2>&1")
            with open(os.path.join(OUPUTPATH, "Annotations", site + ".jpg.bboxes.txt"), "w") as ann:
                ann.write(str(0) + " " + str(0) + " " + str(128) + " " + str(128) + "\n")
        if i % 100 == 0:
            print "processed: " + str(i) + ", logos: " + str(logos)
        i += 1

