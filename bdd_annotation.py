import xml.etree.ElementTree as ET
from os import getcwd

sets=[('train')]

classes = ['car', 'bus', 'person', 'bike', 'truck', 'motor', 'train', 'rider', 'traffic sign', 'traffic light']

def convert_annotation(image_id, list_file):
    in_file = open('/home/bhap/Documents/datasets/BDD100K/Annotations/train/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    if root.find('object')==None:
        return
    list_file.write('/home/bhap/Documents/datasets/BDD100K/images/100k/train/%s.jpg'%(image_id))
    for obj in root.iter('object'):
        difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for image_set in sets:
    image_ids = open('/home/bhap/Documents/datasets/BDD100K/%s.txt'%(image_set)).read().strip().split()
    list_file = open('BDD100K_%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        convert_annotation(image_id, list_file)
    list_file.close()
