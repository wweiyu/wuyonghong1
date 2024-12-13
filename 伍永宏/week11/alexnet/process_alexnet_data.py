import os

def process_alexnet_data():
    with open('./alexnet_train_data/dataset.txt','w') as ff:
        path = './alexnet_train_data/train/'
        contents = ''
        for n in os.listdir(path):
            if n.endswith('.jpg') or n.endswith('.png') :
                label = 0
                name = n.split('.')[0]
                if name == 'cat':
                    label = 0
                elif name == 'dog':
                    label = 1
                contents = f'{contents}#{n},{label}'
        contents = contents[1:]
        ff.writelines(contents)

# process_alexnet_data()
