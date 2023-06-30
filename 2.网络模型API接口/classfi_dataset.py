import os
from PIL import Image
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset import GeneratorDataset
from mindspore import dtype as mstype
# 注意没有使用ImageFloder进行加载，原因是'.ipynb_checkpoints'缓存文件夹会被当作类文件夹进行识别，导致数据集加载错误
class Iterable:
    def __init__(self ,data_path):
        self._data = []
        self._label = []
        self._error_list = []
        if data_path.endswith(('JPG' ,'jpg' ,'png' ,'PNG')):
            # 用作推理，所以没有label
            image = Image.open(data_path)
            self._data.append(image)
            self._label.append(0)
        else:
            classes = os.listdir(data_path)
            # 问题：文件夹里会出现'.ipynb_checkpoints'文件夹，但是本身不显示
            if '.ipynb_checkpoints' in classes:
                classes.remove('.ipynb_checkpoints')
            print(classes)
            for (i ,class_name) in enumerate(classes):
                # new_path = os.path.join(path,class_name)
                new_path =  data_path +"/" +class_name
                for image_name in os.listdir(new_path):
                    try:
                        image = Image.open(new_path + "/" + image_name)
                        self._data.append(image)
                        self._label.append(i)
                    except:
                        pass


    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)

    def get_error_list(self ,):
        return self._error_list


# 加载dataloader
def create_dataset_zhongyao(dataset_dir,usage,resize,batch_size,workers):
    data = Iterable(dataset_dir)
    data_set = GeneratorDataset(data,column_names=['image','label'])
    trans = []
    if usage == "train":
        trans += [
            vision.RandomCrop(700, (4, 4, 4, 4)),
            # 这里随机裁剪尺度可以设置
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    trans += [
        vision.Resize((resize,resize)),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    target_trans = transforms.TypeCast(mstype.int32)
    # 数据映射操作
    data_set = data_set.map(
        operations=trans,
        input_columns='image',
        num_parallel_workers=workers)

    data_set = data_set.map(
        operations=target_trans,
        input_columns='label',
        num_parallel_workers=workers)

    # 批量操作
    data_set = data_set.batch(batch_size,drop_remainder=True)

    return data_set

def get_labelmap(train_dir):
    # index_label的映射
    index_label_dict = {}
    classes = os.listdir(train_dir)
    if '.ipynb_checkpoints' in classes:
        classes.remove('.ipynb_checkpoints')
    for i, label in enumerate(classes):
        index_label_dict[i] = label
    return index_label_dict
