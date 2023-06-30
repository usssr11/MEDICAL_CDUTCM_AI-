import argparse
from model import get_resnet50
from classfi_dataset import create_dataset_zhongyao,get_labelmap
import numpy as np
import mindspore as ms


def predict_one(input_img,image_size,workers,model,index_label_dict):
    # 加载验证集的数据进行验证
    dataset_one = create_dataset_zhongyao(dataset_dir=input_img,
                                       usage="test",
                                       resize=image_size,
                                       batch_size=1,
                                       workers=workers)
    data = next(dataset_one.create_tuple_iterator())
    # 预测图像类别
    output = model.predict(ms.Tensor(data[0]))
    pred = np.argmax(output.asnumpy(), axis=1)
    # print(f'预测结果：{index_label_dict[pred[0]]}')
    return index_label_dict[pred[0]]
def main(args):
    num_classes = args.num_classes
    image_size = args.image_size
    workers = args.workers
    input_img = args.input_image
    network = get_resnet50(num_classes,args.pretrained_path)
    model = ms.Model(network)
    print(predict_one(input_img,image_size,workers,model,get_labelmap('./zhongyiyao/train')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input_image', default='./zhongyiyao/valid/sz_bj/IMG_4078.JPG', type=str)
    parser.add_argument('--pretrained_path', default='./BestCheckpoint/resnet50-best.ckpt', type=str)
    parser.add_argument('--num_classes', default=12, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--workers', default=1, type=int)
    args = parser.parse_args()
    main(args)
