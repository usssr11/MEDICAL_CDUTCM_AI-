from model import get_resnet50,get_opt,get_lossfn
from mindspore import nn
from mindspore import ops
from sklearn.metrics import classification_report
import mindspore as ms
import os
from classfi_dataset import create_dataset_zhongyao,get_labelmap
import argparse
import numpy as np


def train_loop(model, dataset, loss_fn, optimizer,step_size_train):
    # Define forward function
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # Get gradient function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0 or batch == step_size_train - 1:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def test_loop(model, dataset, loss_fn,index_label_dict):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    y_true = []
    y_pred = []
    for data, label in dataset.create_tuple_iterator():
        y_true.extend(label.asnumpy().tolist())
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        y_pred.extend(pred.argmax(1).asnumpy().tolist())
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # print(y_true)
    # print(y_pred)
    classes = list(index_label_dict.values())

    print(classification_report(y_true,y_pred,labels = np.arange(0,len(classes)),target_names= classes,digits=3))
    return correct,test_loss

def main(args):
    # 加载dataloader
    data_dir = args.data_dir
    train_dir = data_dir+"/"+"train"
    valid_dir = data_dir+"/"+"valid"
    num_epochs = args.num_epochs
    # early stopping
    patience = args.patience
    batch_size = args.batch_size
    image_size = args.image_size
    workers = args.workers
    num_classes = args.num_classes

    dataset_train = create_dataset_zhongyao(dataset_dir=train_dir,
                                           usage="train",
                                           resize=image_size,
                                           batch_size=batch_size,
                                           workers=workers)
    step_size_train = dataset_train.get_dataset_size()

    dataset_val = create_dataset_zhongyao(dataset_dir=valid_dir,
                                         usage="valid",
                                         resize=image_size,
                                         batch_size=batch_size,
                                         workers=workers)

    lr = nn.cosine_decay_lr(min_lr=0.00001, max_lr=0.001, total_step=step_size_train * num_epochs,
                            step_per_epoch=step_size_train, decay_epoch=num_epochs)
    network = get_resnet50(num_classes)
    opt = get_opt(network, lr)
    loss_fn = get_lossfn()
    index_label_dict = get_labelmap(train_dir)


    # 最佳模型存储路径
    best_acc = 0
    best_ckpt_dir = args.best_ckpt_dir
    best_ckpt_path = os.path.join(best_ckpt_dir,'resnet50-best.ckpt')
    # best_ckpt_path = "./BestCheckpoint/Googlenet-best.ckpt"
    loss_list = []
    for t in range(num_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(network, dataset_train, loss_fn, opt,step_size_train)
        # acc = model.eval(dataset_val)['Accuracy']
        acc,loss = test_loop(network, dataset_val, loss_fn,index_label_dict)
        loss_list.append(loss)
        if acc > best_acc:
            best_acc = acc
            if not os.path.exists(best_ckpt_dir):
                os.mkdir(best_ckpt_dir)
            ms.save_checkpoint(network, best_ckpt_path)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count > patience:
                print('Early stopping triggered. Restoring best weights...')
                # model.load_state_dict(best_weights)
                break
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data_dir', default='./zhongyiyao', type=str)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--num_classes', default=12, type=int)
    parser.add_argument('--best_ckpt_dir', default='./BestCheckpoint', type=str)
    args = parser.parse_args()
    main(args)



