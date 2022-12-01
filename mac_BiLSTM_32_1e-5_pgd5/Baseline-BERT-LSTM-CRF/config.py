import os
import torch

# data_dir = os.getcwd() + '/data/clue/'
data_dir = os.path.abspath(os.path.dirname(__file__)) + '/data/clue/'
train_dir = data_dir + 'train.npz'
test_dir = data_dir + 'test.npz'
files = ['train_input', 'test_input']
# 转换成tokenizer的model, 'bert-base-chinese'
bert_model = 'hfl/chinese-macbert-large'#'hfl/rbt6'
# train的model
roberta_model = 'hfl/chinese-macbert-large' #'hfl/rbt6'
# model_dir = os.path.abspath(os.path.dirname(__file__)) + '/experiments/clue/'
model_dir = os.path.abspath(os.path.dirname(__file__)) + '/experiments/clue/train/'
# log_dir = model_dir + 'train.log'
log_dir = model_dir + 'train.log'
# case_dir = os.getcwd() + '/case/bad_case.txt'
case_dir = os.path.abspath(os.path.dirname(__file__)) + '/case/train/bad_case.txt'
# case_dir = os.path.abspath(os.path.dirname(__file__)) + '/case/train/bad_case.txt'
output_dir = os.path.abspath(os.path.dirname(__file__)) + '/case/train/output.json'
# output_dir = os.path.abspath(os.path.dirname(__file__)) + '/case/output.json'
exp_dir = os.path.abspath(os.path.dirname(__file__)) + '/experiments/clue/train/'
# exp_dir = os.path.abspath(os.path.dirname(__file__)) + '/experiments/clue/'
output_txt_dir = os.path.abspath(os.path.dirname(__file__)) + '/case/train/output.txt'

# 训练集、验证集划分比例
dev_split_size = 0.1

# 是否加载训练好的NER模型
load_before = False

# 是否对整个BERT进行fine tuning
full_fine_tuning = True

# hyper-parameter
learning_rate = 3e-5 # 3e-5
weight_decay = 0.01
clip_grad = 5

batch_size = 16 #32
epoch_num = 50
min_epoch_num = 5
patience = 0.0002
patience_num = 10

gpu = '0'

if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
labels = ['O','I','S']

label2id = {
    'O':0,
    'I':0,
    'S':0
}

id2label = {_id: _label for _label, _id in list(label2id.items())}
