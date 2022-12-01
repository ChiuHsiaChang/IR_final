import re
import numpy as np
import matplotlib.pyplot as plt
import logging
import config
import utils

def result_plot():
    with open(config.log_dir,'r') as f:
        content=f.read()
    input_str = content
    #train loss
    pattern_train_loss = "(train loss: )(\d+\.?\d*)"
    match_train_loss = re.findall(pattern_train_loss, input_str)
    results_train_loss = []
    for i in range(len(match_train_loss)):
        result_train_loss = float(match_train_loss[i][1])
        results_train_loss.append(result_train_loss)    
    # print(results_train_loss)

    #dev_loss
    pattern_dev_loss = "(dev loss: )(\d+\.?\d*)"
    match_dev_loss = re.findall(pattern_dev_loss, input_str)
    results_dev_loss = []
    for i in range(len(match_dev_loss)):
        result_dev_loss = float(match_dev_loss[i][1])
        results_dev_loss.append(result_dev_loss)    
    # print(results_dev_loss)

    #train_f1
    pattern_train_f1 = "(train loss: )(\d+\.?\d*\,\s)(f1 score: )(\d+\.?\d*)"
    match_train_f1 = re.findall(pattern_train_f1, input_str)
    results_train_f1 = []
    print(match_train_f1)
    for i in range(len(match_train_f1)):
        result_train_f1 = float(match_train_f1[i][3])
        results_train_f1.append(result_train_f1)    
    print(results_train_f1)

    #dev_f1
    pattern_dev_f1 = "(dev loss: )(\d+\.?\d*\,\s)(f1 score: )(\d+\.?\d*)"
    match_dev_f1 = re.findall(pattern_dev_f1, input_str)
    results_dev_f1 = []
    print(match_dev_f1)
    for i in range(len(match_dev_f1)):
        result_dev_f1 = float(match_dev_f1[i][3])
        results_dev_f1.append(result_dev_f1)    
    # print(results_dev_f1)

    x1 = range(0,len(results_train_loss))
    y1 = results_train_loss
    x2 = range(0,len(results_dev_loss))
    y2 = results_dev_loss
    # print(x1,y1,x2,y2)
    plt.plot(x1, y1,color='green', label='train_loss')
    plt.plot(x2, y2,color='red', label='dev_loss')
    plt.legend()
    plt.title('Loss')
    plt.xlabel('steps')
    plt.ylabel('loss')
    # plt.show()
    img_path_loss = config.exp_dir + 'train_loss.jpg'
    plt.savefig(img_path_loss)
    plt.close()

    x1 = range(0,len(results_train_f1))
    y1 = results_train_f1
    x2 = range(0,len(results_dev_f1))
    y2 = results_dev_f1
    # print(x1,y1,x2,y2)
    plt.plot(x1, y1,color='green', label='train_f1')
    plt.plot(x2, y2,color='red', label='dev_f1')
    plt.legend()
    plt.title('F1 Score')
    plt.xlabel('steps')
    plt.ylabel('f1_score')
    # plt.show()
    img_path_loss = config.exp_dir + 'train_f1.jpg'
    plt.savefig(img_path_loss)
    plt.close()
    # x2 = range(0,len(results_dev_f1))
    # y2 = results_dev_f1
    # plt.plot(x2, y2,color='red', label='dev_f1_score')
    # plt.legend()
    # plt.title('F1 Score')
    # plt.xlabel('steps')
    # plt.ylabel('f1_score')
    # # plt.show()
    # img_path_loss = config.exp_dir + 'dev_f1.jpg'
    # plt.savefig(img_path_loss)
    # plt.close()
    f.close()
    utils.set_logger(config.log_dir)
    logging.info("Finish plot!")