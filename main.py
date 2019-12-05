import os
import platform
import time
import traceback
from collections import OrderedDict

from gdrive import upload_exp_results_to_gdrive
from utils.utils import generate_results_csv
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import sys
import sklearn
os.environ["CUDA_VISIBLE_DEVICES"] = ''
NORMALIZE = False


def add_exp(architecture, dataset, iteration, total_time, result):
    res_dict = OrderedDict()
    res_dict['algorithm'] = architecture
    res_dict['architecture'] = 'best'
    res_dict['measure'] = 'accuracy'
    res_dict['dataset'] = dataset
    res_dict['iteration'] = iteration
    res_dict['result'] = result
    res_dict['runtime'] = total_time
    res_dict['omniboard_id'] = ''
    res_dict['machine'] = platform.node()
    res_dict['local_exp_name'] = ''
    return list(res_dict.values())


def fit_classifier(): 
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    if NORMALIZE:
        for chan in range(x_train.shape[2]):
            scaler = MinMaxScaler()
            scaler.fit(x_train[:, :, chan])
            x_train[:, :, chan] = scaler.transform(x_train[:, :, chan])
            x_test[:, :, chan] = scaler.transform(x_test[:, :, chan])

    nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))

    # make the min to zero of labels
    y_train,y_test = transform_labels(y_train,y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64) 
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,y_test),axis =0).reshape(-1,1))
    y_train = enc.transform(y_train.reshape(-1,1)).toarray()
    y_test = enc.transform(y_test.reshape(-1,1)).toarray()

    if len(x_train.shape) == 2: # if univariate 
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
        x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))

    input_shape = x_train.shape[1:]
    classifier = create_classifier(classifier_name,input_shape, nb_classes, output_directory_name)

    start_time = time.time()
    res = classifier.fit(x_train,y_train,x_test,y_test, y_true)
    total_time = time.time() - start_time
    return res, total_time

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose = True):
    if classifier_name=='fcn': 
        from classifiers import fcn        
        return fcn.Classifier_FCN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mlp':
        from  classifiers import  mlp 
        return mlp.Classifier_MLP(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='resnet':
        from  classifiers import resnet 
        return resnet.Classifier_RESNET(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcnn':
        from  classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory,verbose)
    if classifier_name=='tlenet':
        from  classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory,verbose)
    if classifier_name=='twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory,verbose)
    if classifier_name=='encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory,input_shape, nb_classes, verbose)
    if classifier_name=='cnn': # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory,input_shape, nb_classes, verbose)

############################################### main 

# change this directory for your machine
# it should contain the archive folder containing both univariate and multivariate archives
root_dir = os.path.dirname(os.path.abspath(__file__))

if sys.argv[1]=='transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1]=='visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1]=='viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1]=='viz_cam':
    viz_cam(root_dir)
elif sys.argv[1]=='generate_results_csv':
    res = generate_results_csv('results.csv',root_dir)
    print(res)
else:
    if len(sys.argv) > 4:
        if sys.argv[5] == 'normalize':
            NORMALIZE = True
    # this is the code used to launch an experiment on a dataset
    for archive_name in sys.argv[1].split(','):
        for dataset_name in sys.argv[2].split(','):
            for classifier_name in sys.argv[3].split(','):
                for itr in sys.argv[4].split(','):
                    try:
                        if itr == '_itr_0':
                            itr = ''

                        output_directory_name = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + \
                                                dataset_name + '/'
                        create_directory(output_directory_name)
                        print('Method: ', archive_name, dataset_name, classifier_name, itr)
                        if os.path.exists(f'{output_directory_name}/DONE'):
                            print('Already done')

                        else:
                            datasets_dict = read_dataset(root_dir, archive_name, dataset_name)
                            res, total_time = fit_classifier()
                            print('DONE')
                            # the creation of this directory means
                            create_directory(output_directory_name + '/DONE')
                            exp_line = add_exp(classifier_name, dataset_name, itr, total_time, res[0])
                            upload_exp_results_to_gdrive(exp_line, 'University/Masters/Experiment Results/EEGNAS_results.xlsx')
                    except Exception as e:
                        from datetime import datetime

                        now = datetime.now()
                        with open(
                                f"{root_dir}/error_logs/{now.strftime('%m.%d.%Y_%H:%M:%S')}_{archive_name}_{classifier_name}_{itr}.txt",
                                "w") as err_file:
                            print('experiment failed. Exception message: %s' % (str(e)), file=err_file)
                            print(traceback.format_exc(), file=err_file)
