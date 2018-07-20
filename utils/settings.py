import socket

if socket.gethostname() == 'purang23':
    project_folder = "e://tscRF_LSTM//Python//TeUS_RNN//TeUS_RNN//"
    intermediate_folder = project_folder + "Datasets"

if socket.gethostname() == 'minerva-VirtualBox':
    project_folder = "/media/sf_Host_Share/tscRF_LSTM/Python/TeUS_RNN/TeUS_RNN/"
    intermediate_folder = project_folder + "Datasets"

if socket.gethostname() == 'purang26':
    project_folder = "/home/shekoofeh/Project/TeUS_RNN/TeUS_RNN/"
    intermediate_folder = project_folder + "Datasets"

if socket.gethostname() == 'purang29':
    project_folder = "/data/home/shekoofeh/TeUS_RNN/TeUS_RNN/"
    intermediate_folder = project_folder + "Datasets"
