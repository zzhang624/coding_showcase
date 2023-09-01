import csv
import os
import pathlib
import re
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
from os import listdir, mkdir, remove
from shutil import rmtree
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, Precision, Recall
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

def input_calculation(model_location, file, output_dir, GPU_name='cuda:0', toks_per_batch=4096, nogpu=False, return_contacts=False, repr_layers=[-1]):
    file = pathlib.Path(file)
    output_dir = pathlib.Path(output_dir)
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
        
    if torch.cuda.is_available() and not nogpu:
        model = model.cuda(GPU_name)
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    batch_converter = alphabet.get_batch_converter()
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=batch_converter, batch_sampler=batches)
    
    print(f"Read {file} with {len(dataset)} proteins")

    output_dir.mkdir(references=True, exist_ok=True)
    
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if batch_idx % 100 == 0:
                print(
                    f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} proteins)"
                )
            if torch.cuda.is_available() and not nogpu:
                toks = toks.to(device=GPU_name, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            inputs = {
                layer: t.to(device="cpu") for layer, t in out["inputs"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                output_file.reference.mkdir(references=True, exist_ok=True)
                result = {"label": label}
                result["mean_inputs"] = {
                    layer: t[i, 1 : len(strs[i]) + 1].mean(0).clone()
                    for layer, t in inputs.items()
                }
                torch.save(
                    result,
                    output_file,
                )

def input_to_np(dir_name, output_file):
    """
    Convert mean_inputs to NumPy array format.

    Args: 
        dir_name (str): The directory containing mean inputs.
        output_file (str): The name of the output .npy file.
    """
    dir_name = pathlib.Path(dir_name)
    output_file = pathlib.Path(output_file)

    li = [torch.load(f)["mean_inputs"][33] for f in dir_name.glob("*")]
    np.save(output_file, np.array(li)) 

def generate_training_dataset_from_csv(filepath, protein_col_name, output_dir, GPU_name = 'cuda'):
    """
    This function reads a CSV file, generates a training dataset based on the specified protein column,
    and saves the resulting training datasets in the given output directory.

    Args:
        filepath (str): The file path of the input CSV file.
        protein_col_name (str): The name of the column in the CSV file containing the protein data.
        output_dir (str): The directory path where the generated training datasets will be saved.
        GPU_name (str, optional): The name of the GPU. Defaults to 'cuda'.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df = pd.read_csv(filepath)
    generate_training_data(df, protein_col_name, output_dir, GPU_name=GPU_name)
    
def generate_training_dataset_from_excel(filepath, protein_col_name, output_dir, GPU_name='cuda', sheet_index=0):
    """
    Args: 
        filepath (str): The filepath of the input Excel file.
        protein_col_name (str): The name of the column in the Excel file containing the protein data.
        output_dir (str): The directory path where the generated training datasets will be saved.
        GPU_name (str, optional): The name of the GPU. Defaults to 'cuda'.
        sheet_index (Union[int, str], optional): The sheet name or index in the Excel file to read data from.
            Strings are used for sheet names. Integers are used in zero-indexed sheet positions (chart sheets do not count as a sheet position). Defaults to 0.
    """
    if not os.path.exists(output_dir):
        mkdir(output_dir)
    df = pd.read_excel(filepath, sheet_index=sheet_index)
    generate_training_data(df, protein_col_name, output_dir, GPU_name=GPU_name)

def generate_training_data(df, protein_col_name, output_dir, GPU_name):
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(references=True, exist_ok=True)

    df = df[df[protein_col_name].apply(lambda x: re.search('[^ARNDCEQGHILKMFPSTWYV]', x) is None)]
    df.to_csv(f"{output_dir}/Refined.csv")
    
    # Write fasta file
    i = 0
    fas_path = output_dir / "refined.fas"
    with fas_path.open('w') as f:
        for index, item in df[protein_col_name].items():
            if i != 0:
                f.write("\n")
            f.write(f">{i}\n{item}")
            i += 1
    


""""............................MoleculeCls........................................"""

class MoleculeCls:
    counts = {}

    def __init__(self, dataset_dir, D_col, B_col, mutation_count_column=None, bin_num=2, using_chain_as_feature=False, name=None):
        """
        Load an molecule dataset generated by the generate_training_dataset function into one object.

        Args: 
            dataset_dir (str): The name of the directory generated by the generate_training_dataset function.
            bin_num (int, optional): The number of gates in the experiments. Defaults to 2.
            using_chain_as_feature (bool, optional): Defaults to False.
            name (str, optional): Used only when using_chain_as_feature is True. Defaults to None.
        """
        self.input = np.load(dataset_dir+"/input.npy")
        self.n_proteins = self.input.shape[0]
        self.df = pd.read_csv(dataset_dir+"/refined.csv")
        self.bin_num = bin_num
        self.name = name
        self.D_col = D_col 
        self.B_col = B_col
        self.using_chain_as_feature = using_chain_as_feature

        if using_chain_as_feature:
            self.chain = 1 if 'XX' in name else 0
        
        self.reference_index = self.get_reference_index(mutation_count_column)
        
    def get_reference_index(self, mutation_count_column):
        if mutation_count_column:
            return self.df[self.df[mutation_count_column] == 0].index[0]
        else:
            return 0
        
    def generate_training_array(self):
        self.calculate_complexes()
        self.calculate_ln_weight()
        self.generate_relative_data()
        self.calculate_training_array_relative()
   
    def calculate_complexes(self):
        if self.bin_num == 4:
            self.df['complexes'] = self.calculate_complexes_4_bins()
        elif self.bin_num == 2:
            self.df['complexes'] = self.calculate_complexes_2_bins()

    def calculate_complexes_4_bins(self):
        df = self.df[['Bin1', 'Bin2', 'Bin3', 'Bin4']]
        df = df / MoleculeCls.counts[self.name]['reads_bins'] * MoleculeCls.counts[self.name]['cells_bins']
        return df.apply(lambda x: MoleculeCls.func2(x['Bin1'], x['Bin2'], x['Bin3'], x['Bin4']), axis=1)

    def calculate_complexes_2_bins(self):
        sum_complex = self.df[self.B_col].sum()
        sum_surface = self.df[self.D_col].sum()
        pop_frac_complex = self.df[self.B_col] / sum_complex
        pop_frac_surface = self.df[self.D_col] / sum_surface
        return pop_frac_complex / pop_frac_surface
            
    @staticmethod
    def func1(a,b,c,d):
        return (a * 1 + b * 2 + c * 3 + d * 4) / (a + b + c + d)
    
    @staticmethod
    def func2(a,b,c,d):
        return (a + b) / (c + d) if (c + d) != 0 else float('inf')
    
    def calculate_ln_weight(self):
        if self.bin_num == 4:
            self.df['sample_weight'] = np.log(self.df['Bin1'] + self.df['Bin2'] + self.df['Bin3'] + self.df['Bin4'])
        elif self.bin_num == 2:
            self.df['sample_weight'] = np.log(self.df[self.B_col] + self.df[self.D_col])
            
    def generate_relative_data(self):
        reference_series = self.df.loc[self.reference_index]
        self.df_relative = self.df.drop(index=self.reference_index)
        self.up_20 = reference_series['complexes'] * 1.2
        self.reference_complex = reference_series['complexes']
        self.down_20 = reference_series['complexes'] * 0.8
        self.df_relative["class"] = self.df_relative['complexes'].apply(self.map_2)
        count = np.bincount(self.df_relative["class"].to_numpy())
        self.classes_ratio = count[0] / count[1]

        reference_repres = self.input[self.reference_index]
        self.input_relative = np.delete(self.input, self.reference_index, axis=0)
        self.input_relative = self.input_relative - reference_repres
        
    @staticmethod
    def map_1(x):
        if x < 2:
            return 2
        elif x < 3:
            return 1
        else:
            return 0
             
    def map_2(self,x):
        if x < self.down_20:
            return 0
        else:
            return 1    
        
    def calculate_training_array_relative(self):
        df_array = self.df_relative[['sample_weight', 'class']].to_numpy()

        if self.using_chain_as_feature:
            chain_feature = np.array([[self.chain]] * (self.n_proteins - 1))
            self.arr_relative = np.concatenate((chain_feature, self.input_relative, df_array), axis=1)
        else:
            self.arr_relative = np.concatenate((self.input_relative, df_array), axis=1)

        del self.input
        del self.input_relative

""""............................Random Forest classifier........................................"""

def calculate_class_weights(array, ratio=1):
    """Calculate class weights"""
    if ratio is None:
        return None
    r = [1, ratio]
    weights = np.bincount(array[:, -1].astype(int), weights=array[:, -2])
    weights = weights / weights.max()
    weights = 1 / weights
    weights = weights * r
    return dict(enumerate(weights))

def print_metrics(y_true, y_pred, sample_weight):
    """Print F1 score for each class"""
    print("F1_score for each class:", f1_score(y_true, y_pred, sample_weight=sample_weight, average=None))

def train_rf(molecule_list=[], verbose=0, n_jobs=25, max_depth=20, tune_hyper=False, parameters={}, fbeta=1):
    """
    Train and test a RandomForestClassifier using the scikit-learn library.

    Args:
        molecule_list (List[molecule]): A list of molecule objects. Defaults to an empty list.
        verbose (int, optional): Controls the verbosity when fitting. Set to 2 for more detailed output. Defaults to 0.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to 25.
        max_depth (int, optional): The maximum depth of the trees in the random forest. Defaults to 20.
        tune_hyper (bool, optional): Whether to perform hyperparameter tuning using GridSearchCV. Defaults to False.
        parameters (Dict[str, List] or List[Dict[str, List]], optional): A dictionary with parameter names as keys and lists
            of parameter settings to try as values, or a list of such dictionaries. This enables searching over any protein
            of parameter settings. Defaults to an empty dictionary.
        fbeta (float, optional): Determines the weight of recall in the F-score. A value of beta less than 1 lends more
            weight to precision, while a value of beta greater than 1 favors recall. As beta approaches 0, only precision
            is considered, while as beta approaches +inf, only recall is considered. Defaults to 1.

    Returns:
        RandomForestClassifier: A trained RandomForestClassifier object.
    """
    array = np.concatenate([i.arr_relative for i in molecule_list])

    X = array[:, :-2]
    w = array[:, -2]
    y = array[:, -1]

    if tune_hyper:
        class_weight = calculate_class_weights(array)
        rfc = RFC(n_jobs=n_jobs, class_weight=class_weight)
        scorer = make_scorer(f1_score, beta=fbeta, pos_label=1)
        clf = GridSearchCV(rfc, parameters, scoring=scorer, verbose=verbose)
    else:
        class_weight = calculate_class_weights(array)
        clf = RFC(class_weight=class_weight, max_depth=max_depth, verbose=verbose, n_jobs=n_jobs)

    del(array)

    clf.fit(X, y, sample_weight=w)

    return clf

def test_rf(clf, molecule_list):
    """
    Test a trained RandomForestClassifier on a list of molecule objects.

    Args:
        clf (RandomForestClassifier): A trained RandomForestClassifier object.
        molecule_list (List[molecule]): A list of molecule objects.
    """
    array = np.concatenate([i.arr_relative for i in molecule_list])
    X = array[:, :-2]
    w = array[:, -2]
    y = array[:, -1]

    y_pred = clf.predict(X)
    print_metrics(y, y_pred, w)
         

""""............................NN classifier........................................"""
    
def train_NN(molecule_list, record_name, batch_size, lr, NN_architecture=None,
              num_epochs=1000, test_size=0.1, num_shuffle=1, early_stopping=True, rolling_average_range=70, 
              DataLoader_num_workers=0, GPU_name="cuda", reload_from_checkpoint=False):
    """
    Train a neural network using Pytorch.
    
    This function does not return a classifier. Instead, it stores the results in 'metrics_NN/', 'tensorboard_NN/', 'Checkpoints_NN/', and 'Models/' folders.

    Args:
        molecule_list (list): A list of molecule objects.
        record_name (str): A string used as the base name for storing results in the directories mentioned above.
        batch_size (int): Batch size for the training process (recommended: 3000 to 10000).
        lr (float): Learning rate (recommended: 0.0001 to 0.003).
        NN_architecture (list, optional): A list of integers representing the neural network architecture. For example, [1280, 100, 100, 2] creates a neural network with 1280 input nodes, two hidden layers with 100 nodes each, and 2 output nodes for binary classification.
        num_epochs (int, optional): Number of training epochs (default: 1000).
        test_size (float, optional): Proportion of data reserved for validation (default: 0.1).
        num_shuffle (int, optional): Number of times to shuffle test_size data and retrain the model (default: 1).
        early_stopping (bool, optional): Whether to use early stopping to prevent overfitting (default: True). If True, the function calculates the F1 score on the test_size data every epoch and then computes the rolling average F1 score over "rolling_average_range" epochs. Training stops if the current rolling average F1 score is lower than the previous one.
        rolling_average_range (int, optional): Number of epochs for calculating the rolling average F1 score (default: 70).
        DataLoader_num_workers (int, optional): Number of workers for the PyTorch DataLoader (default: 0). Setting this to a positive integer enables multi-process data loading. However, it may not work well on all systems.
        GPU_name (str, optional): GPU device name for training (default: "cuda").
        reload_from_checkpoint (bool, optional): Whether to reload the training process from a checkpoint (default: False).
    """

    def create_directories(directory_list):
        for folder_name in directory_list:
            os.makedirs(folder_name, exist_ok=True)

    def initialize_metrics(device):
        return {
            'F1': F1Score(num_classes=2, average=None).to(device),
            'Precision': Precision(num_classes=2, average=None).to(device),
            'Recall': Recall(num_classes=2, average=None).to(device)
        }

    def load_checkpoint(record_name_loop):
        loaded_checkpoint = torch.load("Checkpoints_NN/{}.pth".format(record_name_loop))
        reloaded_epoch = loaded_checkpoint['epoch']
        model.load_state_dict(loaded_checkpoint['model_state'])
        optimizer.load_state_dict(loaded_checkpoint['optim_state'])
        return reloaded_epoch

    def save_checkpoint(epoch, model, optimizer, record_name_loop):
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }
        torch.save(checkpoint, "Checkpoints_NN/{}.pth".format(record_name_loop))

    def early_stopping_check(metrics_names, current_score, last_average_score, rolling_score, rolling_average_range, epoch):
        for name in metrics_names:
            current_score[name] = current_score[name] / 5

        for name in metrics_names:
            rolling_score[name] += current_score[name]

        if (epoch + 1) % rolling_average_range == 0:
            current_average_score = {name: rolling_score[name] / rolling_average_range for name in metrics_names}
            print(f'epoch {epoch+1}, test_F1 = {current_average_score["F1"]:.4f}')

            if current_average_score["F1"] < last_average_score["F1"]:
                return True, current_average_score
            else:
                last_average_score = current_average_score
                for name in metrics_names:
                    rolling_score[name] = 0.0
        return False, last_average_score

    # Create necessary folders
    create_directories(['metrics_NN', 'tensorboard_NN', 'Checkpoints_NN', 'Models'])

    with open('metrics_NN/' + record_name, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['F1', 'Precision', 'Recall'])

        device = torch.device(GPU_name)
        metrics = initialize_metrics(device)
        metrics_names = ['F1', 'Precision', 'Recall']

        for i in range(num_shuffle):
            model = NeuralNet(NN_architecture).to(device)
            writer = SummaryWriter('tensorboard_NN/{}_r{}'.format(record_name, i + 1))
            record_name_loop = '{}_r{}'.format(record_name, i + 1)
            tensor = load_data(molecule_list)
            class_weight = cal_class_weight_tensor(tensor).to(device)
            tensor_train, tensor_test = train_test_split(tensor, test_size=test_size)
            del(tensor)
            test_sample_len = tensor_test.shape[0]

            trainingset = TrainingSet(tensor_train)
            Training_dataloader = DataLoader(dataset=trainingset, batch_size=batch_size, shuffle=True, num_workers=DataLoader_num_workers, drop_last=True)

            criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='none')
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            if reload_from_checkpoint:
                reloaded_epoch = load_checkpoint(record_name_loop)
            else:
                reloaded_epoch = 0

            n_step = len(Training_dataloader)
            running_f1_score = 0.0
            current_score = {}
            rolling_score = {}
            last_average_score = {}
            last_average_score['F1'] = 0.0
            for name in metrics_names:
                rolling_score[name] = 0.0

            for epoch in range(num_epochs):
                for i, (x, w, y) in enumerate(Training_dataloader):
                    x, w, y = x.to(device), w.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss = loss * w
                    loss = loss.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(outputs, 1)
                    running_f1_score += metrics['F1'](predicted, y)[1].item()

                current_f1 = running_f1_score / n_step
                writer.add_scalar('training F1', current_f1, epoch + reloaded_epoch)
                running_f1_score = 0.0

                if early_stopping:
                    for name in metrics_names:
                        current_score[name] = 0.0
                    for i in range(5):
                        shuffle_index = torch.randint(high=test_sample_len, size=(3000,))
                        shufflled_testset = tensor_test[shuffle_index].to(device)
                        outputs = model(shufflled_testset[:, :-2])
                        _, predicted = torch.max(outputs, 1)
                        for name in metrics_names:
                            current_score[name] += metrics[name](predicted, shufflled_testset[:, -1].long())[1].item()

                    writer.add_scalar('test F1', current_score['F1'], epoch + reloaded_epoch)
                    stop_training, last_average_score = early_stopping_check(metrics_names, current_score, last_average_score, rolling_score, rolling_average_range, epoch)

                    if stop_training:
                        save_checkpoint(epoch + reloaded_epoch, model, optimizer, record_name_loop)
                        model.load_state_dict(torch.load("Checkpoints_NN/{}.pth".format(record_name_loop))["model_state"])
                        torch.save(model, "Models/{}.pth".format(record_name_loop))
                        csv_writer.writerow([last_average_score['F1'], last_average_score['Precision'], last_average_score['Recall']])
                        del(model, optimizer)
                        break
                    else:
                        save_checkpoint(epoch + reloaded_epoch, model, optimizer, record_name_loop)


def load_data(molecule_list):
    """
    Load data from the given molecule list and convert it into a tensor.
    
    Args:
        molecule_list (list): A list of molecule objects.

    Returns:
        tensor (torch.Tensor): A tensor containing the loaded data.
    """
    array = np.concatenate([i.arr_relative for i in molecule_list])
    tensor = torch.from_numpy(array).float()
    return tensor

def cal_class_weight_tensor(tensor):
    """
    Calculate class weights for the given tensor.
    
    Args:
        tensor (torch.Tensor): A tensor containing the data.

    Returns:
        li (torch.Tensor): A tensor containing the calculated class weights.
    """
    li = torch.bincount(tensor[:,-1].long(), weights=tensor[:,-2])
    li = li/torch.max(li)
    li = 1/li
    return li

class NeuralNet(nn.Module):
    """
    Define a neural network model with the given architecture.
    
    Args: 
        NN_architecture (list): A list of integers representing the size of each layer, for example: [input_size, hidden_layer1_size, ..., output_size]
    """
    def __init__(self, NN_architecture):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = len(NN_architecture) - 1
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(NN_architecture[i], NN_architecture[i+1]))
        self.relu = nn.ReLU()
        
    def forward(self, x):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if n != self.num_layers - 1:
                x = self.relu(x)
        return x

class TrainingSet(Dataset):
    """
    A custom PyTorch Dataset class for the molecule training set.
    
    Args:
        tensor (torch.Tensor): A tensor containing the training data.
    """
    def __init__(self, tensor):
        self.x = tensor[:,:-2]
        self.w = tensor[:,-2]
        self.y = tensor[:,-1].long()
        self.n_samples = tensor.shape[0]
        
    def __getitem__(self, index):
        return self.x[index], self.w[index], self.y[index]
    
    def __len__(self):
        return self.n_samples
    
def test_NN(model_name, molecule_list, map_location='cpu'):
    """
    Test a trained neural network (NN) on a given molecule list.
    
    Args:
        model_name (str): The name of a trained NN model stored in the 'Models/' directory.
        molecule_list (list): A list of molecule objects.
        map_location (str): The device to map the model's tensors to ('cpu' or 'cuda').
    """
    metrics = {
        'F1': F1Score(num_classes=2, average=None),
        'Precision': Precision(num_classes=2, average=None),
        'Recall': Recall(num_classes=2, average=None)
    }
    metrics_names = ['F1', 'Precision', 'Recall']
    
    model = torch.load('Models/{}'.format(model_name), map_location=map_location)
    model.eval()
    tensor = load_data(molecule_list)
    outputs = model(tensor[:,:-2])
    _, predicted = torch.max(outputs, 1)
    scores = {}
    for metric_name in metrics_names:
        scores[metric_name] =  metrics[metric_name](predicted, tensor[:,-1].long())[1].item()
        print("{}: {:.3f}".format(metric_name, scores[metric_name]))
    del(tensor, model)

